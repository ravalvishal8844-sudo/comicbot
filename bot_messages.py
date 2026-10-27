"""
Журнал исходящих сообщений бота — чтобы из Web UI можно было удалить любое из них.

Бот не получает апдейты о собственных сообщениях, поэтому install() оборачивает
методы клиента send_*/edit_*/delete_messages и записывает, что и куда ушло.
Журнал на диске: data/bot_messages.json, последние MAX_PER_CHAT записей на чат:
    {"<chat_id>": [{"id": 17049, "ts": "2026-10-09T16:17:21", "kind": "text", "preview": "..."}]}

Для сообщений, отправленных до перезапуска (их в журнале нет), есть scan():
ищет сообщения бота напрямую в Telegram через getMessages. В обычных группах
id сообщений лежат в общем пространстве бота, в супергруппах — своём на чат.
"""
import asyncio
import inspect
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from pyrogram import Client
from pyrogram.errors import FloodWait
from pyrogram.types import Message

log = logging.getLogger("bot_messages")

PATH = Path(__file__).parent / "data" / "bot_messages.json"
MAX_PER_CHAT = 100
PREVIEW_LEN = 200

_SEND_METHODS = (
    "send_message", "send_photo", "send_sticker", "send_document", "send_animation",
    "send_video", "send_audio", "send_voice", "send_media_group", "copy_message", "forward_messages",
)
_EDIT_METHODS = ("edit_message_text", "edit_message_caption", "edit_message_media")

_KIND_ATTRS = ("photo", "sticker", "document", "animation", "video", "audio", "voice")
KIND_LABELS = {
    "text": "текст", "photo": "фото", "sticker": "стикер", "document": "файл",
    "animation": "гифка", "video": "видео", "audio": "аудио", "voice": "голосовое", "other": "другое",
}

_log: dict[int, list[dict]] = {}
_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not PATH.exists():
        return
    try:
        with PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        for chat_id, entries in raw.items():
            _log[int(chat_id)] = list(entries)[-MAX_PER_CHAT:]
        log.info("Журнал сообщений бота: %d чатов из %s", len(_log), PATH.name)
    except Exception as exc:
        log.error("Не удалось прочитать %s: %s", PATH.name, exc)


def _save() -> None:
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in _log.items()}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, PATH)
    except Exception as exc:
        log.error("Не удалось записать %s: %s", PATH.name, exc)


def describe(m: Message) -> dict:
    """Запись журнала по сообщению pyrogram: id, локальное время, тип, начало текста."""
    kind = next((a for a in _KIND_ATTRS if getattr(m, a, None)), "text" if m.text else "other")
    preview = " ".join((m.text or m.caption or "").split())[:PREVIEW_LEN]
    try:
        ts = m.date.astimezone().isoformat(timespec="seconds")
    except Exception:
        ts = datetime.now().isoformat(timespec="seconds")
    return {"id": m.id, "ts": ts, "kind": kind, "preview": preview}


def record(m: Message) -> None:
    """Добавить сообщение в журнал (или обновить запись, если оно уже там — после edit_*)."""
    if not isinstance(m, Message) or not m.chat or not m.id:
        return
    _load()
    entries = _log.setdefault(m.chat.id, [])
    entry = describe(m)
    for i, old in enumerate(entries):
        if old["id"] == entry["id"]:
            entries[i] = entry
            break
    else:
        entries.append(entry)
        entries.sort(key=lambda e: e["id"])
        del entries[:-MAX_PER_CHAT]
    _save()


def forget(chat_id: int, msg_ids) -> None:
    _load()
    ids = set(msg_ids) if isinstance(msg_ids, (list, tuple, set)) else {msg_ids}
    entries = _log.get(chat_id)
    if not entries:
        return
    _log[chat_id] = [e for e in entries if e["id"] not in ids]
    _save()


def recent(chat_id: int, n: int = 20) -> list[dict]:
    """Последние n записей журнала чата, новые первыми, с человеческой меткой типа."""
    _load()
    out = []
    for e in list(_log.get(chat_id, []))[-n:][::-1]:
        out.append({**e, "kind_label": KIND_LABELS.get(e["kind"], e["kind"])})
    return out


def _record_result(result) -> None:
    try:
        for m in (result if isinstance(result, list) else [result]):
            record(m)
    except Exception as exc:
        log.warning("Не удалось записать сообщение в журнал: %s", exc)


def install(app: Client) -> None:
    """Обернуть методы отправки/редактирования/удаления клиента, чтобы журнал вёлся сам.

    message.reply_text(), status_msg.edit_media(), msg.delete() и т.п. внутри pyrogram
    зовут те же методы клиента, так что обёртка на экземпляре ловит и их.
    """
    for name in _SEND_METHODS + _EDIT_METHODS:
        orig = getattr(app, name, None)
        if orig is None:
            continue

        def make(orig):
            async def wrapper(*args, **kwargs):
                result = await orig(*args, **kwargs)
                _record_result(result)
                return result
            wrapper.__name__ = orig.__name__
            return wrapper

        setattr(app, name, make(orig))

    orig_delete = app.delete_messages
    sig = inspect.signature(orig_delete)

    async def delete_wrapper(*args, **kwargs):
        result = await orig_delete(*args, **kwargs)
        try:
            bound = sig.bind(*args, **kwargs)
            chat_id, ids = bound.arguments.get("chat_id"), bound.arguments.get("message_ids")
            if isinstance(chat_id, int) and ids is not None:
                forget(chat_id, list(ids) if isinstance(ids, (list, tuple, set, range)) else ids)
        except Exception as exc:
            log.warning("Не удалось убрать сообщение из журнала: %s", exc)
        return result

    app.delete_messages = delete_wrapper
    log.info("Журнал сообщений бота подключён (%s)", PATH.name)


# ---------------------------------------------------------------------------
# Поиск сообщений бота в Telegram (для тех, что отправлены до перезапуска)
# ---------------------------------------------------------------------------

_BATCH = 200          # лимит getMessages за один вызов
_PAUSE = 1.0          # пауза между вызовами: getMessages быстро упирается в флуд-лимит
_MAX_FLOOD_WAIT = 60  # дольше не ждём — отдаём ошибку
_SCAN_BATCHES = 20    # сколько окон по _BATCH просматриваем назад (4000 id)


def _is_channel(chat_id: int) -> bool:
    return str(chat_id).startswith("-100")


async def _get_batch(app: Client, chat_id: int, ids: range) -> list[Message]:
    while True:
        try:
            msgs = await app.get_messages(chat_id, list(ids), replies=0)
            break
        except FloodWait as e:
            if e.value > _MAX_FLOOD_WAIT:
                raise RuntimeError(f"Telegram просит подождать {e.value} с — попробуйте позже") from e
            log.info("Скан сообщений бота: FloodWait %d с", e.value)
            await asyncio.sleep(e.value + 1)
    await asyncio.sleep(_PAUSE)
    return [m for m in msgs if m and not getattr(m, "empty", False)]


def _known_lower_bound(chat_id: int) -> int:
    """Самый большой известный id в том же пространстве id, что и чат (чтобы не искать с нуля)."""
    _load()
    best = 0
    for cid, entries in _log.items():
        same_space = (cid == chat_id) if _is_channel(chat_id) else (not _is_channel(cid))
        if same_space and entries:
            best = max(best, entries[-1]["id"])
    return best


async def _find_top_id(app: Client, chat_id: int) -> int:
    """Верхняя граница id: галопом от известного id вверх, пока окно не окажется пустым,
    затем бинарный поиск внутри последнего непустого интервала."""
    lo = _known_lower_bound(chat_id)
    step = _BATCH
    probe = lo + 1
    while True:
        found = await _get_batch(app, chat_id, range(probe, probe + _BATCH))
        if not found:
            hi = probe
            break
        lo = max(m.id for m in found)
        probe = lo + step
        step = min(step * 2, 1 << 24)
    while hi - lo > _BATCH:
        mid = (lo + hi) // 2
        found = await _get_batch(app, chat_id, range(mid, mid + _BATCH))
        if found:
            lo = max(m.id for m in found)
        else:
            hi = mid
    return hi


async def scan(app: Client, chat_id: int, want: int = 10) -> list[dict]:
    """Найти последние want сообщений бота в чате через Telegram и положить их в журнал."""
    me = await app.get_me()
    _load()
    # Ниже самого старого известного id не спускаемся: тот участок уже просмотрен прошлым сканом
    # или записан при отправке, а журнал и так отдаёт всё, что знает.
    stop_at = min((e["id"] for e in _log.get(chat_id, [])), default=0)
    top = await _find_top_id(app, chat_id)
    found: list[Message] = []
    cur, batches = top, 0
    while cur > stop_at and len(found) < want and batches < _SCAN_BATCHES:
        lo = max(1, cur - _BATCH + 1)
        for m in await _get_batch(app, chat_id, range(lo, cur + 1)):
            if m.chat and m.chat.id == chat_id and m.from_user and m.from_user.id == me.id:
                found.append(m)
        cur, batches = lo - 1, batches + 1
    for m in found:
        record(m)
    log.info("Скан сообщений бота в %d: верхняя граница id %d, найдено %d за %d окон",
             chat_id, top, len(found), batches)
    return recent(chat_id, want)
