"""
Постоянная история чатов: JSONL на чат, append-only.

Формат файла data/history/{chat_id}.jsonl — одна запись на строку:
    {"ts": "2026-04-14T12:34:56", "name": "Лёша", "text": "привет"}

При старте последние MAX_CACHE записей каждого чата подгружаются в RAM-кеш лениво
(по первому обращению). Добавление идёт и в файл, и в кеш, чтобы не читать с диска
на каждый тик.
"""
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
import rag

log = logging.getLogger("history")

HISTORY_DIR = Path(__file__).parent / "data" / "history"
MAX_CACHE = 500  # сколько последних записей держим в памяти на чат

_cache: dict[int, deque[dict]] = {}
_loaded: set[int] = set()
_file_locks: dict[int, Lock] = {}


def _lock_for(chat_id: int) -> Lock:
    lock = _file_locks.get(chat_id)
    if lock is None:
        lock = Lock()
        _file_locks[chat_id] = lock
    return lock


def _path_for(chat_id: int) -> Path:
    return HISTORY_DIR / f"{chat_id}.jsonl"


def _ensure_loaded(chat_id: int) -> deque[dict]:
    if chat_id in _loaded:
        return _cache[chat_id]

    dq: deque[dict] = deque(maxlen=MAX_CACHE)
    path = _path_for(chat_id)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-MAX_CACHE:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    dq.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Битая строка в истории %s: %r", path.name, line[:120])
            log.info("История %d: подгружено %d записей из %s", chat_id, len(dq), path.name)
        except Exception as exc:
            log.error("Не удалось прочитать историю %s: %s", path.name, exc)

    _cache[chat_id] = dq
    _loaded.add(chat_id)
    return dq


def append(chat_id: int, name: str, text: str, ts: str | None = None) -> None:
    """Добавить сообщение в историю чата (и в кеш, и в файл).

    Если передан ts (ISO-8601), используется он; иначе — datetime.now().
    """
    record = {
        "ts": ts or datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "text": text,
    }
    dq = _ensure_loaded(chat_id)
    dq.append(record)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(chat_id)
    try:
        with _lock_for(chat_id):
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.error("Не удалось записать историю %s: %s", path.name, exc)
        
    # Добавляем в RAG базу для поиска
    try:
        rag.add_message(chat_id, name, text, record["ts"])
    except Exception as exc:
        log.error("Не удалось добавить сообщение в RAG: %s", exc)


def get_lines(chat_id: int, n: int) -> list[str]:
    """
    Последние n записей как строки «Имя: текст» — для подстановки в промпты.
    Если n больше, чем в кеше, — читаем хвост файла.
    """
    dq = _ensure_loaded(chat_id)
    if n <= len(dq):
        records = list(dq)[-n:]
    else:
        records = _read_tail(chat_id, n)
    return [f"{r.get('name', '?')}: {r.get('text', '')}" for r in records]


def get_raw(chat_id: int, n: int) -> list[dict]:
    """Последние n записей как список словарей — для Web UI."""
    dq = _ensure_loaded(chat_id)
    if n <= len(dq):
        return list(dq)[-n:]
    return _read_tail(chat_id, n)


def size(chat_id: int) -> int:
    """Сколько записей в кеше (ориентировочный размер буфера)."""
    return len(_ensure_loaded(chat_id))


def file_size(chat_id: int) -> int:
    """Общее число записей в файле истории (может быть больше кеша)."""
    path = _path_for(chat_id)
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _read_tail(chat_id: int, n: int) -> list[dict]:
    path = _path_for(chat_id)
    if not path.exists():
        return list(_cache.get(chat_id, deque()))[-n:]
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        log.error("Не удалось прочитать хвост %s: %s", path.name, exc)
        return list(_cache.get(chat_id, deque()))[-n:]

    out: list[dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def known_chats() -> list[int]:
    """Все чаты, для которых на диске есть файл истории."""
    if not HISTORY_DIR.exists():
        return []
    ids: list[int] = []
    for p in HISTORY_DIR.glob("*.jsonl"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return ids
