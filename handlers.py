import asyncio
import html
import logging
import re
import time
import json
import os
from io import BytesIO

import struct
import zlib

from pyrogram import Client, enums, filters
from pyrogram.types import (
    Message,
    InputMediaPhoto,
    ForceReply,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)

import history
from config import Settings

log = logging.getLogger("handlers")

_cooldowns: dict[int, float] = {}
_passive_msg_count: dict[int, int] = {}
_passive_running: dict[int, bool] = {}
_chat_titles: dict[int, str] = {}

# Ожидание ответа на ForceReply-подсказку «сколько сообщений?» после /summary[_all]
# без числа. Ключ — message_id нашей подсказки. Значение — контекст генерации.
_pending_summary: dict[int, dict] = {}
SUMMARY_PROMPT_TTL = 300  # секунд, после которых state считается мёртвым

# Счётчик «новых сообщений с последнего комикса» — только с момента старта бота.
# История на диске предыдущей сессии сюда не попадает (чтобы после рестарта сразу
# не рисовать комикс по старым сообщениям).
_runtime_msg_count: dict[int, int] = {}

# Хранилище веток (threads) для команды /recall. 
# Ключ — message_id ответа бота, значение — список строк (вопросы и ответы в этой ветке).
# TTL ветки и контекст архитектора вынесены в config (recall_thread_ttl, architect_context).
_recall_threads: dict[int, dict] = {}

MIN_MESSAGES_SINCE_COMIC = 5
SUMMARY_MAX_MESSAGES = 500  # верхний предел для /summary N
_TG_MAX_LEN = 3900
MENU_PAGE_SIZE = 10  # сколько кнопок на странице инлайн-меню (/passive — характеры, /comic — стили)


def _make_placeholder_image() -> BytesIO:
    """Генерирует минимальный 100x100 серый PNG как placeholder (без PIL)."""
    width, height = 100, 100
    r, g, b = 60, 60, 60

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + bytes([r, g, b]) * width

    idat = chunk(b"IDAT", zlib.compress(raw_data))
    iend = chunk(b"IEND", b"")

    buf = BytesIO(header + ihdr + idat + iend)
    buf.name = "loading.png"
    return buf


def _split_message_text(text: str, max_len: int = _TG_MAX_LEN) -> list[str]:
    """Split text into chunks that fit within Telegram's message limit."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        quarter = max_len // 4
        split_pos = remaining.rfind('\n\n', 0, max_len)
        if split_pos > quarter:
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos + 2:]
        else:
            split_pos = remaining.rfind('\n', 0, max_len)
            if split_pos > quarter:
                chunks.append(remaining[:split_pos])
                remaining = remaining[split_pos + 1:]
            else:
                split_pos = remaining.rfind(' ', 0, max_len)
                if split_pos > quarter:
                    chunks.append(remaining[:split_pos])
                    remaining = remaining[split_pos + 1:]
                else:
                    chunks.append(remaining[:max_len])
                    remaining = remaining[max_len:]
        remaining = remaining.lstrip()

    return chunks


def _tag(chat_id: int) -> str:
    """Short tag for log messages: [Title (id)]."""
    title = _chat_titles.get(chat_id, "?")
    return f"[{title} ({chat_id})]"


def _strip_command(text: str, client: Client) -> str:
    """Убрать команду/упоминание из начала текста."""
    if text.startswith("/"):
        text = re.sub(r"^/\w+(?:@\w+)?\s*", "", text)
    if client.me:
        text = re.sub(rf"@{client.me.username}\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)@comicditbot\s*", "", text)
    return text.strip()


def _build_choice_keyboard(
    items: list[tuple[str, str]],
    page: int,
    prefix: str,
    select_code: str,
    random_label: str,
) -> InlineKeyboardMarkup:
    """Постраничное инлайн-меню выбора одного элемента (характер пассива, стиль комикса).

    items — список (slug, подпись кнопки), по две кнопки в ряд. Callback-данные:
    «{prefix}:{select_code}:{slug}» — выбор, «{prefix}:p:{page}» — страница,
    «{prefix}:x» — закрыть, «{prefix}:n» — ничего не делать (счётчик страниц, заглушки).
    На первой странице сверху кнопка random_label со slug «random».
    """
    total = len(items)
    pages = max(1, (total + MENU_PAGE_SIZE - 1) // MENU_PAGE_SIZE)
    page = max(0, min(page, pages - 1))

    rows: list[list[InlineKeyboardButton]] = []
    if page == 0:
        rows.append([InlineKeyboardButton(random_label, callback_data=f"{prefix}:{select_code}:random")])

    start = page * MENU_PAGE_SIZE
    chunk = items[start:start + MENU_PAGE_SIZE]
    for i in range(0, len(chunk), 2):
        rows.append([
            InlineKeyboardButton(label, callback_data=f"{prefix}:{select_code}:{slug}")
            for slug, label in chunk[i:i + 2]
        ])

    rows.append([
        InlineKeyboardButton(
            "« Назад" if page > 0 else " ",
            callback_data=f"{prefix}:p:{page - 1}" if page > 0 else f"{prefix}:n",
        ),
        InlineKeyboardButton(f"{page + 1}/{pages}", callback_data=f"{prefix}:n"),
        InlineKeyboardButton(
            "Вперёд »" if page < pages - 1 else " ",
            callback_data=f"{prefix}:p:{page + 1}" if page < pages - 1 else f"{prefix}:n",
        ),
    ])
    rows.append([InlineKeyboardButton("❌ Закрыть", callback_data=f"{prefix}:x")])
    return InlineKeyboardMarkup(rows)


def register_handlers(app: Client, settings: Settings) -> None:
    keyword_pattern = re.escape(settings.trigger_keyword)
    # Ключевое слово в тексте — сразу комикс в случайном стиле. Команда /comic — отдельный
    # хендлер с меню выбора стиля (on_comic_menu), сюда она не попадает.
    keyword_filter = (
        filters.group & filters.text
        & filters.regex(rf"(?i){keyword_pattern}")
        & ~filters.command("comic")
    )

    log.info(
        "Хендлеры зарегистрированы: триггер=«%s», архитект-контекст=%d, кулдаун=%dс, "
        "пассив=%s (порог %d), allowed=%s",
        settings.trigger_keyword, settings.architect_context, settings.cooldown_seconds,
        "ВКЛ" if settings.passive_enabled else "ВЫКЛ",
        settings.passive_min_messages,
        settings.allowed_chat_ids or "все",
    )

    @app.on_disconnect()
    async def on_disconnect(client: Client) -> None:
        log.warning("🔌 Бот потерял соединение с Telegram!")

    @app.on_message(filters.group & filters.text & ~filters.media, group=-1)
    async def on_any_group_message(client: Client, message: Message) -> None:
        chat_id = message.chat.id

        if chat_id not in _chat_titles:
            _chat_titles[chat_id] = message.chat.title or str(chat_id)
            log.info("Новый чат: %s", _tag(chat_id))

        # Не пишем в историю команды бота — они шумят
        text = message.text or ""
        is_command = text.startswith("/")

        if not is_command:
            sender = message.from_user.first_name if message.from_user else "Unknown"
            history.append(chat_id, sender, text)
            _runtime_msg_count[chat_id] = _runtime_msg_count.get(chat_id, 0) + 1
            _passive_msg_count[chat_id] = _passive_msg_count.get(chat_id, 0) + 1

        count = _passive_msg_count.get(chat_id, 0)

        if (
            not is_command
            and settings.passive_enabled
            and count >= settings.passive_min_messages
            and not _passive_running.get(chat_id, False)
        ):
            if not settings.allowed_chat_ids or chat_id in settings.allowed_chat_ids:
                log.info(
                    "%s Пассив: порог достигнут (%d/%d) — запускаю",
                    _tag(chat_id), count, settings.passive_min_messages,
                )
                _passive_running[chat_id] = True
                asyncio.create_task(_run_passive(client, chat_id, settings))
            else:
                log.debug(
                    "%s Пассив: чат не в allowed_chat_ids, пропуск (count=%d)",
                    _tag(chat_id), count,
                )

        message.continue_propagation()

    def _comic_refusal(chat_id: int) -> str | None:
        """Почему сейчас нельзя рисовать комикс (текст для пользователя) или None, если можно."""
        tag = _tag(chat_id)
        remaining = int(settings.cooldown_seconds - (time.time() - _cooldowns.get(chat_id, 0)))
        if remaining > 0:
            log.info("%s Кулдаун: осталось %dс", tag, remaining)
            return f"Подождите {remaining} сек. перед следующим комиксом."

        runtime_count = _runtime_msg_count.get(chat_id, 0)
        if runtime_count < MIN_MESSAGES_SINCE_COMIC:
            log.info("%s Мало новых сообщений с запуска: %d/%d", tag, runtime_count, MIN_MESSAGES_SINCE_COMIC)
            return (
                f"Недостаточно новых сообщений с последнего запуска ({runtime_count}/{MIN_MESSAGES_SINCE_COMIC}). "
                f"Напишите ещё немного."
            )
        return None

    @app.on_message(keyword_filter)
    async def on_keyword(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)

        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            log.info("%s Триггер «%s» проигнорирован: чат не в allowed_chat_ids", tag, settings.trigger_keyword)
            return

        refusal = _comic_refusal(chat_id)
        if refusal:
            await message.reply_text(refusal)
            return

        _cooldowns[chat_id] = time.time()
        log.info("%s Триггер «%s»: рисую комикс в случайном стиле", tag, settings.trigger_keyword)
        await _run_comic(client, chat_id, settings)

    @app.on_message(filters.group & filters.text & filters.command("ask"))
    async def on_architect_question(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)

        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            return

        text = _strip_command(message.text or "", client)
        if not text:
            await message.reply_text("Вы вызвали меня, но не задали вопрос. Напишите, что вас интересует, и я как опытный архитектор помогу.")
            return

        status_msg = await message.reply_text("🤔 Анализирую вопрос и контекст...")

        try:
            conversation = "\n".join(history.get_lines(chat_id, settings.architect_context))

            from gemini_web import generate_architect_response
            t0 = time.time()
            response_text = await generate_architect_response(conversation, text, settings)
            elapsed = time.time() - t0

            chunks = _split_message_text(response_text)
            await status_msg.edit_text(chunks[0], parse_mode=enums.ParseMode.DISABLED)
            for chunk in chunks[1:]:
                await message.reply_text(chunk, parse_mode=enums.ParseMode.DISABLED)

            log.info(
                "%s Ответ архитектора отправлен за %.1fс (%d симв., %d частей)",
                tag, elapsed, len(response_text), len(chunks),
            )

        except Exception as exc:
            log.error("%s ОШИБКА ответа архитектора: %s", tag, exc, exc_info=True)
            await status_msg.edit_text(f"Не удалось получить ответ: {exc}")

    async def _do_recall(client: Client, message: Message, chat_id: int, tag: str, text: str) -> None:
        import random
        placeholders = [
            "🔍 Ищу в секретных архивах ФСБ...",
            "🔍 Взламываю серверы Пентагона для поиска...",
            "🔍 Запрашиваю данные у рептилоидов...",
            "🔍 Поднимаю засекреченные стенограммы...",
            "🔍 Ищу в закромах мирового правительства...",
        ]
        status_msg = await message.reply_text(random.choice(placeholders))

        try:
            import rag
            
            # Очистка старых веток
            now = time.time()
            stale_keys = [k for k, v in _recall_threads.items() if now - v.get("ts", 0) > settings.recall_thread_ttl]
            for k in stale_keys:
                del _recall_threads[k]
                
            thread_context = []
            reply_to = message.reply_to_message
            if reply_to and reply_to.id in _recall_threads:
                thread_data = _recall_threads[reply_to.id]
                thread_context = thread_data.get("history", [])
                log.info("%s recall (ответ): найдена ветка %d, контекст %d сообщений", tag, reply_to.id, len(thread_context))
                
                # Переписываем запрос с учетом ветки
                search_query = await rag.rewrite_query_with_local_llm(text, thread_context, settings)
            else:
                search_query = text
                
            # Запускаем векторный поиск в отдельном потоке, чтобы не блокировать event loop.
            # Берем rag_search_limit кандидатов, отсекая явно непохожие по порогу косинусной
            # дистанции и приоритизируя более свежие сообщения.
            context_messages = await asyncio.to_thread(
                rag.search_context, chat_id, search_query, settings.rag_search_limit,
                settings.rag_max_distance, settings.rag_recency_weight,
            )

            # Строго фильтруем до rag_filter_limit самых релевантных через локальную LLM
            if context_messages:
                context_messages = await rag.filter_context_with_local_llm(
                    search_query, context_messages, settings, max_results=settings.rag_filter_limit
                )

            # Восстанавливаем хронологию: Gemini должен видеть сообщения по времени,
            # чтобы отличать ранние договоренности от более поздних (финальных) решений.
            if context_messages:
                context_messages = rag.sort_messages_chronologically(context_messages)
            
            if not context_messages and not thread_context:
                await status_msg.edit_text("Я не нашел ничего похожего в истории этого чата.")
                return

            if getattr(settings, "rag_use_local_llm_for_response", False):
                t0 = time.time()
                response_text = await rag.generate_rag_response_with_local_llm(
                    context_messages, text, thread_context, settings
                )
                elapsed = time.time() - t0
                source_tag = "LOCAL LLM (Response)"
            else:
                from gemini_web import generate_rag_response
                t0 = time.time()
                response_text = await generate_rag_response(context_messages, text, thread_context, settings)
                elapsed = time.time() - t0
                source_tag = "Gemini"

            chunks = _split_message_text(response_text)
            await status_msg.edit_text(chunks[0], parse_mode=enums.ParseMode.DISABLED)
            
            msg_ids = [status_msg.id]
            for chunk in chunks[1:]:
                msg = await message.reply_text(chunk, parse_mode=enums.ParseMode.DISABLED)
                msg_ids.append(msg.id)
                
            # Сохраняем новую ветку для всех частей ответа
            new_history = thread_context.copy()
            user_name = message.from_user.first_name if message.from_user else "User"
            new_history.append(f"{user_name}: {text}")
            new_history.append(f"Bot: {response_text}")
            
            for mid in msg_ids:
                _recall_threads[mid] = {
                    "ts": time.time(),
                    "history": new_history
                }

            log.info(
                "%s Ответ RAG (%s) отправлен за %.1fс (%d симв., найдено %d сообщений контекста, сохранен тред)",
                tag, source_tag, elapsed, len(response_text), len(context_messages)
            )

        except Exception as exc:
            log.error("%s ОШИБКА RAG: %s", tag, exc, exc_info=True)
            await status_msg.edit_text(f"Не удалось вспомнить: {exc}")

    @app.on_message(filters.group & filters.text & filters.command("recall"))
    async def on_recall_command(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)

        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            return

        text = _strip_command(message.text or "", client)
        if not text:
            await message.reply_text("Напишите, что именно нужно найти в истории. Например: /recall что решили по поводу деплоя?")
            return

        await _do_recall(client, message, chat_id, tag, text)

    @app.on_message(filters.group & filters.text & filters.reply, group=-2)
    async def on_recall_reply(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)

        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            return

        reply_to = message.reply_to_message
        if not reply_to:
            return
            
        # Проверяем, является ли сообщение ответом на предыдущий RAG-результат
        if reply_to.id not in _recall_threads:
            return

        # Если это команда (например, /summary), не перехватываем
        if (message.text or "").startswith("/"):
            return

        text = _strip_command(message.text or "", client)
        if not text:
            return

        await _do_recall(client, message, chat_id, tag, text)
        message.stop_propagation()

    @app.on_message(filters.command("sticker"))
    async def on_sticker_command(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)

        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            return

        photo = None
        if message.photo:
            photo = message.photo
        elif message.document and message.document.mime_type and message.document.mime_type.startswith('image/'):
            photo = message.document
        elif message.reply_to_message:
            if message.reply_to_message.photo:
                photo = message.reply_to_message.photo
            elif message.reply_to_message.document and message.reply_to_message.document.mime_type and message.reply_to_message.document.mime_type.startswith('image/'):
                photo = message.reply_to_message.document

        if not photo:
            await message.reply_text("Пожалуйста, отправьте картинку с командой /sticker или ответьте на картинку этой командой.")
            return

        status_msg = await message.reply_text("⏳ Делаю стикер...")

        try:
            photo_bytes = await client.download_media(photo, in_memory=True)
            if not photo_bytes:
                raise Exception("Не удалось скачать картинку")

            from PIL import Image
            import io

            img = Image.open(io.BytesIO(photo_bytes.getbuffer() if hasattr(photo_bytes, 'getbuffer') else photo_bytes))
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Telegram требует: одна сторона ровно 512, другая <= 512.
            # Считаем размеры явно, чтобы float-погрешность не дала 511.
            w, h = img.size
            if w >= h:
                new_w, new_h = 512, max(1, min(512, round(h * 512 / w)))
            else:
                new_w, new_h = max(1, min(512, round(w * 512 / h))), 512
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            out = io.BytesIO()
            img.save(out, format="WEBP")
            webp_bytes = out.getvalue()

            import httpx
            import json

            bot_username = client.me.username
            user_id = message.from_user.id if message.from_user else None
            if not user_id:
                raise Exception("Не удалось определить ваш ID пользователя.")
                
            # Загружаем или создаем информацию об общем публичном паке
            pack_info_file = "public_sticker_pack.json"
            pack_owner_id = None
            pack_name = f"public_comicbot_by_{bot_username}"
            
            if os.path.exists(pack_info_file):
                try:
                    with open(pack_info_file, "r") as f:
                        pack_info = json.load(f)
                        pack_owner_id = pack_info.get("owner_id")
                except Exception as e:
                    log.warning("Не удалось прочитать %s: %s", pack_info_file, e)
            
            if not pack_owner_id:
                pack_owner_id = user_id
                try:
                    with open(pack_info_file, "w") as f:
                        json.dump({"owner_id": pack_owner_id, "pack_name": pack_name}, f)
                except Exception as e:
                    log.warning("Не удалось сохранить %s: %s", pack_info_file, e)

            token = settings.telegram_bot_token

            async with httpx.AsyncClient() as http_client:
                files = {
                    'file': ('sticker.webp', webp_bytes, 'image/webp')
                }
                data = {
                    'user_id': str(pack_owner_id),
                    'name': pack_name,
                    'sticker': json.dumps({
                        'sticker': 'attach://file',
                        'emoji_list': ['✨']
                    })
                }

                resp = await http_client.post(
                    f"https://api.telegram.org/bot{token}/addStickerToSet",
                    data=data,
                    files=files,
                    timeout=30.0
                )
                res_json = resp.json()

                if not res_json.get('ok'):
                    if 'STICKERSET_INVALID' in res_json.get('description', ''):
                        files = {
                            'file': ('sticker.webp', webp_bytes, 'image/webp')
                        }
                        data = {
                            'user_id': str(pack_owner_id),
                            'name': pack_name,
                            'title': "Общие стикеры ComicBot",
                            'stickers': json.dumps([{
                                'sticker': 'attach://file',
                                'emoji_list': ['✨']
                            }]),
                            'sticker_format': 'static'
                        }
                        resp2 = await http_client.post(
                            f"https://api.telegram.org/bot{token}/createNewStickerSet",
                            data=data,
                            files=files,
                            timeout=30.0
                        )
                        res2_json = resp2.json()
                        if not res2_json.get('ok'):
                            raise Exception(f"Не удалось создать пак: {res2_json.get('description')}")
                    else:
                        raise Exception(f"Не удалось добавить стикер: {res_json.get('description')}")

                pack_resp = await http_client.get(f"https://api.telegram.org/bot{token}/getStickerSet?name={pack_name}")
                pack_json = pack_resp.json()
                if pack_json.get('ok'):
                    stickers = pack_json['result']['stickers']
                    last_sticker_file_id = stickers[-1]['file_id']
                    await client.send_sticker(chat_id, last_sticker_file_id)
                    await status_msg.delete()
                else:
                    await status_msg.edit_text("✅ Стикер добавлен в ваш пак, но не удалось его отправить.")

            log.info("%s Стикер успешно создан и добавлен в пак %s", tag, pack_name)

        except Exception as exc:
            log.error("%s ОШИБКА создания стикера: %s", tag, exc, exc_info=True)
            await status_msg.edit_text(f"Не удалось сделать стикер: {exc}")

    async def _do_summary(
        client: Client,
        message: Message,
        chat_id: int,
        tag: str,
        cmd: str,
        n: int,
        send_dm: bool,
    ) -> None:
        lines = history.get_lines(chat_id, n)
        if not lines:
            await message.reply_text("В истории этого чата пока нет сообщений для сводки.")
            return

        log.info("%s /%s запрос: N=%d, собрано=%d, DM=%s", tag, cmd, n, len(lines), send_dm)
        status_msg = await message.reply_text(f"📝 Собираю сводку по {len(lines)} сообщениям...")

        try:
            conversation = "\n".join(lines)
            from gemini_web import generate_summary
            t0 = time.time()
            summary = await generate_summary(conversation, len(lines), settings)
            elapsed = time.time() - t0

            safe_title = html.escape(str(_chat_titles.get(chat_id, chat_id)))
            header = f"📋 <b>Сводка по последним {len(lines)} сообщениям</b>\n<i>{safe_title}</i>\n\n"
            full = header + summary
            chunks = _split_message_text(full)

            delivered_dm = False
            dm_error: Exception | None = None
            if send_dm and message.from_user:
                try:
                    user_id = message.from_user.id
                    for chunk in chunks:
                        await client.send_message(user_id, chunk, parse_mode=enums.ParseMode.HTML)
                    delivered_dm = True
                    await status_msg.edit_text("✅ Сводка отправлена вам в личные сообщения.")
                except Exception as dm_exc:
                    dm_error = dm_exc
                    log.warning("%s Не удалось отправить сводку в ЛС (%s), публикую в чат", tag, dm_exc)

            if not delivered_dm:
                if send_dm and dm_error is not None:
                    bot_username = client.me.username if client.me else "бота"
                    note = (
                        "⚠️ Не получилось прислать сводку в ЛС — Telegram не разрешает боту "
                        f"писать первым. Открой @{bot_username}, нажми <b>Start</b>, и со следующего "
                        "раза сводка будет приходить в личку. Пока публикую здесь.\n\n"
                    )
                    chunks[0] = note + chunks[0]
                await status_msg.edit_text(chunks[0], parse_mode=enums.ParseMode.HTML)
                for chunk in chunks[1:]:
                    await message.reply_text(chunk, parse_mode=enums.ParseMode.HTML)

            log.info(
                "%s /%s готово за %.1fс (%d симв., %d частей, DM=%s)",
                tag, cmd, elapsed, len(summary), len(chunks), delivered_dm,
            )

        except Exception as exc:
            log.error("%s ОШИБКА сводки: %s", tag, exc, exc_info=True)
            try:
                await status_msg.edit_text(f"Не удалось сделать сводку: {exc}")
            except Exception:
                pass

    @app.on_message(filters.group & filters.text & filters.command([
        "summary", "summary20", "summary50", "summary100",
        "summary_all", "summary20_all", "summary50_all", "summary100_all",
    ]))
    async def on_summary(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)

        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            return

        text = message.text or ""
        cmd_match = re.match(r"^/(\w+)(?:@\w+)?\s*(.*)$", text, flags=re.DOTALL)
        cmd = cmd_match.group(1).lower() if cmd_match else "summary"
        rest = cmd_match.group(2).strip() if cmd_match else ""

        send_dm = not cmd.endswith("_all")

        m_preset = re.match(r"^summary(\d+)?(?:_all)?$", cmd)
        preset_n = m_preset.group(1) if m_preset and m_preset.group(1) else None

        n: int | None = None
        if preset_n:
            n = int(preset_n)
        elif rest:
            m = re.match(r"(\d+)", rest)
            if m:
                n = int(m.group(1))

        if n is None:
            # /summary или /summary_all без числа — спросим пользователя через ForceReply
            prompt = await message.reply_text(
                f"📝 Сколько последних сообщений включить в сводку? Введите число от 5 до {SUMMARY_MAX_MESSAGES}.",
                reply_markup=ForceReply(selective=True),
            )
            # чистим просроченные pending, заодно экономим память
            now = time.time()
            for stale_id in [k for k, v in _pending_summary.items() if now - v.get("ts", 0) > SUMMARY_PROMPT_TTL]:
                _pending_summary.pop(stale_id, None)

            _pending_summary[prompt.id] = {
                "chat_id": chat_id,
                "user_id": message.from_user.id if message.from_user else None,
                "send_dm": send_dm,
                "cmd": cmd,
                "ts": now,
            }
            log.info(
                "%s /%s без N — отправлен ForceReply (prompt_id=%d, DM=%s)",
                tag, cmd, prompt.id, send_dm,
            )
            return

        n = max(5, min(n, SUMMARY_MAX_MESSAGES))
        await _do_summary(client, message, chat_id, tag, cmd, n, send_dm)

    @app.on_message(filters.group & filters.text & filters.reply, group=-2)
    async def on_summary_reply(client: Client, message: Message) -> None:
        reply_to = message.reply_to_message
        if not reply_to:
            return
        state = _pending_summary.get(reply_to.id)
        if not state:
            return
        if state["chat_id"] != message.chat.id:
            return
        if time.time() - state.get("ts", 0) > SUMMARY_PROMPT_TTL:
            _pending_summary.pop(reply_to.id, None)
            return
        if (
            state.get("user_id")
            and message.from_user
            and message.from_user.id != state["user_id"]
        ):
            return  # чужой ответ — пусть пропагандирует дальше

        _pending_summary.pop(reply_to.id, None)

        m = re.match(r"^\s*(\d+)", message.text or "")
        if not m:
            await message.reply_text("Нужно прислать число. Вызовите команду заново.")
            message.stop_propagation()
            return

        n = max(5, min(int(m.group(1)), SUMMARY_MAX_MESSAGES))
        chat_id = message.chat.id
        tag = _tag(chat_id)
        cmd = state.get("cmd", "summary")
        send_dm = bool(state.get("send_dm"))

        await _do_summary(client, message, chat_id, tag, cmd, n, send_dm)
        message.stop_propagation()

    def _build_character_keyboard(page: int) -> InlineKeyboardMarkup:
        from gemini_web import SPONTANEOUS_CHARACTERS
        return _build_choice_keyboard(
            [(slug, label) for slug, label, _desc in SPONTANEOUS_CHARACTERS],
            page, prefix="p", select_code="c", random_label="🎲 Случайный характер",
        )

    def _build_style_keyboard(page: int) -> InlineKeyboardMarkup:
        return _build_choice_keyboard(
            [(style["name"], style["title"]) for style in settings.unique_comic_styles()],
            page, prefix="c", select_code="s", random_label="🎲 Случайный стиль",
        )

    def _build_humor_keyboard(slug: str) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = [
            [InlineKeyboardButton("🎲 Рандомный уровень", callback_data=f"p:l:{slug}:0")]
        ]
        
        row1, row2 = [], []
        for i in range(1, 6):
            row1.append(InlineKeyboardButton(str(i), callback_data=f"p:l:{slug}:{i}"))
        for i in range(6, 11):
            row2.append(InlineKeyboardButton(str(i), callback_data=f"p:l:{slug}:{i}"))
            
        rows.append(row1)
        rows.append(row2)
        
        rows.append([InlineKeyboardButton("« Назад к характерам", callback_data="p:p:0")])
        rows.append([InlineKeyboardButton("❌ Закрыть", callback_data="p:x")])
        return InlineKeyboardMarkup(rows)

    @app.on_message(filters.text & filters.command("passive"))
    async def on_passive_menu(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)
        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            log.info("%s /passive проигнорирован: чат не в allowed_chat_ids", tag)
            return
        await message.reply_text(
            "🎭 Выбери, кто сегодня в кадре:",
            reply_markup=_build_character_keyboard(0),
        )

    @app.on_callback_query(filters.regex(r"^p:"))
    async def on_passive_callback(client: Client, callback: CallbackQuery) -> None:
        msg = callback.message
        chat_id = msg.chat.id if msg else None
        data = callback.data or ""

        if data == "p:x":
            if msg is not None:
                try:
                    await msg.edit_text("🎭 Меню закрыто.")
                except Exception:
                    pass
            await callback.answer()
            return

        if data.startswith("p:p:"):
            parts = data.split(":")
            try:
                new_page = int(parts[2])
            except (ValueError, IndexError):
                await callback.answer()
                return
            if msg is not None:
                try:
                    await msg.edit_text(
                        "🎭 Выбери, кто сегодня в кадре:",
                        reply_markup=_build_character_keyboard(new_page)
                    )
                except Exception:
                    pass
            await callback.answer()
            return

        if data.startswith("p:c:"):
            parts = data.split(":")
            slug = parts[2]
            if msg is not None:
                try:
                    await msg.edit_text(
                        "🎚 Выбери уровень чернухи (1 - лайт, 10 - полная жесть):",
                        reply_markup=_build_humor_keyboard(slug)
                    )
                except Exception:
                    pass
            await callback.answer()
            return

        if data.startswith("p:l:"):
            parts = data.split(":")
            slug = parts[2]
            humor = int(parts[3])
            tag = _tag(chat_id) if chat_id is not None else "[?]"

            if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
                await callback.answer("Этот чат не в списке разрешённых.", show_alert=True)
                return
            if _passive_running.get(chat_id, False):
                await callback.answer(
                    "Уже готовлю предыдущий пассив, подожди немного…",
                    show_alert=True,
                )
                return

            from gemini_web import SPONTANEOUS_CHARACTERS
            label = "🎲 Случайный характер"
            if slug != "random":
                found = next((c for c in SPONTANEOUS_CHARACTERS if c[0] == slug), None)
                if found is None:
                    await callback.answer("Не знаю такого характера.", show_alert=True)
                    return
                label = found[1]

            humor_label = "Рандом" if humor == 0 else str(humor)
            await callback.answer(f"Готовлю: {label} (Чернуха: {humor_label})")
            if msg is not None:
                try:
                    await msg.edit_text(f"⏳ Готовлю пассив: {label} (Чернуха: {humor_label})…")
                except Exception:
                    pass

            force = None if slug == "random" else slug
            humor_level = None if humor == 0 else humor
            log.info("%s /passive: запуск через инлайн-меню (force=%s, humor=%s)", tag, force, humor_level)
            _passive_running[chat_id] = True
            asyncio.create_task(_run_passive(client, chat_id, settings, force, humor_level))
            return

        await callback.answer()

    @app.on_message(filters.group & filters.text & filters.command("comic"))
    async def on_comic_menu(client: Client, message: Message) -> None:
        chat_id = message.chat.id
        tag = _tag(chat_id)
        if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
            log.info("%s /comic проигнорирован: чат не в allowed_chat_ids", tag)
            return
        # Не показываем меню, если после выбора всё равно откажем (кулдаун, мало сообщений)
        refusal = _comic_refusal(chat_id)
        if refusal:
            await message.reply_text(refusal)
            return
        await message.reply_text(
            "🎨 Выбери стиль комикса:",
            reply_markup=_build_style_keyboard(0),
        )

    @app.on_callback_query(filters.regex(r"^c:"))
    async def on_comic_callback(client: Client, callback: CallbackQuery) -> None:
        msg = callback.message
        chat_id = msg.chat.id if msg else None
        data = callback.data or ""

        if data == "c:x":
            if msg is not None:
                try:
                    await msg.edit_text("🎨 Меню закрыто.")
                except Exception:
                    pass
            await callback.answer()
            return

        if data.startswith("c:p:"):
            try:
                new_page = int(data.split(":")[2])
            except (ValueError, IndexError):
                await callback.answer()
                return
            if msg is not None:
                try:
                    await msg.edit_text(
                        "🎨 Выбери стиль комикса:",
                        reply_markup=_build_style_keyboard(new_page),
                    )
                except Exception:
                    pass
            await callback.answer()
            return

        if data.startswith("c:s:") and chat_id is not None:
            slug = data.split(":", 2)[2]
            tag = _tag(chat_id)

            if settings.allowed_chat_ids and chat_id not in settings.allowed_chat_ids:
                await callback.answer("Этот чат не в списке разрешённых.", show_alert=True)
                return
            refusal = _comic_refusal(chat_id)
            if refusal:
                await callback.answer(refusal, show_alert=True)
                return

            force: str | None = None
            title = "🎲 Случайный стиль"
            if slug != "random":
                style = next((s for s in settings.unique_comic_styles() if s["name"] == slug), None)
                if style is None:
                    await callback.answer("Не знаю такого стиля.", show_alert=True)
                    return
                force, title = style["name"], style["title"]

            _cooldowns[chat_id] = time.time()
            await callback.answer(f"Рисую: {title}")
            # Меню больше не нужно — убираем, чтобы в чате остался только placeholder → комикс
            try:
                await msg.delete()
            except Exception:
                try:
                    await msg.edit_text(f"⏳ Рисую комикс: {title}…")
                except Exception:
                    pass

            log.info("%s /comic: запуск через инлайн-меню (style=%s)", tag, force or "random")
            await _run_comic(client, chat_id, settings, force, title if force else None)
            return

        await callback.answer()


async def _run_comic(
    client: Client, chat_id: int, settings: Settings,
    force_style: str | None = None, style_title: str | None = None,
) -> None:
    """Нарисовать комикс по последним сообщениям чата и отправить его.

    Общий код для ключевого слова (случайный стиль) и меню /comic: force_style — name стиля
    из settings.comic_prompts, style_title — его русское название для подписи placeholder.
    Проверки (allowed_chat_ids, кулдаун, минимум сообщений) и установка кулдауна — на
    вызывающей стороне; при ошибке генерации кулдаун сбрасывается здесь.
    """
    tag = _tag(chat_id)
    caption = f"Рисую комикс: {style_title}… 🎨" if style_title else "Рисую комикс... 🎨"
    status_msg = await client.send_photo(chat_id, photo=_make_placeholder_image(), caption=caption)

    try:
        recent = history.get_lines(chat_id, settings.message_count)
        conversation = "\n".join(recent)
        log.info(
            "%s Комикс: стиль=%s, отправляю=%d, символов=%d",
            tag, force_style or "random", len(recent), len(conversation),
        )

        from gemini_web import generate_comic_image_web
        t0 = time.time()
        image_bytes = await generate_comic_image_web(conversation, settings, force_style=force_style)
        elapsed = time.time() - t0

        photo = BytesIO(image_bytes)
        photo.name = "comic.png"
        try:
            await status_msg.edit_media(media=InputMediaPhoto(media=photo))
        except Exception as edit_err:
            log.warning("edit_media failed (%s), fallback send+delete", edit_err)
            photo.seek(0)
            await client.send_photo(chat_id, photo=photo)
            try:
                await status_msg.delete()
            except Exception:
                pass

        _runtime_msg_count[chat_id] = 0
        _passive_msg_count[chat_id] = 0

        log.info(
            "%s Комикс отправлен за %.1fс (%d КБ). Счётчики сброшены.",
            tag, elapsed, len(image_bytes) // 1024,
        )

    except Exception as exc:
        log.error("%s ОШИБКА генерации комикса: %s", tag, exc, exc_info=True)
        error_text = f"Не удалось создать комикс: {exc}"
        try:
            await status_msg.edit_media(
                media=InputMediaPhoto(media=_make_placeholder_image(), caption=error_text[:1024])
            )
        except Exception:
            try:
                await status_msg.delete()
            except Exception:
                pass
            await client.send_message(chat_id, error_text)
        _cooldowns[chat_id] = 0
        log.info("%s Кулдаун сброшен из-за ошибки", tag)


async def _run_passive(
    client: Client, chat_id: int, settings: Settings, force_character: str | None = None, humor_level: int | None = None,
) -> None:
    """Fire-and-forget passive action for a single chat."""
    tag = _tag(chat_id)
    try:
        buf_size = history.size(chat_id)
        msg_count = _passive_msg_count.get(chat_id, 0)

        if buf_size < MIN_MESSAGES_SINCE_COMIC:
            log.info("%s Пассив: отмена, мало сообщений в истории (%d/%d)", tag, buf_size, MIN_MESSAGES_SINCE_COMIC)
            return

        log.info(
            "%s Пассив: старт (msgs_since_last=%d, история=%d, force=%s, humor=%s)",
            tag, msg_count, buf_size, force_character, humor_level,
        )

        from gemini_web import generate_spontaneous_content

        recent = history.get_lines(chat_id, settings.passive_min_messages)
        conversation = "\n".join(recent)
        log.info("%s Пассив: отправляю %d сообщений (%d символов)", tag, len(recent), len(conversation))

        t0 = time.time()
        text, image_bytes = await generate_spontaneous_content(
            conversation, settings, force_character=force_character, humor_level=humor_level,
        )
        elapsed = time.time() - t0

        if image_bytes:
            photo = BytesIO(image_bytes)
            photo.name = "spontaneous.png"
            caption = text[:1024] if text else None
            await client.send_photo(chat_id, photo=photo, caption=caption)
            log.info(
                "%s Пассив: фото отправлено за %.1fс (%d КБ, caption=%d симв.)",
                tag, elapsed, len(image_bytes) // 1024,
                len(caption) if caption else 0,
            )
        elif text:
            await client.send_message(chat_id, text)
            log.info("%s Пассив: текст отправлен за %.1fс (%d символов)", tag, elapsed, len(text))

        _passive_msg_count[chat_id] = 0
        log.info("%s Пассив: готово! Счётчик сброшен.", tag)

    except Exception as exc:
        log.error("%s Пассив: ОШИБКА: %s", tag, exc, exc_info=True)
    finally:
        _passive_running[chat_id] = False
        log.debug("%s Пассив: блокировка снята", tag)
