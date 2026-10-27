import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
for _noisy in ("pyrogram", "httpx", "httpcore", "hpack"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from pyrogram import Client

from config import load_settings
from handlers import register_handlers
import bot_messages
import rag

log = logging.getLogger("bot")

settings = load_settings()

# Инициализируем базу данных RAG и переносим старую историю
rag.init_db(settings.embedding_model, settings.rag_min_message_length)
rag.backfill_from_jsonl()

app = Client(
    name="comicbot",
    api_id=settings.telegram_api_id,
    api_hash=settings.telegram_api_hash,
    bot_token=settings.telegram_bot_token,
)

# Журнал исходящих сообщений бота (для удаления из Web UI) — до регистрации хендлеров
bot_messages.install(app)
register_handlers(app, settings)

log.info(
    "ComicBot запущен. Триггер: «%s», пассив: %s (порог %d), чаты: %s",
    settings.trigger_keyword,
    "ВКЛ" if settings.passive_enabled else "ВЫКЛ",
    settings.passive_min_messages,
    settings.allowed_chat_ids or "все",
)

if __name__ == "__main__":
    import uvicorn
    from web_ui import fastapp
    
    async def run_server():
        log.info("🌐 Web UI запущен на http://localhost:8091")
        config = uvicorn.Config(fastapp, host="0.0.0.0", port=8091, loop="asyncio", log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    loop = asyncio.get_event_loop()

    def global_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        log.error("🔥 Необработанное исключение в event loop: %s", msg)

    loop.set_exception_handler(global_exception_handler)

    async def main():
        loop.create_task(run_server())
        
        await app.start()
        
        async def health_check():
            while True:
                t0 = asyncio.get_event_loop().time()
                await asyncio.sleep(60)
                delay = asyncio.get_event_loop().time() - t0 - 60
                if delay > 5.0:
                    log.warning("⚠️ Event loop был заблокирован на %.1f секунд!", delay)
                try:
                    if not app.is_connected:
                        log.warning("⚠️ Бот не подключен к Telegram! (app.is_connected == False)")
                except Exception as e:
                    log.error("⚠️ Ошибка health check: %s", e)
                    
        loop.create_task(health_check())
        
        from pyrogram.types import (
            BotCommand,
            BotCommandScopeAllPrivateChats,
            BotCommandScopeAllGroupChats,
            BotCommandScopeDefault,
        )
        commands = [
            BotCommand("comic", "Комикс: выбрать стиль и нарисовать"),
            BotCommand("sticker", "Сделать стикер из картинки"),
            BotCommand("ask", "Задать вопрос опытному архитектору"),
            BotCommand("recall", "Вспомнить что-то из старой истории чата"),
            BotCommand("passive", "Меню пассива: выбрать характер и запустить"),
            BotCommand("summary", "Сводка по N сообщениям: /summary 75 (в ЛС)"),
            BotCommand("summary20", "Быстрая сводка 20 сообщений (в ЛС)"),
            BotCommand("summary50", "Быстрая сводка 50 сообщений (в ЛС)"),
            BotCommand("summary100", "Быстрая сводка 100 сообщений (в ЛС)"),
            BotCommand("summary_all", "Сводка по N сообщениям: /summary_all 75 (в чат)"),
            BotCommand("summary20_all", "Быстрая сводка 20 сообщений (в чат)"),
            BotCommand("summary50_all", "Быстрая сводка 50 сообщений (в чат)"),
            BotCommand("summary100_all", "Быстрая сводка 100 сообщений (в чат)"),
        ]
        try:
            # Ставим в трёх областях, чтобы Telegram-клиент не дорисовывал «@comicditbot»
            # в панельке команд: дефолтная, приваты, группы.
            await app.set_bot_commands(commands, scope=BotCommandScopeDefault())
            await app.set_bot_commands(commands, scope=BotCommandScopeAllPrivateChats())
            await app.set_bot_commands(commands, scope=BotCommandScopeAllGroupChats())
            log.info("Меню команд бота успешно установлено (3 scope).")
        except Exception as e:
            log.error("Ошибка при установке команд: %s", e)
            
        from pyrogram import idle
        await idle()
        
        await app.stop()

    loop.run_until_complete(main())
