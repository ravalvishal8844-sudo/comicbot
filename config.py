import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Settings:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_bot_token: str
    trigger_keyword: str
    message_count: int
    cooldown_seconds: int
    image_style: str
    # Стили комикса: список {"name": слаг для логов и колбэков, "title": русское название для кнопок,
    # "prompt": описание стиля и персонажей, "reference_image": путь к референсу или None,
    # "reference_group": True, если на референсе целая группа персонажей, а не один}.
    # На каждый комикс случайно выбирается один элемент. Если список пуст — используется image_style.
    comic_prompts: list[dict] = field(default_factory=list)
    # Если True — после безопасного получения картинки (canvas, ~1024px) дополнительно жмём кнопку
    # «Скачать» ради полного размера. В headless Chrome клик иногда роняет браузер; тогда отдаётся
    # уже полученная картинка, а браузер перезапускается на следующем запросе.
    comic_download_fullres: bool = True
    image_max_size: int = 2048
    passive_enabled: bool = True
    passive_min_messages: int = 15
    passive_prompt_suffix: str = ""
    gemini_web_secure_1psid: str = ""
    gemini_web_secure_1psidts: str = ""
    # Прокси для Playwright (Chrome), например "http://127.0.0.1:2080" или "socks5://127.0.0.1:10808"
    gemini_proxy: str = ""
    allowed_chat_ids: list[int] = field(default_factory=list)
    local_llm_url: str = "http://localhost:1234/v1"
    local_llm_model: str = "qwen/qwen3.5-9b"
    local_llm_debug: bool = False
    # Модель эмбеддингов для RAG. e5-base заметно точнее MiniLM на русском.
    # При смене модели векторная база автоматически пересобирается в новую коллекцию.
    embedding_model: str = "intfloat/multilingual-e5-base"
    # Порог косинусной дистанции для векторного поиска (0 = идентично, больше = менее похоже).
    # Сообщения с дистанцией выше порога отбрасываются ещё до LLM-фильтра. Меньше = строже.
    # ВНИМАНИЕ: значение зависит от модели эмбеддингов (у e5 дистанции другие, чем у MiniLM).
    rag_max_distance: float = 0.30
    # Вес свежести при ранжировании кандидатов (0 = только релевантность, 1 = только дата).
    # При прочих равных более свежие сообщения получают приоритет.
    rag_recency_weight: float = 0.25
    # Сколько кандидатов вытягиваем из векторной базы перед LLM-фильтрацией.
    rag_search_limit: int = 400
    # Максимум сообщений, отправляемых в Gemini после строгой LLM-фильтрации (верхняя граница).
    rag_filter_limit: int = 150
    # Размер батча сообщений на один запрос к локальной LLM-фильтру (баланс скорость/контекст).
    rag_filter_batch_size: int = 100
    # До скольки символов обрезается каждое сообщение в промпте фильтра (защита от переполнения).
    rag_msg_char_limit: int = 300
    # Минимальная длина сообщения для индексации (короче — шум: «да», «ок», смайлы).
    rag_min_message_length: int = 8
    # Лимит токенов ответа локальной LLM (фильтр/переписывание запроса — однострочные ответы).
    local_llm_max_tokens: int = 256
    # Если True, для генерации финального RAG-ответа (/recall) используется локальная LLM вместо Gemini
    rag_use_local_llm_for_response: bool = False
    # Сколько последних сообщений берём как контекст для ответа «архитектора».
    architect_context: int = 30
    # Время жизни ветки /recall (сек): в её рамках работают уточняющие вопросы.
    recall_thread_ttl: int = 3600

    def unique_comic_styles(self) -> list[dict]:
        """Стили комикса без дублей по name, в порядке объявления.

        Дубли в config.yaml нужны только для веса при случайном выборе; в меню /comic
        и в списке Web UI каждый стиль показывается один раз.
        """
        seen: set[str] = set()
        result: list[dict] = []
        for style in self.comic_prompts:
            if style["name"] in seen:
                continue
            seen.add(style["name"])
            result.append(style)
        return result


def _parse_comic_prompts(raw, fallback_style: str) -> list[dict]:
    """Разбирает comic_prompts из config.yaml: элементы {name, title?, prompt, reference_image?} или просто строки.

    title — русское название стиля для кнопок в Telegram и Web UI; если не задано, берётся name.
    reference_image — путь к картинке-референсу относительно папки проекта; она прикрепляется
    к запросу комикса. Если файла нет, стиль работает без референса.
    reference_group — на референсе ансамбль персонажей (кто где стоит, описывается в prompt).
    """
    project_dir = Path(__file__).parent
    prompts = []
    for i, item in enumerate(raw or [], start=1):
        reference = None
        if isinstance(item, dict):
            name = str(item.get("name") or f"#{i}")
            title = str(item.get("title") or name).strip() or name
            prompt = str(item.get("prompt") or "")
            reference_group = bool(item.get("reference_group"))
            if item.get("reference_image"):
                ref_path = project_dir / str(item["reference_image"])
                if ref_path.is_file():
                    reference = str(ref_path)
                else:
                    print(f"Предупреждение: референс для стиля '{name}' не найден: {ref_path}")
        else:
            name, prompt = f"#{i}", str(item)
            title = name
            reference_group = False
        if prompt.strip():
            prompts.append({
                "name": name, "title": title, "prompt": prompt.strip(),
                "reference_image": reference, "reference_group": reference_group,
            })

    if not prompts:
        prompts.append({"name": "image_style", "title": "image_style", "prompt": fallback_style, "reference_image": None,
                        "reference_group": False})
    return prompts


def load_settings() -> Settings:
    load_dotenv()

    missing = []
    for var in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_BOT_TOKEN"):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        print(f"Ошибка: не заданы переменные окружения: {', '.join(missing)}")
        print("Заполните значения в файле .env")
        sys.exit(1)

    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print("Ошибка: файл config.yaml не найден.")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not os.getenv("GEMINI_WEB_SECURE_1PSID"):
        print("Ошибка: требуется GEMINI_WEB_SECURE_1PSID в .env")
        sys.exit(1)

    image_style = cfg.get("image_style", "Old cartoon style")

    return Settings(
        telegram_api_id=int(os.environ["TELEGRAM_API_ID"]),
        telegram_api_hash=os.environ["TELEGRAM_API_HASH"],
        telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
        gemini_web_secure_1psid=os.getenv("GEMINI_WEB_SECURE_1PSID", ""),
        gemini_web_secure_1psidts=os.getenv("GEMINI_WEB_SECURE_1PSIDTS", ""),
        gemini_proxy=cfg.get("gemini_proxy", ""),
        trigger_keyword=cfg.get("trigger_keyword", "комикс"),
        message_count=cfg.get("message_count", 30),
        cooldown_seconds=cfg.get("cooldown_seconds", 300),
        image_style=image_style,
        comic_prompts=_parse_comic_prompts(cfg.get("comic_prompts"), image_style),
        comic_download_fullres=bool(cfg.get("comic_download_fullres", True)),
        image_max_size=int(cfg.get("image_max_size", 2048)),
        passive_enabled=cfg.get("passive_enabled", True),
        passive_min_messages=int(cfg.get("passive_min_messages", 12)),
        passive_prompt_suffix=cfg.get("passive_prompt_suffix", ""),
        local_llm_url=cfg.get("local_llm_url", "http://localhost:1234/v1"),
        local_llm_model=cfg.get("local_llm_model", "qwen/qwen3.5-9b"),
        local_llm_debug=bool(cfg.get("local_llm_debug", False)),
        embedding_model=cfg.get("embedding_model", "intfloat/multilingual-e5-base"),
        rag_max_distance=float(cfg.get("rag_max_distance", 0.30)),
        rag_recency_weight=float(cfg.get("rag_recency_weight", 0.25)),
        rag_search_limit=int(cfg.get("rag_search_limit", 400)),
        rag_filter_limit=int(cfg.get("rag_filter_limit", 150)),
        rag_filter_batch_size=int(cfg.get("rag_filter_batch_size", 100)),
        rag_msg_char_limit=int(cfg.get("rag_msg_char_limit", 300)),
        rag_min_message_length=int(cfg.get("rag_min_message_length", 8)),
        local_llm_max_tokens=int(cfg.get("local_llm_max_tokens", 256)),
        rag_use_local_llm_for_response=bool(cfg.get("rag_use_local_llm_for_response", False)),
        architect_context=int(cfg.get("architect_context", 30)),
        recall_thread_ttl=int(cfg.get("recall_thread_ttl", 3600)),
        allowed_chat_ids=[int(x) for x in (cfg.get("allowed_chat_ids", []) or [])],
    )
