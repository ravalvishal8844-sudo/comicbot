import json
from pathlib import Path
import logging
from datetime import datetime
import threading
from queue import Queue

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger("rag")

DB_PATH = Path(__file__).parent / "data" / "chroma_db"

# Глобальные переменные для базы и модели, чтобы не грузить их каждый раз
_chroma_client = None
_collection = None
_embed_model = None
_embed_model_name = "intfloat/multilingual-e5-base"
_embed_is_e5 = True  # e5-модели требуют префиксы query:/passage:
_min_message_length = 8  # минимальная длина сообщения для индексации (настраивается через init_db)

# Очередь для фонового добавления сообщений в векторную базу
_add_queue = Queue()
_worker_thread = None

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def _prefix_text(text: str, kind: str) -> str:
    """
    Добавляет префикс, требуемый моделями семейства e5 ('query: ' / 'passage: ').
    Для остальных моделей возвращает текст без изменений.
    """
    if not _embed_is_e5:
        return text
    prefix = "query: " if kind == "query" else "passage: "
    return f"{prefix}{text}"


def _collection_name_for(model_name: str) -> str:
    """
    Имя коллекции, привязанное к модели эмбеддингов. При смене модели используется
    новая коллекция (с другой размерностью векторов), что исключает падения из-за
    несовместимости и обеспечивает автоматическую пересборку базы.
    """
    import re as _re_local
    safe = _re_local.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"chat_messages__{safe}"


def _is_indexable(text: str) -> bool:
    """
    Отсеивает шумовые короткие сообщения ('да', 'ок', '+1', смайлы), которые
    лишь засоряют векторный поиск и контекст. Возвращает False для таких сообщений.
    """
    if not text:
        return False
    flat = " ".join(text.split())
    if len(flat) < _min_message_length:
        return False
    if not any(c.isalnum() for c in flat):
        return False
    return True

def _process_queue():
    """Фоновый воркер, который кодирует тексты и пишет в ChromaDB, не блокируя основной event loop."""
    log.info("Воркер RAG запущен.")
    while True:
        item = _add_queue.get()
        if item is None:
            break
        
        chat_id, name, text, ts = item
        try:
            if not _collection or not _embed_model:
                continue

            if not _is_indexable(text):
                continue

            import hashlib
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
            msg_id = f"{chat_id}_{ts}_{text_hash}"
            
            try:
                embedding = _embed_model.encode(_prefix_text(text, "passage"), show_progress_bar=False).tolist()
            except Exception as e:
                log.error("Ошибка при кодировании текста: %s", e)
                continue
            
            try:
                _collection.add(
                    ids=[msg_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[{"chat_id": chat_id, "name": name, "ts": ts}]
                )
            except Exception as e:
                log.error("Ошибка при добавлении в коллекцию: %s", e)
        except Exception as e:
            log.error("Ошибка фонового добавления в RAG: %s", e)
        finally:
            _add_queue.task_done()

def init_db(embedding_model: str = DEFAULT_EMBEDDING_MODEL, min_message_length: int = 8):
    global _chroma_client, _collection, _embed_model, _worker_thread
    global _embed_model_name, _embed_is_e5, _min_message_length

    _embed_model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
    _embed_is_e5 = "e5" in _embed_model_name.lower()
    _min_message_length = max(0, int(min_message_length))

    log.info("Инициализация ChromaDB и загрузка модели эмбеддингов '%s' (это может занять пару минут при первом запуске)...", _embed_model_name)
    
    # Отключаем логи и прогресс-бары huggingface
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    import transformers
    transformers.logging.set_verbosity_error()
    
    import logging as py_logging
    py_logging.getLogger("sentence_transformers").setLevel(py_logging.WARNING)
    
    # Жестко отключаем прогресс-бары внутри sentence_transformers
    from sentence_transformers import SentenceTransformer
    class QuietSentenceTransformer(SentenceTransformer):
        def encode(self, *args, **kwargs):
            kwargs['show_progress_bar'] = False
            return super().encode(*args, **kwargs)
            
    _embed_model = QuietSentenceTransformer(_embed_model_name)
    
    # Инициализируем ChromaDB
    DB_PATH.mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
    
    # Коллекция привязана к модели: смена модели => новая коллекция и авто-пересборка,
    # без конфликта размерностей векторов со старыми данными.
    collection_name = _collection_name_for(_embed_model_name)
    _collection = _chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Используем косинусное расстояние для поиска
    )
    
    log.info("Векторная база RAG инициализирована (коллекция '%s'). Записей в базе: %d", collection_name, _collection.count())
    
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_process_queue, daemon=True)
        _worker_thread.start()

def add_message(chat_id: int, name: str, text: str, ts: str = None):
    """Добавляет сообщение в очередь на индексацию (не блокирует поток)."""
    if not ts:
        ts = datetime.now().isoformat(timespec="seconds")
    
    _add_queue.put((chat_id, name, text, ts))

def search_context(chat_id: int, query: str, limit: int = 50, max_distance: float | None = None,
                   recency_weight: float = 0.0) -> list[str]:
    """
    Ищет сообщения по смыслу (семантический векторный поиск) с учётом свежести.
    Возвращает список строк '[дата] Имя: текст', отсортированных по убыванию
    комбинированного приоритета (релевантность + свежесть).

    max_distance: порог косинусной дистанции (0 = идентично). Результаты с дистанцией
    выше порога отбрасываются ещё до LLM-фильтра. None — порог не применяется.
    recency_weight: вес свежести при ранжировании (0 = только релевантность, 1 = только дата).
    """
    if not _collection or not _embed_model:
        log.error("База данных или модель не инициализированы!")
        return []

    try:
        # Превращаем запрос пользователя в вектор (с префиксом query: для e5-моделей)
        try:
            query_embedding = _embed_model.encode(_prefix_text(query, "query"), show_progress_bar=False).tolist()
        except Exception as e:
            log.error("Ошибка при кодировании поискового запроса: %s", e)
            return []
        
        # Над-выборка: берём кандидатов больше, чем нужно, чтобы свежесть могла реально
        # влиять на ИТОГОВЫЙ отбор (а не только переупорядочивать фиксированный набор).
        chat_count = len(_collection.get(where={"chat_id": chat_id})['ids'])
        over_fetch = min(limit * 2, chat_count) if recency_weight > 0 else min(limit, chat_count)

        if over_fetch == 0:
            return []
            
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=over_fetch,
            where={"chat_id": chat_id}, # Фильтруем только по текущему чату
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        distances = results.get('distances') or [[]]
        distances = distances[0] if distances else []

        candidates = []
        dropped = 0
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = distances[i] if i < len(distances) else None

            # Мягкий порог по косинусной дистанции: отсекаем явно непохожие сообщения
            # ещё до LLM-фильтра. Чем меньше max_distance, тем строже отбор.
            if max_distance is not None and dist is not None and dist > max_distance:
                dropped += 1
                continue

            candidates.append({
                "ts": meta["ts"],
                "name": meta["name"],
                "text": doc,
                "distance": dist if dist is not None else 1.0,
            })

        if max_distance is not None:
            log.info(
                "Векторный поиск: оставлено %d, отброшено %d по порогу дистанции %.2f",
                len(candidates), dropped, max_distance,
            )

        # Ранжируем по комбинированному скору (релевантность + свежесть) и берём top-limit.
        ranked = _rank_candidates(candidates, recency_weight)[:limit]

        # Добавляем дату к каждой строке. Это нужно, чтобы дальше по пайплайну (и в Gemini)
        # можно было восстановить хронологию: если команда «переобулась», то более позднее
        # сообщение отражает финальное/актуальное решение.
        return [f"[{_format_ts(msg['ts'])}] {msg['name']}: {msg['text']}" for msg in ranked]
        
    except Exception as e:
        log.error(f"Векторный поиск упал с ошибкой: {e}")
        return []

import re as _re

def _strip_thinking(text: str) -> str:
    """Удаляет блоки рассуждений reasoning-моделей (<think>...</think> и т.п.)."""
    if not text:
        return ""
    # Удаляем закрытые блоки <think>...</think>
    cleaned = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    # Удаляем незакрытый блок (если генерация оборвалась внутри размышления)
    cleaned = _re.sub(r"<think>.*$", "", cleaned, flags=_re.DOTALL | _re.IGNORECASE)
    # Иногда модель оставляет одиночный закрывающий тег
    cleaned = cleaned.replace("</think>", "")
    return cleaned.strip()


def _format_ts(ts: str) -> str:
    """Превращает ISO-таймстамп в читаемый вид 'YYYY-MM-DD HH:MM'. При ошибке возвращает как есть."""
    if not ts:
        return "??"
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _ts_to_epoch(ts: str) -> float:
    """Парсит ISO-таймстамп в секунды эпохи. При ошибке возвращает 0.0."""
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def _rank_candidates(candidates: list[dict], recency_weight: float) -> list[dict]:
    """
    Ранжирует кандидатов по комбинированному скору: релевантность (1 - дистанция) и свежесть.
    Обе компоненты нормируются в [0, 1] внутри набора, затем смешиваются с весом recency_weight.
    Возвращает список, отсортированный по убыванию итогового приоритета.
    """
    if not candidates:
        return []

    w = max(0.0, min(1.0, recency_weight))
    if w == 0.0:
        # Только релевантность: сохраняем порядок ChromaDB (по возрастанию дистанции).
        return sorted(candidates, key=lambda c: c.get("distance", 1.0))

    sims = [1.0 - c.get("distance", 1.0) for c in candidates]
    epochs = [_ts_to_epoch(c.get("ts", "")) for c in candidates]

    def _norm(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return [0.5 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    sim_n = _norm(sims)
    rec_n = _norm(epochs)

    scored = []
    for c, s, r in zip(candidates, sim_n, rec_n):
        c = dict(c)
        c["_score"] = (1.0 - w) * s + w * r
        scored.append(c)

    scored.sort(key=lambda c: c["_score"], reverse=True)
    return scored


def _line_recency_key(line: str) -> float:
    """Возвращает epoch из префикса '[YYYY-MM-DD HH:MM] ...'. Без даты -> 0.0 (самое старое)."""
    m = _re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]", line or "")
    if not m:
        return 0.0
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").timestamp()
    except Exception:
        return 0.0


def sort_messages_chronologically(messages: list[str]) -> list[str]:
    """
    Сортирует строки контекста по дате в префиксе '[YYYY-MM-DD HH:MM] ...' (по возрастанию).
    Строки без распознаваемой даты уходят в конец, сохраняя исходный относительный порядок.
    """
    def key(item):
        idx, line = item
        m = _re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]", line)
        if m:
            return (0, m.group(1), idx)
        return (1, "", idx)

    return [line for _, line in sorted(enumerate(messages), key=key)]


def _truncate_for_prompt(text: str, limit: int) -> str:
    """Обрезает текст до limit символов (для промпта фильтрации), схлопывая переносы строк."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def _parse_filter_output(raw: str) -> list[int]:
    """
    Извлекает номера сообщений из типизированного ответа `List: 12,13,21`.
    Парсим ТОЛЬКО подстроку после префикса `List:`, чтобы не выдергивать
    случайные числа из возможного текста рассуждений. Возвращает [] при отсутствии
    префикса или при `List: EMPTY`.
    """
    if not raw:
        return []

    m = _re.search(r"List:\s*(.*)", raw, flags=_re.IGNORECASE)
    if not m:
        return []

    payload = m.group(1).splitlines()[0].strip()
    if not payload or payload.upper().startswith("EMPTY"):
        return []

    return [int(x) for x in _re.findall(r"\d+", payload)]


async def _call_local_llm(prompt: str, settings, max_tokens: int = 512, task: str = "llm") -> str:
    """
    Единая точка обращения к локальной LLM (OpenAI-совместимый эндпоинт).
    Логирует запрос/ответ при включенном settings.local_llm_debug,
    вырезает блоки рассуждений (<think>) и учитывает reasoning_content.
    Возвращает очищенный текст ответа (или пустую строку при ошибке).
    """
    import httpx

    debug = getattr(settings, "local_llm_debug", False)
    # Режим раздумий Qwen3.5 отключается в самой LM Studio (мягкий /no_think он игнорирует).
    # Поэтому работаем только с полем content и НИКОГДА не используем reasoning_content как ответ.

    if debug:
        log.info("=== LOCAL LLM [%s] REQUEST (max_tokens=%d) ===\n%s\n=== END REQUEST ===", task, max_tokens, prompt)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.local_llm_url.rstrip('/')}/chat/completions",
                json={
                    "model": settings.local_llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            if debug:
                log.info(
                    "=== LOCAL LLM [%s] RESPONSE ===\nreasoning_content: %s\ncontent: %s\n=== END RESPONSE ===",
                    task, reasoning.strip(), content.strip(),
                )

            # Работаем строго с content. Если он пустой (модель ушла в раздумья) —
            # это считается провалом вызова, чтобы НЕ тригериться на текст рассуждений.
            cleaned = _strip_thinking(content)
            if not cleaned:
                log.warning("Локальная LLM [%s] вернула пустой content (вероятно, не отключен reasoning).", task)
                return ""
            return cleaned
    except Exception as e:
        log.error("Ошибка при обращении к локальной LLM [%s]: %s", task, e)
        return ""


async def rewrite_query_with_local_llm(query: str, thread_context: list[str], settings) -> str:
    """
    Переписывает уточняющий запрос в полноценный поисковый запрос,
    используя контекст предыдущих сообщений в ветке.
    """
    if not thread_context:
        return query
    
    prompt = f"""Ты — интеллектуальный маршрутизатор векторного поиска.
Пользователь задает уточняющий вопрос в рамках текущей беседы.
Сформируй ОДНУ строку поискового запроса для векторной базы (RAG) на русском языке.

ПРАВИЛА:
1. Разреши все местоимения (он, она, это, там) в конкретные сущности из истории диалога.
2. Добавь ключевые слова, названия и термины, которые обсуждались в ветке.
3. Это должна быть одна короткая строка (до 200 символов), без переносов строк и рассуждений.
4. Отвечай СТРОГО в формате одной строки: Query: <текст запроса>
   Пример: Query: алгоритм подписи ГОСТ Р 34.11-2012, метка времени, createDetachedSignature

История диалога:
---
{chr(10).join(thread_context)}
---

Последний (уточняющий) вопрос: "{query}"

Ответ (одна строка):"""

    raw = await _call_local_llm(prompt, settings, max_tokens=getattr(settings, "local_llm_max_tokens", 256), task="rewrite")

    rewritten = _parse_rewrite_output(raw)

    # Если LLM вернула пустую строку, мусор или сломалась — откатываемся на исходный вопрос.
    if not rewritten:
        log.info("Rewrite дал некорректный результат, используем исходный запрос: '%s'", query)
        return query

    log.info("Запрос переписан локальной LLM: '%s' -> '%s'", query, rewritten)
    return rewritten


def _parse_rewrite_output(raw: str) -> str:
    """
    Извлекает строку запроса из типизированного ответа `Query: ...`.
    Возвращает "" если ответ отсутствует, похож на дамп рассуждений,
    слишком длинный, многострочный или преимущественно латиница/английский.
    """
    if not raw:
        return ""

    text = raw.strip()

    # Берем только то, что после префикса Query:
    m = _re.search(r"Query:\s*(.+)", text, flags=_re.IGNORECASE)
    if not m:
        return ""
    candidate = m.group(1).strip().strip('"\'').strip()

    # Только первая строка (на случай если модель добавила что-то ниже)
    candidate = candidate.splitlines()[0].strip() if candidate else ""

    if len(candidate) < 3 or len(candidate) > 300:
        return ""

    # Отсекаем явные признаки утечки рассуждений
    lowered = candidate.lower()
    junk_markers = ("thinking process", "analyze the", "*goal:", "expanded query", "dialogue history")
    if any(marker in lowered for marker in junk_markers):
        return ""

    # Если строка почти целиком на латинице — это, скорее всего, англоязычное рассуждение,
    # а не русскоязычный поисковый запрос.
    letters = [c for c in candidate if c.isalpha()]
    if letters:
        latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
        if latin / len(letters) > 0.6:
            return ""

    return candidate

async def filter_context_with_local_llm(query: str, messages: list[str], settings, max_results: int = 50) -> list[str]:
    """
    Фильтрует набор сообщений, оставляя только наиболее релевантные запросу,
    используя локальную LLM. Обрабатывает сообщения батчами (по 100 шт.),
    чтобы не превысить лимит токенов (HTTP 400).

    Сообщения предполагаются отсортированными по релевантности (сначала самые близкие
    по вектору) — это важно для корректного fallback.
    """
    if not messages or len(messages) <= max_results:
        return messages

    batch_size = getattr(settings, "rag_filter_batch_size", 100)
    # Длинные сообщения (огромные JWT-токены, curl-дампы, JSON) обрезаем для промпта:
    # для оценки релевантности достаточно начала, а полный текст переполняет контекст (HTTP 400).
    msg_char_limit = getattr(settings, "rag_msg_char_limit", 300)
    max_tokens = getattr(settings, "local_llm_max_tokens", 256)
    valid_indices = set()

    for i in range(0, len(messages), batch_size):
        batch_msgs = messages[i:i+batch_size]
        numbered_msgs = [
            f"[{j+i}] {_truncate_for_prompt(msg, msg_char_limit)}"
            for j, msg in enumerate(batch_msgs)
        ]
        messages_text = "\n".join(numbered_msgs)

        prompt = f"""Тебе дан список из {len(batch_msgs)} сообщений из чата и поисковый запрос.
Каждое сообщение начинается с даты в формате [ГГГГ-ММ-ДД ЧЧ:ММ].
Отбери ТОЛЬКО те сообщения, которые НАПРЯМУЮ относятся к запросу и реально помогают на него ответить.

СТРОГИЕ ПРАВИЛА ОТБОРА:
- Бери только прямые попадания по смыслу. Если сомневаешься — НЕ включай.
- Игнорируй сообщения, связанные с темой лишь косвенно или по общим словам.
- При прочих равных ПРЕДПОЧИТАЙ более свежие сообщения (с более поздней датой) — они актуальнее.
- Лучше выбрать меньше, но точнее. Максимум {max_results} сообщений.

ФОРМАТ ОТВЕТА — строго ОДНА строка:
List: 12,13,21
Если ничего не подходит, ответь ровно: List: EMPTY
Не пиши ничего, кроме этой строки.

Поисковый запрос: "{query}"

Сообщения:
{messages_text}

Ответ:"""

        content = await _call_local_llm(prompt, settings, max_tokens=max_tokens, task="filter")

        for idx in _parse_filter_output(content):
            if i <= idx < i + len(batch_msgs):
                valid_indices.add(idx)

    if not valid_indices:
        # Fallback: отдаем самые близкие по вектору сообщения (первые max_results),
        # а не хронологический "хвост". messages уже отранжированы (релевантность + свежесть).
        log.warning("Локальная LLM не выбрала ни одного сообщения для RAG, берем top-%d по приоритету", max_results)
        return messages[:max_results]

    selected = [messages[idx] for idx in sorted(valid_indices)]

    # Если выбрано больше лимита — отбрасываем самые старые (приоритет свежим),
    # при равенстве дат сохраняем исходный порядок (релевантность).
    if len(selected) > max_results:
        ordered = sorted(
            enumerate(selected),
            key=lambda item: (_line_recency_key(item[1]), -item[0]),
            reverse=True,
        )
        selected = [line for _, line in ordered[:max_results]]

    log.info("Локальная LLM отфильтровала %d -> %d сообщений", len(messages), len(selected))
    return selected


async def generate_rag_response_with_local_llm(context_messages: list[str], question: str, thread_context: list[str], settings) -> str:
    """
    Генерирует финальный ответ на основе найденной истории, используя локальную LLM.
    Альтернатива Gemini, работающая без прокси и браузера.
    """
    if not context_messages and not thread_context:
        return "Я не нашел в истории чата ничего похожего на ваш запрос."

    context = "\n".join(context_messages)
    thread_str = ""
    if thread_context:
        thread_str = "\nИстория текущего диалога (ветки):\n---\n" + "\n".join(thread_context) + "\n---\n"

    prompt = f"""Ты — умный и внимательный ИИ-ассистент в рабочем чате.
Твоя задача — дать КОНКРЕТНЫЙ ответ на конкретный вопрос пользователя, опираясь на предоставленные фрагменты истории чата.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Отвечай СТРОГО на заданный вопрос. Не уходи в сторону и не поднимай смежные темы.
2. Изучи историю чата. Если точного ответа нет, честно скажи об этом. Не выдумывай факты.
3. Сообщения упорядочены ПО ВРЕМЕНИ (дата указана в начале каждой строки).
4. ВАЖНО ПРО ИЗМЕНЕНИЕ РЕШЕНИЙ: команда могла сначала договориться об одном, а позже передумать. Если по теме есть противоречащие сообщения в разные даты — АКТУАЛЬНЫМ считается БОЛЕЕ ПОЗДНЕЕ решение. Если подход менялся, прямо скажи, что было раньше и что стало в итоге.
5. Цитируй или ссылайся на конкретных людей и даты, если это уместно.
6. Отвечай ИНФОРМАТИВНО, но КРАТКО И ПО ДЕЛУ. Уложись в 1-3 небольших абзаца.

Найденные фрагменты истории (по возрастанию времени, могут быть с пропусками):
---
{context}
---
{thread_str}
Вопрос пользователя: "{question}"

Ответ:"""

    # Для финального ответа даем побольше токенов, чем для служебных команд
    result = await _call_local_llm(prompt, settings, max_tokens=1024, task="response")
    if not result:
        return "Не удалось сгенерировать ответ (локальная LLM вернула пустой результат)."
    return result


def backfill_from_jsonl():
    """Загружает существующую историю из jsonl файлов в ChromaDB, если БД пуста."""
    if not _collection or not _embed_model:
        return
        
    if _collection.count() > 0:
        log.info("База уже содержит %d записей. Если хотите переиндексировать, удалите папку data/chroma_db", _collection.count())
        # Временно убираем return, чтобы догрузить недостающие сообщения
        # return # Уже заполнено
        
    history_dir = Path(__file__).parent / "data" / "history"
    if not history_dir.exists():
        return
        
    log.info("Перенос истории в векторную базу (Backfill)... Это займет время, так как тексты кодируются нейросетью.")
    
    batch_size = 100
    
    for file_path in history_dir.glob("*.jsonl"):
        try:
            chat_id = int(file_path.stem)
            
            ids = []
            documents = []
            metadatas = []
            seen_ids = set()
            indexed = 0

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        ts = data.get("ts", "")
                        name = data.get("name", "")
                        text = data.get("text", "")
                        
                        if not text: continue
                        if not _is_indexable(text): continue
                        
                        import hashlib
                        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
                        msg_id = f"{chat_id}_{ts}_{text_hash}"
                        
                        if msg_id in seen_ids:
                            continue
                        seen_ids.add(msg_id)
                        
                        # Проверяем, есть ли уже такой ID в базе (чтобы не дублировать при догрузке)
                        try:
                            existing = _collection.get(ids=[msg_id])
                            if existing and existing['ids']:
                                continue
                        except Exception:
                            pass
                            
                        ids.append(msg_id)
                        documents.append(text)
                        metadatas.append({"chat_id": chat_id, "name": name, "ts": ts})
                        
                        # Отправляем пачками по 100 сообщений, чтобы не перегружать память
                        if len(documents) >= batch_size:
                            try:
                                to_encode = [_prefix_text(d, "passage") for d in documents]
                                embeddings = _embed_model.encode(to_encode, show_progress_bar=False).tolist()
                                _collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                                indexed += len(documents)
                                log.info("  чат %d: проиндексировано %d сообщений...", chat_id, indexed)
                            except Exception as e:
                                log.error("Ошибка при пакетном добавлении в ChromaDB: %s", e)
                            ids, documents, metadatas = [], [], []
                            
                    except json.JSONDecodeError:
                        pass
                        
            # Добиваем остатки
            if documents:
                try:
                    to_encode = [_prefix_text(d, "passage") for d in documents]
                    embeddings = _embed_model.encode(to_encode, show_progress_bar=False).tolist()
                    _collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                    indexed += len(documents)
                except Exception as e:
                    log.error("Ошибка при добавлении остатков в ChromaDB: %s", e)
                
            log.info("Загружен чат %d (проиндексировано %d сообщений)", chat_id, indexed)
            
        except ValueError:
            pass
            
    log.info("Векторный Backfill завершен. Всего записей: %d", _collection.count())

if __name__ == "__main__":
    init_db()
    backfill_from_jsonl()
