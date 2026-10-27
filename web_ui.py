import asyncio
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import yaml
from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

import __main__
app = getattr(__main__, 'app', None)
settings = getattr(__main__, 'settings', None)

import bot_messages
import history
from handlers import _chat_titles, _passive_msg_count, _run_passive, _runtime_msg_count
from gemini_web import _get_page, generate_comic_image_web, _context, SPONTANEOUS_CHARACTERS

WEB_HISTORY_VIEW = 50  # сколько последних записей показываем в дашборде
WEB_BOT_MESSAGES_VIEW = 15  # сколько последних сообщений бота показываем в дашборде
MAX_PHOTO_BYTES = 10 * 1024 * 1024  # лимит Telegram на размер фото
MAX_CAPTION_LEN = 1024  # лимит Telegram на длину подписи

# Заголовок сообщения в формате копипасты из Telegram Desktop:
#   [14.04.2026 20:01] Имя Фамилия: первая строка текста
# Остальные строки (до следующего такого заголовка) принадлежат этому же сообщению.
_TG_HEADER_RE = re.compile(
    r"^\[(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})(?::(\d{2}))?\]\s+([^:]+?):\s?(.*)$"
)


def parse_telegram_paste(raw: str) -> list[dict]:
    """Разобрать копипасту из Telegram в список {ts, name, text}.

    Поддерживается вставка из нескольких сообщений подряд. Многострочный текст
    внутри одного сообщения сохраняется целиком. Если строка не матчит
    заголовок — считается продолжением предыдущего сообщения.
    """
    messages: list[dict] = []
    current: dict | None = None
    for line in raw.splitlines():
        m = _TG_HEADER_RE.match(line)
        if m:
            if current is not None:
                current["text"] = current["text"].rstrip("\n")
                messages.append(current)
            dd, mm, yyyy, hh, mi, ss, name, first = m.groups()
            ts = datetime(
                int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss or 0)
            ).isoformat(timespec="seconds")
            current = {"ts": ts, "name": name.strip(), "text": first}
        elif current is not None:
            current["text"] += "\n" + line
        # строки до первого заголовка игнорируем
    if current is not None:
        current["text"] = current["text"].rstrip("\n")
        messages.append(current)
    return messages


fastapp = FastAPI()


class _ConfigDumper(yaml.SafeDumper):
    """Дампер config.yaml: многострочные строки (промпты комикса) пишет блоком `|`."""


def _represent_str(dumper, data):
    if "\n" in data:
        # Хвостовые пробелы в строках заставляют PyYAML откатиться к строке в кавычках с \n
        data = "\n".join(line.rstrip() for line in data.split("\n"))
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_ConfigDumper.add_representer(str, _represent_str)


def update_config_yaml(new_allowed_chats):
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["allowed_chat_ids"] = [str(x) for x in new_allowed_chats]

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, Dumper=_ConfigDumper, allow_unicode=True, sort_keys=False)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>ComicBot Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2, h3 { margin-top: 0; }
        .chat-block { border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
        .btn { padding: 8px 12px; border: none; border-radius: 4px; cursor: pointer; color: white; margin-right: 5px; margin-bottom: 5px; font-size: 14px; transition: filter 0.15s ease, transform 0.05s ease; }
        .btn:hover { filter: brightness(1.08); }
        .btn:active { transform: translateY(1px); }
        .btn-blue { background: #007bff; }
        .btn-green { background: #28a745; }
        .btn-red { background: #dc3545; }
        .btn-gray { background: #6c757d; }
        .btn-purple { background: linear-gradient(135deg, #667eea, #764ba2); font-weight: 600; box-shadow: 0 2px 6px rgba(118,75,162,0.35); }
        textarea, input[type="text"] { width: 100%; padding: 8px; margin: 5px 0 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        .log-box { background: #333; color: #0f0; padding: 10px; font-family: monospace; height: 150px; overflow-y: scroll; border-radius: 4px; }
        .badge { display: inline-block; padding: 3px 7px; border-radius: 10px; font-size: 12px; color: white; background: #6c757d; }
        .badge.active { background: #28a745; }
        .passive-bar { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
        .char-select { padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; background: white; min-width: 240px; font-size: 14px; cursor: pointer; color: #333; }
        .char-select:hover { border-color: #aaa; }
        .char-select:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102,126,234,0.25); }
        .char-select optgroup { font-weight: 600; color: #666; }
        .bot-msg { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-bottom: 1px solid #eee; font-size: 13px; }
        .bot-msg .meta { color: #888; white-space: nowrap; }
        .bot-msg .preview { flex-grow: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .bot-msg .btn { margin: 0; padding: 3px 8px; font-size: 12px; }
    </style>
</head>
<body>
    <h1>ComicBot Dashboard</h1>
    
    <div class="card">
        <h2>Система</h2>
        <button class="btn btn-blue" onclick="checkHealth()">Проверить соединение с Gemini</button>
        <span id="health-status"></span>
    </div>

    <div class="card">
        <h2>Настройки чатов</h2>
        <div id="chats-config">
            <p>Загрузка...</p>
        </div>
        <div style="margin-top: 10px;">
            <input type="text" id="new-chat-id" placeholder="ID нового чата (напр. -100123456)" style="width: 200px; display: inline-block;">
            <button class="btn btn-green" onclick="addAllowedChat()">Добавить ID</button>
        </div>
    </div>

    <div class="card">
        <h2>Активные чаты (в памяти)</h2>
        <button class="btn btn-gray" onclick="loadChats()">Обновить список</button>
        <div id="chats-container" style="margin-top: 15px;"></div>
    </div>

    <script>
        let CHARACTERS_CACHE = null;
        let STYLES_CACHE = null;

        async function apiCall(method, url, body=null) {
            const opts = { method: method, headers: {} };
            if (body) {
                opts.headers['Content-Type'] = 'application/json';
                opts.body = JSON.stringify(body);
            }
            const res = await fetch(url, opts);
            return await res.json();
        }

        async function loadCharacters() {
            if (CHARACTERS_CACHE) return CHARACTERS_CACHE;
            const res = await apiCall('GET', '/api/characters');
            CHARACTERS_CACHE = res.characters || [];
            return CHARACTERS_CACHE;
        }

        function populateCharSelect(selectEl, characters) {
            if (!selectEl || selectEl.dataset.populated === '1') return;
            const random = characters.find(c => c.id === 'random');
            const rest = characters.filter(c => c.id !== 'random');
            let html = '';
            if (random) {
                html += `<option value="${random.id}" selected>${random.label}</option>`;
            }
            html += '<optgroup label="Конкретный характер">';
            for (const c of rest) {
                html += `<option value="${c.id}">${c.label}</option>`;
            }
            html += '</optgroup>';
            selectEl.innerHTML = html;
            selectEl.dataset.populated = '1';
        }

        async function loadComicStyles() {
            if (STYLES_CACHE) return STYLES_CACHE;
            const res = await apiCall('GET', '/api/comic_styles');
            STYLES_CACHE = res.styles || [];
            return STYLES_CACHE;
        }

        function populateStyleSelect(selectEl, styles) {
            if (!selectEl || selectEl.dataset.populated === '1') return;
            selectEl.innerHTML = styles.map((s, i) =>
                `<option value="${s.id}" ${i === 0 ? 'selected' : ''}>${s.label}</option>`
            ).join('');
            selectEl.dataset.populated = '1';
        }

        async function checkHealth() {
            document.getElementById('health-status').innerText = 'Проверка...';
            const res = await apiCall('GET', '/api/health');
            document.getElementById('health-status').innerText = res.status === 'ok' ? `✅ OK: ${res.title}` : `❌ Ошибка: ${res.error}`;
        }

        async function loadConfig() {
            const res = await apiCall('GET', '/api/config');
            const html = res.allowed_chat_ids.map(id => `
                <div>
                    <span class="badge active">${id}</span>
                    <button class="btn btn-red" style="padding: 2px 5px; font-size: 10px;" onclick="removeAllowedChat(${id})">Удалить</button>
                </div>
            `).join('');
            document.getElementById('chats-config').innerHTML = html || '<p>Список пуст (разрешены все чаты)</p>';
        }

        async function addAllowedChat() {
            const val = document.getElementById('new-chat-id').value;
            if (val) {
                await apiCall('POST', '/api/config/chat', {chat_id: parseInt(val)});
                document.getElementById('new-chat-id').value = '';
                loadConfig();
                loadChats();
            }
        }

        async function removeAllowedChat(id) {
            await apiCall('DELETE', `/api/config/chat/${id}`);
            loadConfig();
            loadChats();
        }

        async function loadChats() {
            const res = await apiCall('GET', '/api/chats');
            const cont = document.getElementById('chats-container');
            if (Object.keys(res).length === 0) {
                cont.innerHTML = '<p>Пока нет активных чатов в памяти.</p>';
                return;
            }
            
            let html = '';
            for (const [id, data] of Object.entries(res)) {
                const isAllowed = data.is_allowed ? '<span class="badge active">Включен</span>' : '<span class="badge">Отключен</span>';
                html += `
                <div class="chat-block">
                    <h3>${data.title} (${id}) ${isAllowed}</h3>
                    <p>Пассив: <b>${data.passive_count}</b> | С запуска: <b>${data.runtime_count}</b> | В истории: <b>${data.history_size}</b></p>

                    <div class="passive-bar">
                        <select id="char-${id}" class="char-select" title="Выберите характер пассивного ответа"></select>
                        <select id="humor-${id}" class="char-select" title="Уровень черного юмора" style="min-width: 120px;">
                            <option value="random">Юмор: Рандом</option>
                            <option value="1">Юмор: 1 (Лайт)</option>
                            <option value="2">Юмор: 2</option>
                            <option value="3">Юмор: 3</option>
                            <option value="4">Юмор: 4</option>
                            <option value="5">Юмор: 5 (Средне)</option>
                            <option value="6">Юмор: 6</option>
                            <option value="7">Юмор: 7</option>
                            <option value="8">Юмор: 8</option>
                            <option value="9">Юмор: 9</option>
                            <option value="10">Юмор: 10 (Жесть)</option>
                        </select>
                        <button class="btn btn-purple" onclick="triggerSpontaneous(${id})">✨ Вызвать пассив</button>
                        <select id="style-${id}" class="char-select" title="Стиль комикса (список из config.yaml)" style="min-width: 180px;"></select>
                        <button class="btn btn-blue" onclick="triggerComic(${id})">🎨 Комикс</button>
                        <button class="btn btn-red" onclick="clearBuffer(${id})">Сбросить счётчики</button>
                    </div>

                    <div style="margin-top: 10px; display: flex; align-items: flex-start; gap: 10px;">
                        <div style="flex-grow: 1;">
                            <textarea id="send-msg-${id}" placeholder="Написать в чат от имени бота (или подпись к картинке)..." rows="3" style="margin: 0; resize: vertical;"></textarea>
                            <input type="file" id="send-photo-${id}" accept="image/*" title="Картинка для отправки в чат от имени бота" style="margin-top: 5px;">
                        </div>
                        <button class="btn btn-green" onclick="sendMessage(${id})" style="margin: 0; white-space: nowrap;">Отправить</button>
                    </div>

                    <div style="margin-top: 10px;">
                        <button class="btn btn-red" onclick="deleteLastBotMessage(${id})" title="Удалить у всех участников последнее сообщение, отправленное ботом в этот чат">🗑 Удалить последнее сообщение бота</button>
                        <button class="btn btn-gray" onclick="loadBotMessages(${id})">Сообщения бота…</button>
                        <button class="btn btn-gray" onclick="scanBotMessages(${id})" title="Поискать сообщения бота напрямую в Telegram — для отправленных до перезапуска, которых нет в журнале. Занимает до пары минут.">🔍 Найти в Telegram</button>
                        <div id="bot-msgs-${id}"></div>
                    </div>

                    <div style="margin-top: 10px; display: flex; align-items: flex-start; gap: 10px;">
                        <textarea id="inject-msg-${id}" placeholder="Вставка из Telegram (формат: [14.04.2026 20:01] Имя: текст — можно несколько сообщений подряд) или строка «Имя: текст»" rows="6" style="flex-grow: 1; margin: 0; resize: vertical; font-family: monospace;"></textarea>
                        <button class="btn btn-gray" onclick="injectBuffer(${id})" style="margin: 0; white-space: nowrap;">В историю</button>
                    </div>

                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; color: #007bff;">Последние 50 записей истории (только просмотр)</summary>
                        <textarea readonly rows="12" style="margin-top: 5px; background:#fafafa;">${data.buffer_text}</textarea>
                    </details>
                </div>
                `;
            }
            cont.innerHTML = html;

            const characters = await loadCharacters();
            const styles = await loadComicStyles();
            for (const id of Object.keys(res)) {
                populateCharSelect(document.getElementById(`char-${id}`), characters);
                populateStyleSelect(document.getElementById(`style-${id}`), styles);
            }
        }

        async function triggerComic(id) {
            const sel = document.getElementById(`style-${id}`);
            const style = sel ? sel.value : 'random';
            const label = sel && sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : 'случайный';
            alert(`Комикс (стиль: ${label}) запущен!`);
            await apiCall('POST', `/api/chats/${id}/comic`, { style });
            loadChats();
        }

        async function triggerSpontaneous(id) {
            const sel = document.getElementById(`char-${id}`);
            const character = sel ? sel.value : 'random';
            const labelEl = sel ? sel.options[sel.selectedIndex] : null;
            const label = labelEl ? labelEl.text : 'случайный';

            const humorSel = document.getElementById(`humor-${id}`);
            const humor_level = humorSel && humorSel.value !== 'random' ? parseInt(humorSel.value) : null;

            alert(`Пассив (${label}, Юмор: ${humor_level || 'Рандом'}) запущен!`);
            await apiCall('POST', `/api/chats/${id}/spontaneous`, { character, humor_level });
            loadChats();
        }

        async function clearBuffer(id) {
            await apiCall('DELETE', `/api/chats/${id}/buffer`);
            loadChats();
        }

        async function sendMessage(id) {
            const input = document.getElementById(`send-msg-${id}`);
            const fileInput = document.getElementById(`send-photo-${id}`);
            const file = fileInput.files[0];
            if (file) {
                // Картинка выбрана — отправляем её, текст из поля уходит подписью
                const fd = new FormData();
                fd.append('photo', file);
                fd.append('caption', input.value);
                const res = await fetch(`/api/chats/${id}/photo`, { method: 'POST', body: fd });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    alert(`Не удалось отправить картинку: ${data.error || 'HTTP ' + res.status}`);
                    return;
                }
                fileInput.value = '';
                input.value = '';
                alert('Картинка отправлена');
            } else if(input.value) {
                await apiCall('POST', `/api/chats/${id}/message`, {text: input.value});
                input.value = '';
                alert('Отправлено');
            }
        }

        function escapeHtml(s) {
            return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        }

        function fmtTs(ts) {
            const d = new Date(ts);
            return isNaN(d) ? ts : d.toLocaleString('ru-RU', {day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'});
        }

        function renderBotMessages(id, messages, emptyText) {
            const box = document.getElementById(`bot-msgs-${id}`);
            if (!messages.length) {
                box.innerHTML = `<p style="color:#888; margin: 5px 0;">${emptyText}</p>`;
                return;
            }
            box.innerHTML = messages.map(m => `
                <div class="bot-msg">
                    <span class="meta">${fmtTs(m.ts)} · ${escapeHtml(m.kind_label)}</span>
                    <span class="preview" title="${escapeHtml(m.preview)}">${escapeHtml(m.preview) || '<i>без текста</i>'}</span>
                    <button class="btn btn-red" onclick="deleteBotMessage(${id}, ${m.id}, this)">Удалить</button>
                </div>`).join('');
        }

        async function loadBotMessages(id) {
            const res = await apiCall('GET', `/api/chats/${id}/bot_messages`);
            renderBotMessages(id, res.messages || [], 'Журнал пуст: бот ничего не отправлял с момента запуска. Для старых сообщений — «Найти в Telegram».');
        }

        async function scanBotMessages(id) {
            const box = document.getElementById(`bot-msgs-${id}`);
            box.innerHTML = '<p style="color:#888; margin: 5px 0;">🔍 Ищу сообщения бота в Telegram, это может занять до пары минут…</p>';
            const res = await fetch(`/api/chats/${id}/bot_messages/scan`, { method: 'POST' });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                box.innerHTML = `<p style="color:#dc3545; margin: 5px 0;">Не удалось: ${escapeHtml(data.error || 'HTTP ' + res.status)}</p>`;
                return;
            }
            renderBotMessages(id, data.messages || [], 'В Telegram сообщений бота в этом чате не найдено.');
        }

        async function deleteBotMessage(id, msgId, btn) {
            if (btn) btn.disabled = true;
            const res = await fetch(`/api/chats/${id}/bot_messages/${msgId}`, { method: 'DELETE' });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                alert(`Не удалось удалить: ${data.error || 'HTTP ' + res.status}`);
                if (btn) btn.disabled = false;
                return false;
            }
            if (!data.deleted) {
                alert('Telegram не нашёл это сообщение — скорее всего, оно уже удалено. Убрал его из журнала.');
            }
            loadBotMessages(id);
            return true;
        }

        async function deleteLastBotMessage(id) {
            const res = await apiCall('GET', `/api/chats/${id}/bot_messages?limit=1`);
            const last = (res.messages || [])[0];
            if (!last) {
                alert('В журнале нет сообщений бота для этого чата. Нажмите «Найти в Telegram», чтобы поискать старые.');
                return;
            }
            const text = last.preview ? `\n\n«${last.preview}»` : '';
            if (!confirm(`Удалить у всех последнее сообщение бота (${fmtTs(last.ts)}, ${last.kind_label})?${text}`)) return;
            if (await deleteBotMessage(id, last.id, null)) alert('Удалено');
        }

        async function injectBuffer(id) {
            const input = document.getElementById(`inject-msg-${id}`);
            if(input.value) {
                const res = await apiCall('POST', `/api/chats/${id}/buffer`, {text: input.value});
                input.value = '';
                if (res && res.added) {
                    alert(`Добавлено сообщений: ${res.added}`);
                }
                loadChats();
            }
        }

        window.onload = () => {
            loadConfig();
            loadChats();
        };
    </script>
</body>
</html>
"""

@fastapp.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE

@fastapp.get("/api/health")
async def health_check():
    try:
        page = await _get_page(settings)
        title = await page.title()
        
        # Check if chat interface is present
        editor = page.locator(".ql-editor").first
        is_ready = await editor.is_visible(timeout=5000)
        
        if is_ready:
            return {"status": "ok", "title": title}
        else:
            return {"status": "error", "error": "Interface not ready (need login?)"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@fastapp.get("/api/config")
async def get_config():
    return {"allowed_chat_ids": settings.allowed_chat_ids}

@fastapp.post("/api/config/chat")
async def add_allowed_chat(req: Request):
    data = await req.json()
    cid = data.get("chat_id")
    if cid and cid not in settings.allowed_chat_ids:
        settings.allowed_chat_ids.append(cid)
        update_config_yaml(settings.allowed_chat_ids)
    return {"status": "ok"}

@fastapp.delete("/api/config/chat/{chat_id}")
async def remove_allowed_chat(chat_id: int):
    if chat_id in settings.allowed_chat_ids:
        settings.allowed_chat_ids.remove(chat_id)
        update_config_yaml(settings.allowed_chat_ids)
    return {"status": "ok"}

@fastapp.get("/api/chats")
async def get_chats():
    res = {}
    # Список чатов: те что засвечены в рантайме + все, у кого есть файл истории
    chat_ids: set[int] = set(_chat_titles.keys()) | set(history.known_chats())
    for cid in chat_ids:
        title = _chat_titles.get(cid, str(cid))
        lines = history.get_lines(cid, WEB_HISTORY_VIEW)
        res[cid] = {
            "title": title,
            "passive_count": _passive_msg_count.get(cid, 0),
            "runtime_count": _runtime_msg_count.get(cid, 0),
            "history_size": history.file_size(cid),
            "buffer_size": len(lines),
            "buffer_text": "\n".join(lines),
            "is_allowed": (not settings.allowed_chat_ids) or (cid in settings.allowed_chat_ids),
        }
    return res

@fastapp.post("/api/chats/{chat_id}/message")
async def send_bot_message(chat_id: int, req: Request):
    data = await req.json()
    text = data.get("text", "")
    if text:
        await app.send_message(chat_id, text)
    return {"status": "ok"}

@fastapp.post("/api/chats/{chat_id}/photo")
async def send_bot_photo(chat_id: int, photo: UploadFile = File(...), caption: str = Form("")):
    """Отправить картинку в чат от имени бота (multipart: photo + необязательная подпись caption)."""
    if not (photo.content_type or "").startswith("image/"):
        return JSONResponse({"status": "error", "error": "Файл не является картинкой"}, status_code=400)
    if len(caption) > MAX_CAPTION_LEN:
        return JSONResponse(
            {"status": "error", "error": f"Подпись длиннее {MAX_CAPTION_LEN} символов ({len(caption)})"},
            status_code=400,
        )

    data = await photo.read()
    if not data:
        return JSONResponse({"status": "error", "error": "Пустой файл"}, status_code=400)
    if len(data) > MAX_PHOTO_BYTES:
        return JSONResponse(
            {"status": "error", "error": f"Файл больше {MAX_PHOTO_BYTES // (1024 * 1024)} МБ"},
            status_code=400,
        )

    image = BytesIO(data)
    image.name = photo.filename or "image.png"
    try:
        await app.send_photo(chat_id, photo=image, caption=caption)
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
    return {"status": "ok"}

@fastapp.get("/api/chats/{chat_id}/bot_messages")
async def list_bot_messages(chat_id: int, limit: int = WEB_BOT_MESSAGES_VIEW):
    """Последние сообщения бота в чате из журнала (новые первыми)."""
    return {"messages": bot_messages.recent(chat_id, max(1, min(limit, bot_messages.MAX_PER_CHAT)))}

@fastapp.post("/api/chats/{chat_id}/bot_messages/scan")
async def scan_bot_messages(chat_id: int):
    """Найти сообщения бота напрямую в Telegram (для отправленных до перезапуска) и дополнить журнал."""
    try:
        found = await bot_messages.scan(app, chat_id, want=WEB_BOT_MESSAGES_VIEW)
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
    return {"status": "ok", "messages": found}

@fastapp.delete("/api/chats/{chat_id}/bot_messages/{msg_id}")
async def delete_bot_message(chat_id: int, msg_id: int):
    """Удалить сообщение бота у всех участников. deleted=0 — Telegram его уже не нашёл."""
    try:
        deleted = await app.delete_messages(chat_id, msg_id, revoke=True)
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
    bot_messages.forget(chat_id, msg_id)  # на случай, если клиент без обёртки install()
    return {"status": "ok", "deleted": int(deleted or 0)}

@fastapp.post("/api/chats/{chat_id}/buffer")
async def inject_buffer(chat_id: int, req: Request):
    """Добавить сообщения в историю чата.

    Поддерживает два формата:
    - Копипаста из Telegram (одно или несколько сообщений подряд, формат
      «[DD.MM.YYYY HH:MM] Имя: текст» с многострочным телом). Временные метки
      из вставки сохраняются.
    - Одна строка «Имя: текст» (упрощённый ручной ввод) — текущее время.
    """
    data = await req.json()
    text = data.get("text", "") or ""
    if not text.strip():
        return {"status": "ok", "added": 0}

    parsed = parse_telegram_paste(text)
    
    import rag
    
    if parsed:
        for msg in parsed:
            history.append(chat_id, msg["name"] or "Web", msg["text"], ts=msg["ts"])
            try:
                rag.add_message(chat_id, msg["name"] or "Web", msg["text"], msg["ts"])
            except Exception:
                pass
        added = len(parsed)
    else:
        name, _, body = text.partition(":")
        if body:
            history.append(chat_id, name.strip() or "Web", body.strip())
            try:
                rag.add_message(chat_id, name.strip() or "Web", body.strip())
            except Exception:
                pass
        else:
            history.append(chat_id, "Web", text.strip())
            try:
                rag.add_message(chat_id, "Web", text.strip())
            except Exception:
                pass
        added = 1

    _passive_msg_count[chat_id] = _passive_msg_count.get(chat_id, 0) + added
    _runtime_msg_count[chat_id] = _runtime_msg_count.get(chat_id, 0) + added
    return {"status": "ok", "added": added}

@fastapp.put("/api/chats/{chat_id}/buffer")
async def update_buffer(chat_id: int, req: Request):
    """
    Редактирование буфера в дашборде больше не поддерживается:
    история хранится в append-only JSONL. Возвращаем 405.
    """
    return JSONResponse(
        {"status": "error", "error": "History is append-only. Use POST to append a line."},
        status_code=405,
    )

@fastapp.delete("/api/chats/{chat_id}/buffer")
async def clear_chat_buffer(chat_id: int):
    """Сбрасываем только runtime-счётчики. История на диске не трогается."""
    _passive_msg_count[chat_id] = 0
    _runtime_msg_count[chat_id] = 0
    return {"status": "ok"}

async def _manual_comic(chat_id: int, style: str | None = None):
    recent = history.get_lines(chat_id, settings.message_count)
    conversation = "\n".join(recent)
    try:
        image_bytes = await generate_comic_image_web(conversation, settings, force_style=style)
        photo = BytesIO(image_bytes)
        photo.name = "comic.png"
        await app.send_photo(chat_id, photo=photo)
        _passive_msg_count[chat_id] = 0
        _runtime_msg_count[chat_id] = 0
    except Exception as e:
        await app.send_message(chat_id, f"Ошибка генерации с веб-морды: {e}")

@fastapp.post("/api/chats/{chat_id}/comic")
async def trigger_comic_route(chat_id: int, req: Request, bg_tasks: BackgroundTasks):
    """Запустить комикс. Body: {"style": "<name>" | "random" | null} — стиль из comic_prompts."""
    style: str | None = None
    try:
        data = await req.json()
        style = data.get("style")
    except Exception:
        # тело пустое или невалидный JSON — рандом
        pass

    force = style if style and style != "random" else None
    bg_tasks.add_task(_manual_comic, chat_id, force)
    return {"status": "started", "style": force or "random"}

@fastapp.get("/api/comic_styles")
async def list_comic_styles():
    """Список стилей комикса из config.yaml (comic_prompts). Первая опция — 'random'.

    id — name стиля (слаг, его ждёт /api/chats/{id}/comic), label — русское title
    (+ 🖼, если у стиля есть картинка-референс).
    """
    items = [{"id": "random", "label": "🎲 Случайный стиль"}]
    for style in settings.unique_comic_styles():
        label = style["title"] + (" 🖼" if style.get("reference_image") else "")
        items.append({"id": style["name"], "label": label})
    return {"styles": items}

@fastapp.get("/api/characters")
async def list_characters():
    """Список доступных характеров для пассивного ответа.

    Первая опция — синтетическая 'random' для случайного выбора. Остальные —
    из SPONTANEOUS_CHARACTERS в порядке объявления.
    """
    items = [{"id": "random", "label": "🎲 Случайный характер"}]
    for slug, display_name, _desc in SPONTANEOUS_CHARACTERS:
        items.append({"id": slug, "label": display_name})
    return {"characters": items}

@fastapp.post("/api/chats/{chat_id}/spontaneous")
async def trigger_spontaneous_route(chat_id: int, req: Request, bg_tasks: BackgroundTasks):
    """Запустить пассив. Body: {"character": "<id>" | "random" | null, "humor_level": int | null}.

    Если character не задан или равен "random" — выбирается случайный характер
    из SPONTANEOUS_CHARACTERS. Иначе — принудительно используется указанный id.
    """
    character: str | None = None
    humor_level: int | None = None
    try:
        data = await req.json()
        character = data.get("character")
        humor_level = data.get("humor_level")
        if humor_level is not None:
            humor_level = int(humor_level)
    except Exception:
        # тело пустое или невалидный JSON — рандом
        pass

    force = character if character and character != "random" else None
    bg_tasks.add_task(_run_passive, app, chat_id, settings, force, humor_level)
    return {"status": "started", "character": force or "random", "humor_level": humor_level or "random"}
