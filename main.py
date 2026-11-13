import os
import json
import time
import asyncio
import uuid
import logging
import threading
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_alisa")

API_URL = "https://routerapi.ru/api/v1/chat/completions"
API_KEY = os.getenv("ROUTER_API_KEY")

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

executor = ThreadPoolExecutor(max_workers=2)
search_jobs = {}
# Последний прерванный многочастный ответ сохраняется отдельно,
# чтобы короткое случайное сообщение не уничтожало контекст продолжения.
interrupted_jobs = {}
MAX_CONVERSATION_TOPICS = 2
jobs_lock = threading.RLock()

# Небольшой серверный кэш снижает число одинаковых запросов к погодным API.
WEATHER_CACHE = {}
WEATHER_FAILURE_CACHE = {}
WEATHER_CACHE_LOCK = threading.Lock()

# Upstash Redis REST: persistent storage for pending and interrupted answers.
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_STATE_KEY = "deepseek-alisa:conversation-jobs:v1"
REDIS_STATE_LOCK = threading.Lock()
REDIS_AVAILABLE = bool(UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN)


# Необязательный мост к личной Яндекс Станции.
# Для запуска навыка используется Quasar text_action, а не phrase_action.
TTS_BRIDGE_URL = os.getenv("TTS_BRIDGE_URL", "").rstrip("/")
TTS_BRIDGE_SECRET = os.getenv("TTS_BRIDGE_SECRET", "")
AUTO_WAIT_COMMAND = "Алиса, я жду"
AUTO_WAIT_INITIAL_DELAY = 5
AUTO_WAIT_REPEAT_INTERVAL = 10
AUTO_WAIT_COMMAND_COOLDOWN = 15


def send_bridge_wait_command(job_key, job_id):
    """Передаёт станции команду навыку и не произносит её как обычную фразу."""
    if not TTS_BRIDGE_URL or not TTS_BRIDGE_SECRET:
        logger.warning("Automatic wait command skipped: TTS bridge is not configured")
        return False

    now = time.time()
    with jobs_lock:
        current = search_jobs.get(job_key)
        if (
            not current
            or current.get("job_id") != job_id
            or not current.get("auto_delivery")
            or current.get("status") not in ("pending", "ready")
        ):
            return False

        # Если первый фрагмент уже выдан, повторная автоматическая команда
        # не должна случайно запросить следующий фрагмент ответа.
        if current.get("status") == "ready" and int(current.get("part_index", 0)) > 0:
            return False

        last_sent = float(current.get("auto_wait_last_sent_at", 0) or 0)
        if now - last_sent < AUTO_WAIT_COMMAND_COOLDOWN:
            return False

        current["auto_wait_last_sent_at"] = now
        persist_job_maps()

    try:
        response = requests.post(
            f"{TTS_BRIDGE_URL}/api/bridge/command",
            headers={"X-Bridge-Secret": TTS_BRIDGE_SECRET},
            json={"text": AUTO_WAIT_COMMAND},
            timeout=25,
        )
        response.raise_for_status()
        logger.info("Automatic text_action command sent for job %s", job_id[:8])
        return True
    except Exception:
        logger.exception("Could not send automatic text_action wait command")
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job_id:
                if current.get("auto_wait_last_sent_at") == now:
                    current["auto_wait_last_sent_at"] = 0
                    persist_job_maps()
        return False


def monitor_delayed_job(job_key, job_id):
    """Через 12 секунд запускает проверку навыка и повторяет её, пока задача активна."""
    def run():
        time.sleep(AUTO_WAIT_INITIAL_DELAY)
        while True:
            with jobs_lock:
                current = search_jobs.get(job_key)
                if (
                    not current
                    or current.get("job_id") != job_id
                    or not current.get("auto_delivery")
                ):
                    return
                status = current.get("status")
                part_index = int(current.get("part_index", 0))

            if status == "ready":
                if part_index == 0:
                    send_bridge_wait_command(job_key, job_id)
                return
            if status != "pending":
                return

            send_bridge_wait_command(job_key, job_id)
            time.sleep(AUTO_WAIT_REPEAT_INTERVAL)

    threading.Thread(
        target=run,
        daemon=True,
        name=f"auto-wait-{job_id[:8]}",
    ).start()


def auto_deliver_ready_job(job_key, job_id):
    """Когда фоновый ответ готов, вызывает навык через text_action."""
    with jobs_lock:
        current = search_jobs.get(job_key)
        if (
            not current
            or current.get("job_id") != job_id
            or current.get("auto_delivery_thread_started")
        ):
            return
        current["auto_delivery_thread_started"] = True
        persist_job_maps()

    def run():
        # Не отправляем команду раньше чем через 12 секунд после начального ответа.
        with jobs_lock:
            current = search_jobs.get(job_key)
            if not current or current.get("job_id") != job_id:
                return
            started_at = float(current.get("auto_wait_started_at", time.time()) or time.time())
        first_command_at = started_at + AUTO_WAIT_INITIAL_DELAY
        time.sleep(max(0, first_command_at - time.time()))

        # Если в это время уже была отправлена команда ожидания, ждём окончания
        # короткого интервала защиты от повторного запроса следующей части.
        for attempt in range(6):
            with jobs_lock:
                current = search_jobs.get(job_key)
                if (
                    not current
                    or current.get("job_id") != job_id
                    or not current.get("auto_delivery")
                    or current.get("status") != "ready"
                    or int(current.get("part_index", 0)) > 0
                ):
                    return
            if send_bridge_wait_command(job_key, job_id):
                return
            time.sleep(5)

    threading.Thread(
        target=run,
        daemon=True,
        name=f"auto-deliver-{job_id[:8]}",
    ).start()



def redis_command(command):
    """Executes one Upstash Redis REST command; never logs credentials."""
    if not REDIS_AVAILABLE:
        raise RuntimeError("Upstash Redis environment variables are missing")
    response = requests.post(
        UPSTASH_REDIS_REST_URL,
        headers={
            "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
            "Content-Type": "application/json",
        },
        json=command,
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError("Upstash Redis command failed")
    return payload.get("result")


def persist_job_maps():
    """Persists current job state; memory remains a fallback if Redis is unavailable."""
    if not REDIS_AVAILABLE:
        return False
    try:
        # Keep the state lock through the write so an older snapshot cannot
        # overtake a newer one when multiple webhook/background threads save.
        with jobs_lock:
            snapshot = {
                "search_jobs": search_jobs,
                "interrupted_jobs": interrupted_jobs,
                "saved_at": int(time.time()),
            }
            serialized = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
            with REDIS_STATE_LOCK:
                redis_command(["SET", REDIS_STATE_KEY, serialized, "EX", 604800])
        return True
    except Exception as exc:
        logger.warning("Redis state save failed; continuing with in-memory state: %s", exc)
        return False


def load_job_maps():
    """Restores pending and interrupted conversation jobs after a Render restart."""
    if not REDIS_AVAILABLE:
        logger.warning("Upstash Redis is not configured; conversation state is memory-only")
        return
    try:
        with REDIS_STATE_LOCK:
            raw = redis_command(["GET", REDIS_STATE_KEY])
        if not raw:
            logger.info("Redis state store is empty; starting with fresh conversation state")
            return
        saved = json.loads(raw)
        restored_search = saved.get("search_jobs", {})
        restored_interrupted_raw = saved.get("interrupted_jobs", {})
        if not isinstance(restored_search, dict) or not isinstance(restored_interrupted_raw, dict):
            raise ValueError("Unexpected Redis state format")

        # Older deployments stored one interrupted job as an object. Migrate it
        # in memory to the new list format without losing the saved answer.
        restored_interrupted = {}
        for key, value in restored_interrupted_raw.items():
            if isinstance(value, dict):
                restored_interrupted[key] = [value] if value else []
            elif isinstance(value, list):
                restored_interrupted[key] = [item for item in value if isinstance(item, dict)][-MAX_CONVERSATION_TOPICS:]
            else:
                restored_interrupted[key] = []

        # Background Python threads do not survive a Render restart.
        # Keep completed/interrupted answers, but mark orphaned pending jobs as failed
        # instead of leaving the user waiting forever for work that no longer exists.
        for saved_job in restored_search.values():
            if saved_job.get("status") == "pending":
                saved_job["status"] = "error"
                saved_job["answer"] = ""
        for saved_jobs in restored_interrupted.values():
            for saved_job in saved_jobs:
                if saved_job.get("status") == "pending":
                    saved_job["status"] = "error"
                    saved_job["answer"] = ""
        # An active answer plus one interrupted answer is the two-topic limit.
        for key in restored_search:
            if key in restored_interrupted:
                restored_interrupted[key] = restored_interrupted[key][-1:]
        with jobs_lock:
            search_jobs.update(restored_search)
            interrupted_jobs.update(restored_interrupted)
        if restored_search or restored_interrupted:
            persist_job_maps()
        logger.info(
            "Restored conversation state from Redis: %s active, %s interrupted",
            len(restored_search), len(restored_interrupted),
        )
    except Exception as exc:
        logger.error("Could not restore conversation state from Redis: %s", exc)



def _weather_cached_json(
    url, params, headers=None, timeout=15, ttl=1800, failure_ttl=0
):
    key = (url, tuple(sorted((str(k), str(v)) for k, v in params.items())))
    now = time.monotonic()
    with WEATHER_CACHE_LOCK:
        cached = WEATHER_CACHE.get(key)
        if cached and now - cached[0] < ttl:
            logger.info("Weather cache hit: %s", url)
            return cached[1]
        failed = WEATHER_FAILURE_CACHE.get(key)
        if failed and now - failed[0] < failure_ttl:
            raise requests.HTTPError(
                f"Cached HTTP {failed[1]} response for {url}"
            )
    response = requests.get(
        url, params=params, headers=headers or {}, timeout=timeout
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        status = getattr(response, "status_code", None)
        if failure_ttl and status == 429:
            with WEATHER_CACHE_LOCK:
                WEATHER_FAILURE_CACHE[key] = (time.monotonic(), status)
        raise
    data = response.json()
    with WEATHER_CACHE_LOCK:
        WEATHER_CACHE[key] = (time.monotonic(), data)
        WEATHER_FAILURE_CACHE.pop(key, None)
    return data



def limit_text(text, limit=1024):
    text = str(text or "").strip()

    if len(text) <= limit:
        return text

    shortened = text[:limit - 3]

    # Сначала ищем конец последнего законченного предложения.
    endings = [
        shortened.rfind("."),
        shortened.rfind("!"),
        shortened.rfind("?"),
    ]

    sentence_end = max(endings)

    # Если предложение заканчивается достаточно близко к лимиту,
    # сохраняем его целиком.
    if sentence_end >= limit * 0.6:
        return shortened[:sentence_end + 1].rstrip()

    # Иначе ищем последнюю границу между предложениями или абзацами.
    for separator in ("\n", ";", ",", " "):
        position = shortened.rfind(separator)
        if position >= limit * 0.6:
            return shortened[:position].rstrip(" ,;:-") + "..."

    return shortened.rstrip(" ,;:-") + "..."

def normalize_command(text):
    text = str(text or "").lower().replace("ё", "е").strip()
    return " ".join(text.split()).strip(" .,!?:;\"'«»")


def remove_source_links(text):
    """Убирает URL и явные предложения перейти к источникам из ответа."""
    text = str(text or "")
    # Убираем Markdown-ссылки, оставляя только их подпись.
    text = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", text)
    # Убираем прямые URL.
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:URL|ссылка):\s*", "", text, flags=re.IGNORECASE)
    # Удаляем отдельные строки со ссылками/предложением показать источники.
    kept_lines = []
    for line in text.splitlines():
        normalized = line.lower().replace("ё", "е")
        if (
            any(phrase in normalized for phrase in (
                "ссылки на источники", "нужны ссылки", "показать источники",
                "перейдите по ссылке", "читайте подробнее по ссылке",
            ))
            or normalized.strip().startswith(("url:", "ссылка:"))
        ):
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_wake_command(text):
    """Имя помощника, которым можно запросить готовый ответ."""
    return normalize_command(text) in ("алиса")


def is_wait_command(text):
    """Команда получения ответа после ожидания; учитываем перехват слова «Алиса» Яндексом."""
    command = normalize_command(text).replace(",", " ")
    command = " ".join(command.split())
    return command in ("алиса я жду", "я жду")


def pending_job_notice(job):
    """Не повторяет «Секундочку» при каждом обращении к ещё работающей задаче."""
    if not job.get("pending_notice_sent", False):
        job["pending_notice_sent"] = True
        persist_job_maps()
        return "Секундочку"

    return "Ответ ещё готовится."


def is_continue_command(text):
    """Распознаёт короткое подтверждение продолжения текущей темы."""
    command = normalize_command(text)
    if command in ("да", "продолжай", "продолжи", "продолжить"):
        return True
    return bool(re.match(r"^(?:алиса\s+)?(?:продолжи|продолжай|продолжить)\b", command))


def is_topic_return_command(text):
    """Распознаёт просьбу вернуться к ранее начатой теме."""
    # Яндекс может передать обращение «Алиса» вместе с запятой.
    command = normalize_command(text).replace(",", " ")
    command = " ".join(command.split())
    return bool(re.match(
        r"^(?:алиса\s+)?(?:(?:пожалуйста)\s+)?(?:вернись|вернуться|возвратись)\s+(?:(?:пожалуйста)\s+)?(?:к|ко)\b",
        command,
    )) or bool(re.match(
        r"^(?:алиса\s+)?(?:продолжи|продолжай|продолжить)\s+(?:тему|рассказ|ответ|вопрос)\b",
        command,
    ))


def interrupted_list(job_key):
    """Возвращает список сохранённых тем и мигрирует старый формат при необходимости."""
    saved = interrupted_jobs.get(job_key)
    if isinstance(saved, dict):
        saved = [saved]
        interrupted_jobs[job_key] = saved
    elif not isinstance(saved, list):
        saved = []
        interrupted_jobs[job_key] = saved
    return saved


def save_interrupted_job(job_key, job):
    """Добавляет/обновляет прерванную тему, не допуская дубликатов."""
    if not job:
        return
    saved = interrupted_list(job_key)
    job_id = job.get("job_id")
    saved[:] = [item for item in saved if item.get("job_id") != job_id]
    saved.append(job)
    del saved[:-MAX_CONVERSATION_TOPICS]
    interrupted_jobs[job_key] = saved


def trim_interrupted_for_active(job_key):
    """Оставляет одну прерванную тему рядом с активной: всего не более двух."""
    saved = interrupted_list(job_key)
    if len(saved) > MAX_CONVERSATION_TOPICS - 1:
        del saved[:len(saved) - (MAX_CONVERSATION_TOPICS - 1)]
    if saved:
        interrupted_jobs[job_key] = saved
    else:
        interrupted_jobs.pop(job_key, None)


_TOPIC_STOP_WORDS = {
    "алиса", "пожалуйста", "вернись", "вернуться", "возвратись", "верни", "к", "ко",
    "предыдущему", "предыдущей", "предыдущий", "предыдущая", "вопросу", "вопрос", "ответу",
    "ответ", "рассказу", "рассказ", "теме", "тему", "тема", "про", "об", "обо", "о", "по",
    "и", "продолжи", "продолжай", "продолжить", "с", "того", "места", "где", "остановилась",
    "остановился", "остановились", "мы", "на", "чем", "именно", "прошлому", "прошлой",
    "начатому", "начатой", "мне", "расскажи", "рассказать", "подробно", "пожалуйста",
    "прежней", "прежнюю", "прежний", "прежняя", "прежнему", "прежнее",
    "прошлую", "прошлый", "прошлое", "прошлому",
}


def topic_tokens(text):
    normalized = normalize_command(text).replace("ё", "е")
    meaningful_short_tokens = {"ии", "ai", "сша", "нлп", "аги"}
    tokens = [
        token for token in re.findall(r"[a-zа-я0-9]+", normalized)
        if (len(token) >= 4 or token in meaningful_short_tokens)
        and token not in _TOPIC_STOP_WORDS
    ]
    # Считаем «ИИ» и полную формулировку «искусственный интеллект» одной темой.
    if (
        any(token.startswith("искусствен") for token in tokens)
        and any(token.startswith("интеллект") for token in tokens)
    ):
        tokens.append("ии")
    return tokens


def _topic_words_match(left, right):
    if left == right:
        return True
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    return common >= 4


def choose_interrupted_topic(user_text, candidates):
    """Выбирает сохранённый ответ по словам темы; общий возврат берёт самый свежий."""
    if not candidates:
        return None
    query_tokens = topic_tokens(user_text)
    if not query_tokens:
        return candidates[-1]

    best_job = None
    best_score = 0.0
    for candidate in candidates:
        saved_tokens = list(dict.fromkeys(topic_tokens(candidate.get("user_text", ""))))
        if not saved_tokens:
            continue
        matched = sum(
            1 for saved_word in saved_tokens
            if any(_topic_words_match(saved_word, query_word) for query_word in query_tokens)
        )
        score = matched / len(saved_tokens)
        if score > best_score:
            best_score = score
            best_job = candidate
    return best_job if best_score >= 0.25 else None


def is_continuation_repair_request(text):
    """Распознаёт пояснение, что пользователь хотел подтвердить продолжение."""
    command = normalize_command(text)
    correction = any(phrase in command for phrase in (
        "я ошибся",
        "я ошиблась",
        "я хотел сказать",
        "я хотела сказать",
        "я имел в виду",
        "я имела в виду",
        "оговорился",
        "оговорилась",
    ))
    explicit_continue = any(word in command for word in (
        "продолжи",
        "продолжай",
        "продолжить",
    ))
    return correction and ("да" in command or explicit_continue)


def is_short_ambiguous_interruption(text):
    """Короткое сообщение может быть оговоркой, а не новым полноценным запросом."""
    command = normalize_command(text)
    return len(command) <= 24 and len(command.split()) <= 2


def is_stop_command(text):
    """Команды отказа от продолжения текущего многочастного ответа."""
    command = normalize_command(text)
    command = re.sub(r"^(?:алиса)\s+", "", command).strip()
    return command in (
        "нет",
        "не надо",
        "не нужно",
        "хватит",
        "стоп",
        "остановись",
        "не продолжай",
        "не продолжать",
    )


def get_social_reply(text):
    """Мгновенные ответы на благодарность, приветствия и короткие вопросы."""
    normalized = normalize_command(text)
    normalized = " ".join(re.sub(r"[^a-zа-я0-9 ]+", " ", normalized).split())

    # Убираем обращение «Алиса» в начале фразы, чтобы варианты
    # «Алиса, спасибо» и «Алиса, как дела?» обрабатывались мгновенно.
    question = re.sub(r"^алиса\s+", "", normalized, count=1).strip()

    thanks_phrases = {
        "спасибо",
        "спасибо большое",
        "большое спасибо",
        "спасибо тебе",
        "спасибо большое тебе",
        "благодарю",
        "очень благодарен",
        "очень благодарна",
        "спасибо алиса",
        "спасибо большое алиса",
        "благодарю алиса",
    }
    if question in thanks_phrases:
        return "Пожалуйста! Рада была помочь!"

    # Убираем приветствие и обращение к Алисе,
    # чтобы распознавались сочетания вроде «Алиса, привет, как дела?».
    greeting_prefix = re.compile(
        r"^(?:алиса\s+)?"
        r"(?:привет|здравствуй|здравствуйте|добрый день|доброе утро|добрый вечер)"
        r"(?:\s+алиса)?(?:\s+|$)"
    )
    while True:
        cleaned = greeting_prefix.sub("", question, count=1).strip()
        if cleaned == question:
            break
        question = cleaned
        question = re.sub(r"^алиса\s+", "", question).strip()

    # Отдельное приветствие — тоже мгновенный локальный ответ.
    standalone_greeting = re.fullmatch(
        r"(?:алиса\s+)?(?:привет|здравствуй|здравствуйте|добрый день|доброе утро|добрый вечер)(?:\s+алиса)?",
        normalized,
    )
    if standalone_greeting:
        return "Привет! Чем могу помочь?"

    how_are_you_phrases = {
        "как дела",
        "как твои дела",
        "как у тебя дела",
        "как поживаешь",
        "как ты",
        "как дела алиса",
        "алиса как дела",
        "как твои дела алиса",
    }
    if question in how_are_you_phrases:
        return "У меня всё отлично! Спасибо, что интересуешься!"

    where_phrases = {
        "а это где",
        "где где",
    }
    if question in where_phrases:
        return "Где-где. В Караганде, где живёт конь в пальто"

    who_phrases = {
        "а это кто",
    }
    if question in who_phrases:
        return "Кто-кто. Конь в пальто, который живёт в Караганде"


def is_joke_request(text):
    """Определяет просьбы рассказать анекдот или шутку."""
    normalized = normalize_command(text)
    return any(phrase in normalized for phrase in (
        "анекдот", "шутку", "шутка", "пошути", "рассмеши"
    ))


def split_answer_into_parts(answer, chunk_size=300):
    """Делит длинный ответ на законченные голосовые фрагменты."""
    text = str(answer or "").strip()
    text = re.sub(r"\s*(?:Продолжить\?|Продолжение следует\.?)[.!?]*\s*$", "", text, flags=re.IGNORECASE)
    if not text:
        return [""]
    if len(text) <= 380:
        return [text]

    # Сначала делим по предложениям, сохраняя знаки конца предложения.
    sentences = re.findall(r".*?(?:[.!?…](?=\s|$)|$)", text, flags=re.DOTALL)
    sentences = [part.strip() for part in sentences if part and part.strip()]
    if not sentences:
        sentences = [text]

    parts = []
    current = ""

    def add_long_sentence(sentence):
        words = sentence.split()
        piece = ""
        for word in words:
            candidate = (piece + " " + word).strip()
            if len(candidate) > chunk_size and piece:
                parts.append(piece)
                piece = word
            else:
                piece = candidate
        if piece:
            parts.append(piece)

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                parts.append(current.strip())
                current = ""
            add_long_sentence(sentence)
            continue

        candidate = (current + " " + sentence).strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                parts.append(current.strip())
            current = sentence

    if current:
        parts.append(current.strip())

    # Если регулярное выражение не смогло корректно выделить предложения,
    # гарантируем непустые части и разрезаем только по границе слова.
    if not parts:
        words = text.split()
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if len(candidate) > chunk_size and current:
                parts.append(current)
                current = word
            else:
                current = candidate
        if current:
            parts.append(current)

    return parts or [text]


def prepare_job_parts(job, answer):
    """Подготавливает части, если вопрос или сам ответ достаточно объёмный."""
    answer = str(answer or "").strip()
    user_text = job.get("user_text", "")
    should_split = is_complex_question(user_text) or len(answer) > 380
    parts = split_answer_into_parts(answer) if should_split else [answer]
    job["answer"] = answer
    job["parts"] = parts
    job["part_index"] = 0
    return parts


def deliver_job_part(body, job_key, job):
    """Возвращает очередную часть и сохраняет задачу до последней части."""
    with jobs_lock:
        current = search_jobs.get(job_key)
        if not current or current.get("job_id") != job.get("job_id"):
            return make_response(body, "Не удалось найти подготовленный ответ. Повтори вопрос, пожалуйста.")

        if not current.get("parts"):
            prepare_job_parts(current, current.get("answer", ""))

        parts = current.get("parts") or [current.get("answer", "")]
        index = int(current.get("part_index", 0))
        if index >= len(parts):
            search_jobs.pop(job_key, None)
            persist_job_maps()
            return make_response(body, "На этом всё.")

        answer = parts[index]
        is_last = index == len(parts) - 1
        if not is_last:
            answer += " Продолжить?"
            current["part_index"] = index + 1
            persist_job_maps()
        else:
            original_text = current.get("user_text", "")
            original_history = current.get("history", [])
            complete_answer = current.get("answer", "")
            new_history = original_history[-3:] + [
                {"role": "user", "content": original_text[:400]},
                {"role": "assistant", "content": complete_answer[:400]},
            ]
            search_jobs.pop(job_key, None)
            persist_job_maps()
            return make_response(body, answer, new_history)

    return make_response(body, answer)

def is_stock_quote_request(text):
    """Распознаёт запросы о биржевых ценах акций."""
    normalized = normalize_command(text)
    if any(word in normalized for word in (
        "котиров", "тикер", "биржевую цену", "биржевая цена",
    )):
        return True
    return bool(
        re.search(r"(?:сколько\s+стоит|цен[аыуе]|стоимост[ьи]|курс)\s+.{0,35}акци", normalized)
        or re.search(r"акци\w*\s+.{0,35}(?:цен[аыуе]|стоимост[ьи]|курс|котиров)", normalized)
    )


def is_currency_rate_request(text):
    """Распознаёт запросы официального курса иностранных валют к рублю."""
    normalized = normalize_command(text)
    has_currency = bool(re.search(
        r"\b(?:доллар\w*|usd|евро|eur|юан\w*|cny|фунт\w*|gbp)\b",
        normalized,
    ))
    has_rate_intent = any(word in normalized for word in (
        "курс", "сколько стоит", "стоимость", "цена", "котиров",
    ))
    return has_currency and has_rate_intent and not is_stock_quote_request(normalized)


def is_weather_request(text):
    """Распознаёт самостоятельные вопросы о погоде и прогнозе."""
    normalized = normalize_command(text)
    weather_terms = (
        "погод", "прогноз", "дожд", "ливень", "снег", "снегопад",
        "гроза", "замороз", "температур", "ветер", "зонт",
    )
    return any(term in normalized for term in weather_terms) and (
        any(term in normalized for term in (
            "погод", "прогноз", "дожд", "ливень", "снег", "снегопад",
            "гроза", "замороз", "температур", "ветер", "брать зонт", "нужен зонт",
            "нужно ли брать зонт",
        ))
    )


def is_oil_request(text):
    """Распознаёт запросы о котировках нефти Brent, WTI и Urals."""
    normalized = str(text or "").lower().replace("ё", "е")
    oil_names = (
        "brent", "брент",
        "wti", "даблъютиай", "дабл ю ти ай", "дабл-ю-ти-ай",
        "urals", "уралс", "юралс", "юралз",
        "нефть",
    )
    return any(name in normalized for name in oil_names)


def needs_search(text):
    text = str(text or "").lower().replace("ё", "е")
    return (
        "найди в интернете" in text
        or "посмотри в интернете" in text
        or is_stock_quote_request(text)
        or is_currency_rate_request(text)
        or is_weather_request(text)
        or is_oil_request(text)
    )


def extract_stock_query(user_text):
    """Оставляет в запросе название компании или биржевой тикер."""
    query = str(user_text or "").strip()
    query = re.sub(
        r"(?i)\bнайди\s+в\s+интернете\b|\bпосмотри\s+в\s+интернете\b",
        " ", query,
    )
    query = re.sub(
        r"(?i)\b(последн\w*|предыдущ\w*|прошл\w*|историческ\w*|текущ\w*|актуальн\w*|свеж\w*|котиров\w*|"
        r"акци\w*|бумаг\w*|стоимост\w*|цен\w*|курс\w*|биржев\w*|"
        r"найди|покажи|скажи|какая|какие|сколько|стоит|сейчас|на сегодня|"
        r"за сегодня|на данный момент)\b",
        " ", query,
    )
    query = re.sub(r"[?!.:,;]+", " ", query)
    return " ".join(query.split()).strip()


def _moex_table_rows(payload, block_name):
    block = payload.get(block_name, {}) if isinstance(payload, dict) else {}
    columns = block.get("columns", [])
    rows = block.get("data", [])
    parsed = []
    for values in rows:
        if not isinstance(values, list):
            continue
        item = dict(zip(columns, values))
        # ISS использует разный регистр названий полей в разных блоках.
        for key, value in list(item.items()):
            item.setdefault(str(key).lower(), value)
            item.setdefault(str(key).upper(), value)
        parsed.append(item)
    return parsed


def fetch_moex_stock_quote(company_query):
    """Ищет российскую акцию и получает последнюю цену из MOEX ISS."""
    headers = {"User-Agent": "Mozilla/5.0 DeepSeekAlice/1.0"}
    terms = [company_query]
    # Русская речь часто использует родительный падеж: «акции Газпрома».
    if len(company_query) > 4 and company_query[-1:].lower() in ("а", "я", "ы", "и"):
        terms.append(company_query[:-1])

    for term in dict.fromkeys(t for t in terms if t):
        search_response = requests.get(
            "https://iss.moex.com/iss/securities.json",
            params={"q": term, "iss.meta": "off"},
            headers=headers,
            timeout=12,
        )
        search_response.raise_for_status()
        candidates = _moex_table_rows(search_response.json(), "securities")
        if not candidates:
            continue

        # Основной режим торгов акциями на MOEX — TQBR.
        candidates.sort(key=lambda item: (
            0 if str(item.get("primary_boardid", "")).upper() == "TQBR" else 1,
            0 if str(item.get("is_trading", "0")) in ("1", "True", "true") else 1,
        ))

        for candidate in candidates[:12]:
            secid = str(candidate.get("secid") or "").strip().upper()
            if not secid:
                continue
            quote_response = requests.get(
                f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{secid}.json",
                params={"iss.meta": "off", "iss.only": "securities,marketdata"},
                headers=headers,
                timeout=12,
            )
            if quote_response.status_code == 404:
                continue
            quote_response.raise_for_status()
            payload = quote_response.json()
            market_rows = _moex_table_rows(payload, "marketdata")
            if not market_rows:
                continue

            market_rows.sort(key=lambda item: (
                0 if str(item.get("BOARDID", "")).upper() == "TQBR" else 1,
                0 if item.get("LAST") is not None else 1,
            ))
            market = market_rows[0]
            price_key = next((key for key in (
                "LAST", "LCURRENTPRICE", "MARKETPRICE", "PREVPRICE"
            ) if market.get(key) not in (None, "", 0, "0")), None)
            if not price_key:
                continue

            security_rows = _moex_table_rows(payload, "securities")
            security = security_rows[0] if security_rows else candidate
            price = float(market[price_key])
            currency = str(security.get("CURRENCYID") or "RUB").upper()
            # MOEX иногда возвращает исторический код SUR для российского рубля.
            if currency in ("SUR", "RUR"):
                currency = "RUB"
            name = str(
                security.get("SHORTNAME") or security.get("NAME")
                or candidate.get("shortname") or candidate.get("name") or secid
            )
            date_value = (
                market.get("LASTTRADEDATE") or market.get("PREVDATE")
                or security.get("PREVDATE")
            )
            time_value = market.get("TIME") or market.get("UPDATETIME")
            return {
                "source": "Московская биржа",
                "symbol": secid,
                "name": name,
                "price": price,
                "currency": currency,
                "date": str(date_value or ""),
                "time": str(time_value or ""),
                "price_type": {
                    "LAST": "цена последней сделки",
                    "LCURRENTPRICE": "текущая цена",
                    "MARKETPRICE": "рыночная цена",
                    "PREVPRICE": "цена закрытия предыдущей сессии",
                }.get(price_key, "последняя доступная цена"),
                "delayed": True,
                "decimals": int(security.get("DECIMALS") or candidate.get("decimals") or 2),
            }
    return None


def fetch_yahoo_stock_quote(company_query):
    """Находит зарубежную акцию и получает цену через Yahoo Finance chart API."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
        "Accept": "application/json",
    }
    search_response = requests.get(
        "https://query1.finance.yahoo.com/v1/finance/search",
        params={"q": company_query, "quotesCount": 12, "newsCount": 0},
        headers=headers,
        timeout=12,
    )
    search_response.raise_for_status()
    quotes = search_response.json().get("quotes", [])
    equities = [item for item in quotes if item.get("quoteType") == "EQUITY" and item.get("symbol")]
    if not equities:
        return None

    query_norm = re.sub(r"[^a-zа-я0-9]", "", company_query.lower().replace("ё", "е"))
    equities.sort(key=lambda item: (
        0 if str(item.get("symbol", "")).lower() == company_query.lower() else 1,
        0 if query_norm and query_norm in re.sub(r"[^a-zа-я0-9]", "", str(item.get("longname") or item.get("shortname") or "").lower().replace("ё", "е")) else 1,
    ))

    for match in equities[:5]:
        symbol = str(match["symbol"])
        chart_response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "5d", "interval": "1d", "includePrePost": "false"},
            headers=headers,
            timeout=12,
        )
        chart_response.raise_for_status()
        chart = chart_response.json().get("chart", {})
        results = chart.get("result") or []
        if not results:
            continue
        meta = results[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is None:
            continue

        timestamp = meta.get("regularMarketTime")
        quote_time = ""
        if timestamp:
            try:
                zone_name = meta.get("exchangeTimezoneName")
                tz = ZoneInfo(zone_name) if zone_name else timezone.utc
                quote_time = datetime.fromtimestamp(int(timestamp), tz=tz).strftime("%d.%m.%Y, %H:%M")
            except Exception:
                quote_time = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%d.%m.%Y, %H:%M UTC")

        return {
            "source": "Yahoo Finance",
            "symbol": symbol,
            "name": str(meta.get("longName") or meta.get("shortName") or match.get("longname") or match.get("shortname") or symbol),
            "price": float(price),
            "currency": str(meta.get("currency") or "валюта не указана"),
            "date": quote_time,
            "time": "",
            "price_type": "последняя доступная биржевая котировка",
            "delayed": False,
            "decimals": int(meta.get("priceHint") or 2),
        }
    return None


def format_stock_quote(quote):
    decimals = max(0, min(int(quote.get("decimals", 2)), 6))
    price = f"{quote['price']:,.{decimals}f}".replace(",", " ").replace(".", ",")
    currency_code = str(quote.get("currency") or "").upper()
    if currency_code in ("SUR", "RUR"):
        currency_code = "RUB"
    currency_names = {"RUB": "рубля", "USD": "доллара США", "EUR": "евро", "GBP": "фунта стерлингов", "HKD": "гонконгских долларов", "CNY": "юаня"}
    currency = currency_names.get(currency_code, currency_code or "валюта не указана")
    answer = (
        f"{quote['name']} ({quote['symbol']}): {quote['price_type']} — "
        f"{price} {currency} за акцию."
    )
    quote_date = quote.get("date", "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", quote_date):
        quote_date = datetime.strptime(quote_date, "%Y-%m-%d").strftime("%d.%m.%Y")
    if quote_date and quote.get("time"):
        answer += f" Дата и время последней сделки: {quote_date}, {quote['time']}."
    elif quote_date:
        answer += f" Дата котировки: {quote_date}."
    elif quote.get("time"):
        answer += f" Время последнего обновления: {quote['time']}."
    else:
        answer += " Точное время сделки источник не указал."
    answer += f" Источник: {quote['source']}."
    if quote.get("delayed"):
        answer += " Бесплатные данные Московской биржи могут поступать с задержкой до 15 минут."
    return answer


def get_stock_quote_answer(user_text, history=None):
    company_query = extract_stock_query(user_text)
    if not company_query and history:
        # Для уточнений вроде «найди предыдущие котировки» берём компанию
        # из недавнего запроса пользователя, если она там явно названа.
        for item in reversed(history[-4:]):
            if isinstance(item, dict) and item.get("role") == "user":
                previous_text = str(item.get("content", ""))
                if is_stock_quote_request(previous_text):
                    company_query = extract_stock_query(previous_text)
                    if company_query:
                        break
    if not company_query:
        return "Уточни, акции какой компании тебя интересуют."

    # Кириллические названия в первую очередь ищем на Московской бирже.
    is_cyrillic = bool(re.search(r"[а-яё]", company_query, flags=re.IGNORECASE))
    providers = (fetch_moex_stock_quote, fetch_yahoo_stock_quote) if is_cyrillic else (fetch_yahoo_stock_quote, fetch_moex_stock_quote)
    errors = []
    for provider in providers:
        try:
            quote = provider(company_query)
            if quote:
                return format_stock_quote(quote)
        except Exception as exc:
            logger.warning("Stock quote provider %s failed: %s", provider.__name__, exc)
            errors.append(provider.__name__)

    return (
        f"Не удалось надёжно найти биржевую котировку для «{company_query}». "
        "Попробуй назвать тикер акции или уточнить биржу."
    )


def extract_currency_code(text):
    normalized = normalize_command(text)
    if re.search(r"\b(?:доллар\w*|usd)\b", normalized):
        return "USD", "доллар США", "долларов США"
    if re.search(r"\b(?:евро|eur)\b", normalized):
        return "EUR", "евро", "евро"
    if re.search(r"\b(?:юан\w*|cny)\b", normalized):
        return "CNY", "юань", "юаней"
    if re.search(r"\b(?:фунт\w*|gbp)\b", normalized):
        return "GBP", "фунт стерлингов", "фунтов стерлингов"
    return None, None, None


def _parse_cbr_currency_xml(xml_content, currency_code):
    """Извлекает курс валюты и дату, с которой он действует, из XML ЦБ РФ."""
    root = ET.fromstring(xml_content)
    effective_date_text = root.attrib.get("Date", "")
    effective_date = (
        datetime.strptime(effective_date_text, "%d.%m.%Y").date()
        if effective_date_text else None
    )

    for item in root.findall("Valute"):
        code = (item.findtext("CharCode") or "").strip().upper()
        if code != currency_code.upper():
            continue
        nominal_text = (item.findtext("Nominal") or "1").strip()
        value_text = (item.findtext("Value") or "").strip().replace(",", ".")
        nominal = float(nominal_text)
        value = float(value_text)
        if nominal <= 0:
            raise ValueError("Некорректный номинал валюты в ответе Банка России")
        return {
            "code": code,
            "name": (item.findtext("Name") or code).strip(),
            "value": value / nominal,
            "nominal": nominal,
            "date": effective_date.strftime("%d.%m.%Y") if effective_date else "",
            "effective_date": effective_date,
        }
    raise LookupError(f"В ответе Банка России нет курса {currency_code}")


def fetch_cbr_currency_rate(currency_code):
    """Получает действующий сегодня и самый новый опубликованный курс ЦБ РФ.

    Запрос с date_req возвращает курс на выбранную дату. Запрос без date_req,
    согласно документации ЦБ РФ, возвращает последнюю зарегистрированную дату;
    это позволяет заметить новый курс, опубликованный сегодня для действия завтра.
    """
    today_moscow = datetime.now(ZoneInfo("Europe/Moscow")).date()
    headers = {"User-Agent": "Mozilla/5.0 DeepSeekAlice/1.0"}
    url = "https://www.cbr.ru/scripts/XML_daily.asp"

    current_response = requests.get(
        url,
        params={"date_req": today_moscow.strftime("%d/%m/%Y")},
        headers=headers,
        timeout=15,
    )
    current_response.raise_for_status()
    current_rate = _parse_cbr_currency_xml(current_response.content, currency_code)

    latest_response = requests.get(
        url,
        headers=headers,
        timeout=15,
    )
    latest_response.raise_for_status()
    latest_rate = _parse_cbr_currency_xml(latest_response.content, currency_code)

    current_rate["today"] = today_moscow
    current_rate["latest"] = latest_rate
    return current_rate


def get_currency_rate_answer(user_text):
    code, currency_singular, currency_plural = extract_currency_code(user_text)
    if not code:
        return "Уточни, курс какой валюты тебя интересует."
    try:
        rate = fetch_cbr_currency_rate(code)
        value = f"{rate['value']:,.4f}".replace(",", " ").replace(".", ",")
        nominal = int(rate["nominal"])
        nominal_phrase = f"за 1 {currency_singular}" if nominal == 1 else f"за {nominal} {currency_plural}"
        answer = (
            f"Официальный курс Банка России на сегодня: {value} рубля {nominal_phrase}."
        )

        latest = rate.get("latest") or {}
        latest_date = latest.get("effective_date")
        today = rate.get("today")
        if latest_date and today and latest_date > today:
            latest_value = f"{latest['value']:,.4f}".replace(",", " ").replace(".", ",")
            latest_nominal = int(latest["nominal"])
            latest_nominal_phrase = (
                f"за 1 {currency_singular}" if latest_nominal == 1
                else f"за {latest_nominal} {currency_plural}"
            )
            answer += (
                f" Сегодня Банк России установил новый курс: {latest_value} рубля "
                f"{latest_nominal_phrase}; он вступит в силу {latest_date.strftime('%d.%m.%Y')}."
            )
        elif rate.get("date"):
            answer += f" Этот курс действует с {rate['date']}."
        return answer
    except Exception as exc:
        logger.warning("CBR currency rate lookup failed for %s: %s", code, exc)
        return (
            f"Не удалось получить последний официальный курс {currency_singular} "
            "с сайта Банка России. Попробуй повторить запрос немного позже."
        )


WEATHER_DEFAULT_LOCATION = {
    "name": "Бурдино",
    "latitude": 52.1373815,
    "longitude": 38.4722745,
    "timezone": "Europe/Moscow",
}


def extract_weather_location(user_text):
    """Извлекает название места и, если указана, страну; иначе использует Бурдино."""
    text = str(user_text or "").strip()
    text = re.sub(r"^алиса[, ]+", "", text, flags=re.IGNORECASE)

    match = re.search(
        r"\b(?:в|для)\s+(?:(?:селе|село|городе|город|деревне|"
        r"поселке|посёлке|поселок|посёлок)\s+)?(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    location = match.group(1).strip(" ,.!?;:")
    stop_pattern = (
        r"\b(?:на ближайшую неделю|на послезавтра|на выходных|"
        r"на выходные|на сегодня|на завтра|на неделю|"
        r"сегодня|завтра|послезавтра|сейчас|в выходные|"
        r"утром|днем|днём|вечером|ночью|пожалуйста)\b"
    )
    location = re.split(stop_pattern, location, maxsplit=1, flags=re.IGNORECASE)[0]
    location = re.sub(
        r"^(?:селе|село|городе|город|деревне|поселке|посёлке)\s+",
        "", location, flags=re.IGNORECASE,
    )
    location = re.sub(
        r"\s+(?:будет ли|будет|ожидается ли|ожидается|какая|какой|какое).*$",
        "", location, flags=re.IGNORECASE,
    ).strip(" ,.!?;:")
    return location or None


# Частые формы названий в русской речи. Для остальных названий
# дополнительно используются аккуратные варианты окончания слова.
WEATHER_LOCATION_ALIASES = {
    "москве": "Москва",
    "санкт-петербурге": "Санкт-Петербург",
    "петербурге": "Санкт-Петербург",
    "питере": "Санкт-Петербург",
    "воронеже": "Воронеж",
    "ростове-на-дону": "Ростов-на-Дону",
    "ростове на дону": "Ростов-на-Дону",
    "ростове": "Ростов",
    "йошкар-оле": "Йошкар-Ола",
    "йошкар оле": "Йошкар-Ола",
    "йошкар-оле": "Йошкар-Ола",
    "липецке": "Липецк",
    "курске": "Курск",
    "орле": "Орёл",
    "тамбове": "Тамбов",
    "туле": "Тула",
    "рязани": "Рязань",
    "казани": "Казань",
    "самаре": "Самара",
    "саратове": "Саратов",
    "волгограде": "Волгоград",
    "краснодаре": "Краснодар",
    "сочи": "Сочи",
    "екатеринбурге": "Екатеринбург",
    "челябинске": "Челябинск",
    "уфе": "Уфа",
    "перми": "Пермь",
    "омске": "Омск",
    "новосибирске": "Новосибирск",
    "красноярске": "Красноярск",
    "иркутске": "Иркутск",
    "владивостоке": "Владивосток",
    "калининграде": "Калининград",
    "париже": "Париж",
    "берлине": "Берлин",
    "лондоне": "Лондон",
    "риме": "Рим",
    "мадриде": "Мадрид",
    "праге": "Прага",
    "вене": "Вена",
    "стамбуле": "Стамбул",
    "нью-йорке": "Нью-Йорк",
}

WEATHER_COUNTRY_CODES = {
    "россия": "RU", "россии": "RU", "рф": "RU",
    "франция": "FR", "франции": "FR",
    "германия": "DE", "германии": "DE",
    "италия": "IT", "италии": "IT",
    "литва": "LT", "литве": "LT",
    "беларусь": "BY", "беларуси": "BY", "белоруссия": "BY",
    "украина": "UA", "украине": "UA",
    "польша": "PL", "польше": "PL",
    "чехия": "CZ", "чехии": "CZ",
    "австрия": "AT", "австрии": "AT",
    "турция": "TR", "турции": "TR",
    "сша": "US", "соединенные штаты": "US",
    "великобритания": "GB", "англия": "GB",
    "испания": "ES", "испании": "ES",
    "греция": "GR", "греции": "GR",
    "китай": "CN", "китае": "CN",
    "япония": "JP", "японии": "JP",
    "казахстан": "KZ", "казахстане": "KZ",
    "армения": "AM", "армении": "AM",
    "грузия": "GE", "грузии": "GE",
}


def _weather_normalize(value):
    return re.sub(r"[^a-zа-я0-9]", "", str(value or "").lower().replace("ё", "е"))


def _weather_city_and_country(location_name):
    """Разделяет город и явно указанную страну (через запятую или предлог)."""
    raw = str(location_name or "").strip(" ,.!?;:")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) > 1:
        city = parts[0]
        country_text = parts[-1].lower()
    else:
        country_match = re.search(
            r"\s+(?:в|во)\s+([а-яёa-z ]+)$", raw, flags=re.IGNORECASE
        )
        if country_match and country_match.group(1).strip().lower() in WEATHER_COUNTRY_CODES:
            city = raw[:country_match.start()].strip()
            country_text = country_match.group(1).strip().lower()
        else:
            city, country_text = raw, ""

    country_text = re.sub(r"^(?:страна\s+)", "", country_text)
    country_code = WEATHER_COUNTRY_CODES.get(country_text)
    return city, country_code


def _weather_query_variants(city):
    variants = [city]
    normalized_key = city.lower().replace("ё", "е")
    alias = WEATHER_LOCATION_ALIASES.get(normalized_key)
    if alias:
        variants.insert(0, alias)

    # Open-Meteo может не находить составные названия с дефисом
    # или русскую транслитерацию. Пробуем несколько эквивалентных форм.
    if normalized_key in ("ростове-на-дону", "ростове на дону",
                          "ростов-на-дону", "ростов на дону"):
        variants.extend(("Ростов-на-Дону", "Ростов на Дону",
                         "Rostov-on-Don", "Rostov-na-Donu", "Rostov"))
    elif normalized_key in ("санкт-петербурге", "санкт-петербург"):
        variants.extend(("Санкт-Петербург", "Санкт Петербург",
                         "Saint Petersburg"))
    elif normalized_key in ("нью-йорке", "нью-йорк"):
        variants.extend(("Нью-Йорк", "New York"))
    elif normalized_key in ("йошкар-оле", "йошкар оле",
                            "йошкар-ола", "йошкар ола"):
        variants.extend(("Йошкар-Ола", "Йошкар Ола", "Yoshkar-Ola", "Yoshkar Ola"))

    # Простые однословные формы: Москве -> Москва, Липецке -> Липецк.
    if " " not in city and "-" not in city:
        lower = city.lower()
        if lower.endswith(("ове", "еве", "ине", "ане")):
            variants.append(city[:-1])
        elif lower.endswith("е"):
            variants.extend((city[:-1], city[:-1] + "а"))
        elif lower.endswith("у"):
            variants.append(city[:-1] + "а")
        elif lower.endswith("ю"):
            variants.append(city[:-1] + "я")
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


def resolve_weather_location(location_name):
    """Находит координаты места через Open-Meteo Geocoding с проверкой страны."""
    if not location_name:
        return dict(WEATHER_DEFAULT_LOCATION)

    city, requested_country = _weather_city_and_country(location_name)
    query_variants = _weather_query_variants(city)

    # Для общеизвестных зарубежных столиц задаём разумный приоритет,
    # чтобы одноимённые небольшие населённые пункты не подменяли город.
    preferred_countries = {
        "москва": "RU", "москве": "RU",
        "воронеж": "RU", "воронеже": "RU",
        "ростов-на-дону": "RU", "ростове-на-дону": "RU",
        "йошкар-ола": "RU", "йошкар-оле": "RU",
        "йошкар ола": "RU", "йошкар оле": "RU",
        "липецк": "RU", "липецке": "RU",
        "париж": "FR", "париже": "FR",
        "берлин": "DE", "берлине": "DE",
        "лондон": "GB", "лондоне": "GB",
        "рим": "IT", "риме": "IT",
        "мадрид": "ES", "мадриде": "ES",
        "прага": "CZ", "праге": "CZ",
        "вена": "AT", "вене": "AT",
        "стамбул": "TR", "стамбуле": "TR",
        "нью-йорк": "US", "нью-йорке": "US",
    }
    preferred_country = requested_country or preferred_countries.get(
        city.lower().replace("ё", "е")
    )

    all_results = []
    seen = set()
    for language in ("ru", "en"):
        for query_variant in query_variants:
            geocoding_data = _weather_cached_json(
                "https://geocoding-api.open-meteo.com/v1/search",
                {
                    "name": query_variant,
                    "count": 10,
                    "language": language,
                    "format": "json",
                },
                headers={"User-Agent": "DeepSeekAlisa/1.0"},
                timeout=12,
                ttl=86400,
            )
            for item in geocoding_data.get("results") or []:
                key = (
                    item.get("id"), item.get("latitude"),
                    item.get("longitude"), item.get("country_code"),
                )
                if key not in seen:
                    seen.add(key)
                    all_results.append(item)
            # Не прекращаем поиск после первого результата: проверяем
            # варианты написания и язык, чтобы выбрать правильную страну.

    if preferred_country:
        matching = [
            item for item in all_results
            if str(item.get("country_code", "")).upper() == preferred_country
        ]
        if not matching:
            country_name = next(
                (name for name, code in WEATHER_COUNTRY_CODES.items()
                 if code == preferred_country), preferred_country
            )
            raise LookupError(
                f"Населённый пункт «{city}» в указанной стране не найден"
            )
        all_results = matching

    if not all_results:
        raise LookupError(f"Населённый пункт «{location_name}» не найден")

    canonical_city = WEATHER_LOCATION_ALIASES.get(
        city.lower().replace("ё", "е"), city
    )
    city_norm = _weather_normalize(canonical_city)
    def rank(item):
        item_name = _weather_normalize(item.get("name", ""))
        feature = item.get("feature_code", "")
        return (
            0 if item_name == city_norm else 1,
            0 if feature in ("PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPL") else 1,
            0 if str(item.get("country_code", "")).upper() == "RU" else 1,
            -int(item.get("population") or 0),
        )

    all_results.sort(key=rank)
    chosen = all_results[0]
    name_parts = [str(chosen.get("name") or city)]
    admin1 = chosen.get("admin1")
    country = chosen.get("country")
    if admin1 and admin1 not in name_parts:
        name_parts.append(str(admin1))
    if country and country not in name_parts:
        name_parts.append(str(country))

    return {
        "name": ", ".join(name_parts),
        "latitude": float(chosen["latitude"]),
        "longitude": float(chosen["longitude"]),
        "timezone": chosen.get("timezone") or "auto",
    }


def weather_description(code):
    """Переводит коды погоды WMO в короткие русские описания."""
    code = int(code)
    if code == 0:
        return "ясно"
    if code == 1:
        return "преимущественно ясно"
    if code == 2:
        return "переменная облачность"
    if code == 3:
        return "пасмурно"
    if code in (45, 48):
        return "туман"
    if code in (51, 53, 55, 56, 57):
        return "морось"
    if code in (61, 63, 65, 66, 67):
        return "дождь"
    if code in (71, 73, 75, 77):
        return "снег"
    if code in (80, 81, 82):
        return "ливень"
    if code in (85, 86):
        return "снегопад"
    if code in (95, 96, 99):
        return "гроза"
    return "переменная погода"


def wind_direction_text(degrees):
    directions = (
        "северный", "северо-восточный", "восточный", "юго-восточный",
        "южный", "юго-западный", "западный", "северо-западный",
    )
    return directions[int((float(degrees) + 22.5) // 45) % 8]


def _met_symbol_to_wmo(symbol):
    """Приближённо сопоставляет погодные символы MET Norway кодам WMO."""
    symbol = str(symbol or "").lower().split("_")[0]
    if symbol.startswith("clearsky"):
        return 0
    if symbol.startswith("fair"):
        return 1
    if symbol.startswith("partlycloudy"):
        return 2
    if symbol.startswith("cloudy"):
        return 3
    if symbol.startswith("fog"):
        return 45
    if "thunder" in symbol:
        return 95
    if "snow" in symbol:
        return 71 if "light" in symbol or "showers" in symbol else 73
    if "sleet" in symbol:
        return 66
    if "rain" in symbol:
        if "heavy" in symbol:
            return 65
        if "showers" in symbol:
            return 80
        return 61
    return 3


def _get_met_weather_data(location):
    """Получает и приводит прогноз MET Norway к структуре Open-Meteo."""
    params = {
        "lat": f"{location['latitude']:.5f}",
        "lon": f"{location['longitude']:.5f}",
    }
    data = _weather_cached_json(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params,
        headers={
            "User-Agent": (
                "DeepSeekAlisa/1.0 "
                "(https://github.com/gennadiyjd-afk/deepseek-alisa/issues)"
            ),
            "Accept": "application/json",
        },
        timeout=20,
        ttl=10800,
    )
    series = (data.get("properties") or {}).get("timeseries") or []
    if not series:
        raise ValueError("MET Norway вернул пустой прогноз")

    tz_name = location.get("timezone") or "Europe/Moscow"
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = ZoneInfo("UTC")

    parsed = []
    for item in series:
        try:
            stamp = datetime.fromisoformat(
                item["time"].replace("Z", "+00:00")
            ).astimezone(local_tz)
        except Exception:
            continue
        details = ((item.get("data") or {}).get("instant") or {}).get("details") or {}
        next_data = (item.get("data") or {}).get("next_1_hours") or (item.get("data") or {}).get("next_6_hours") or {}
        summary = next_data.get("summary") or {}
        symbol = summary.get("symbol_code")
        precip = (next_data.get("details") or {}).get("precipitation_amount")
        parsed.append((stamp, details, _met_symbol_to_wmo(symbol), precip))
    if not parsed:
        raise ValueError("MET Norway не вернул пригодные временные точки")

    now_local = datetime.now(local_tz)
    current_entry = min(parsed, key=lambda row: abs((row[0] - now_local).total_seconds()))
    _, current_details, current_code, _ = current_entry
    current = {
        "temperature_2m": current_details.get("air_temperature"),
        "relative_humidity_2m": current_details.get("relative_humidity"),
        "weather_code": current_code,
        "wind_speed_10m": current_details.get("wind_speed"),
        "wind_direction_10m": current_details.get("wind_from_direction"),
    }

    grouped = {}
    for row in parsed:
        grouped.setdefault(row[0].date().isoformat(), []).append(row)
    dates = sorted(grouped)
    daily = {
        "time": [], "weather_code": [], "temperature_2m_max": [],
        "temperature_2m_min": [], "precipitation_probability_max": [],
        "precipitation_sum": [], "wind_speed_10m_max": [],
    }
    for date_text in dates:
        rows = grouped[date_text]
        temps = [r[1].get("air_temperature") for r in rows if r[1].get("air_temperature") is not None]
        if not temps:
            continue
        representative = min(
            rows,
            key=lambda r: abs((r[0].hour * 60 + r[0].minute) - 12 * 60),
        )
        winds = [r[1].get("wind_speed") for r in rows if r[1].get("wind_speed") is not None]
        precip_values = [r[3] for r in rows if r[3] is not None]
        daily["time"].append(date_text)
        daily["weather_code"].append(representative[2])
        daily["temperature_2m_max"].append(max(temps))
        daily["temperature_2m_min"].append(min(temps))
        daily["precipitation_probability_max"].append(None)
        daily["precipitation_sum"].append(sum(precip_values) if precip_values else None)
        daily["wind_speed_10m_max"].append(max(winds) if winds else None)

    if not daily["time"]:
        raise ValueError("MET Norway не вернул дневные значения температуры")
    return {
        "current": current,
        "daily": daily,
        "timezone": tz_name,
    }


def get_weather_answer(user_text):
    """Получает прогноз через Open-Meteo с резервным источником MET Norway."""
    try:
        location_query = extract_weather_location(user_text)
        location = resolve_weather_location(location_query)
        forecast_params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,rain,showers,snowfall,weather_code,"
                "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum,"
                "wind_speed_10m_max,sunrise,sunset"
            ),
            "forecast_days": 10,
            "timezone": location.get("timezone", "auto"),
            "wind_speed_unit": "ms",
        }
        weather_source = "Open-Meteo"
        try:
            data = _weather_cached_json(
                "https://api.open-meteo.com/v1/forecast",
                forecast_params,
                headers={"User-Agent": "DeepSeekAlisa/1.0"},
                timeout=15,
                ttl=1800,
                failure_ttl=600,
            )
        except Exception as primary_exc:
            logger.warning(
                "Open-Meteo failed; trying MET Norway fallback: %s",
                primary_exc,
            )
            data = _get_met_weather_data(location)
            weather_source = "MET Norway"
        current = data.get("current") or {}
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        if not current or not dates:
            raise ValueError("Open-Meteo вернул неполный прогноз")

        tz_name = data.get("timezone") or "Europe/Moscow"
        try:
            local_now = datetime.now(ZoneInfo(tz_name))
        except Exception:
            local_now = datetime.now(ZoneInfo("Europe/Moscow"))
        today = local_now.date()
        tomorrow = today.fromordinal(today.toordinal() + 1)
        query = normalize_command(user_text)
        is_tomorrow = "завтра" in query or "послезавтра" in query
        is_day_after = "послезавтра" in query
        is_weekend = "выходн" in query
        is_week = any(term in query for term in ("на неделю", "неделю", "на 7 дней", "семь дней"))

        place = location["name"]
        if location_query is None:
            place = "Бурдино"

        if is_week:
            day_lines = []
            for index, date_text in enumerate(dates[:7]):
                date_obj = datetime.strptime(date_text, "%Y-%m-%d").date()
                weekday = ("понедельник", "вторник", "среда", "четверг",
                           "пятница", "суббота", "воскресенье")[date_obj.weekday()]
                code = (daily.get("weather_code") or [])[index]
                low = (daily.get("temperature_2m_min") or [])[index]
                high = (daily.get("temperature_2m_max") or [])[index]
                day_lines.append(
                    f"{weekday.capitalize()}: {weather_description(code)}, "
                    f"от {round(low)} до {round(high)} градусов"
                )
            return f"Прогноз на неделю для {place}. " + ". ".join(day_lines) + f". Данные прогноза — {weather_source}."

        if is_weekend:
            days_until_saturday = (5 - today.weekday()) % 7
            if today.weekday() == 6:
                days_until_saturday = 6
            weekend_start = today + timedelta(days=days_until_saturday)
            weekend_indexes = []
            for index, date_text in enumerate(dates):
                date_obj = datetime.strptime(date_text, "%Y-%m-%d").date()
                if date_obj.weekday() >= 5 and date_obj >= weekend_start:
                    weekend_indexes.append(index)
                if len(weekend_indexes) == 2:
                    break
            if weekend_indexes:
                day_lines = []
                for index in weekend_indexes:
                    date_obj = datetime.strptime(dates[index], "%Y-%m-%d").date()
                    weekday = "субботу" if date_obj.weekday() == 5 else "воскресенье"
                    code = daily["weather_code"][index]
                    low = daily["temperature_2m_min"][index]
                    high = daily["temperature_2m_max"][index]
                    day_lines.append(
                        f"в {weekday} — {weather_description(code)}, "
                        f"от {round(low)} до {round(high)} градусов"
                    )
                return f"Прогноз на выходные. Населённый пункт: {place}. " + "; ".join(day_lines) + f". Данные прогноза — {weather_source}."

        target_date = today
        if is_tomorrow:
            target_date = tomorrow if not is_day_after else today.fromordinal(today.toordinal() + 2)
        target_index = next(
            (i for i, date_text in enumerate(dates) if date_text == target_date.isoformat()),
            None,
        )
        if target_index is None:
            raise LookupError("На запрошенную дату прогноза нет")

        code = daily["weather_code"][target_index]
        low = daily["temperature_2m_min"][target_index]
        high = daily["temperature_2m_max"][target_index]
        rain_chance = (daily.get("precipitation_probability_max") or [None] * len(dates))[target_index]
        rain_sum = (daily.get("precipitation_sum") or [None] * len(dates))[target_index]
        wind_max = (daily.get("wind_speed_10m_max") or [None] * len(dates))[target_index]
        day_label = "послезавтра" if is_day_after else ("завтра" if is_tomorrow else "сегодня")

        if not is_tomorrow:
            temp = current.get("temperature_2m")
            feels = current.get("apparent_temperature")
            humidity = current.get("relative_humidity_2m")
            current_code = current.get("weather_code", code)
            wind = current.get("wind_speed_10m")
            wind_dir = current.get("wind_direction_10m")
            if location_query is None:
                answer = (
                    f"Погода в {place} сейчас: {round(temp)} градусов, "
                    f"{weather_description(current_code)}"
                )
            else:
                answer = (
                    f"Погода сейчас. Населённый пункт: {place}. "
                    f"Температура {round(temp)} градусов, {weather_description(current_code)}"
                )
            if feels is not None:
                answer += f", ощущается как {round(feels)}"
            answer += (
                f". Сегодня ожидается от {round(low)} до {round(high)} градусов"
                f", {weather_description(code)}"
            )
            if rain_chance is not None:
                answer += f", вероятность осадков до {int(rain_chance)} процентов"
            if wind is not None:
                answer += f". Ветер {wind_direction_text(wind_dir or 0)}, скорость около {round(wind)} метров в секунду"
            if humidity is not None:
                answer += f". Влажность {int(humidity)} процентов"
            return answer + f". Данные прогноза — {weather_source}."
        if location_query is None:
            place_phrase = f"для {place}"
        else:
            place_phrase = f"для населённого пункта {place}"
        answer = (
            f"Прогноз на {day_label} {place_phrase}: {weather_description(code)}, "
            f"температура от {round(low)} до {round(high)} градусов"
        )
        if rain_chance is not None:
            answer += f", вероятность осадков до {int(rain_chance)} процентов"
        if rain_sum is not None and float(rain_sum) > 0:
            answer += f", ожидается около {float(rain_sum):.1f}".replace(".", ",") + " миллиметра осадков"
        if wind_max is not None:
            answer += f". Максимальная скорость ветра около {round(wind_max)} метров в секунду"
        return answer + f". Данные прогноза — {weather_source}."

    except Exception as exc:
        logger.warning("Weather lookup failed: %s", exc)
        if isinstance(exc, LookupError):
            return str(exc) + ". Уточни название населённого пункта или попробуй ещё раз."
        return "Не удалось получить прогноз погоды. Попробуй повторить запрос немного позже."


def is_complex_question(text):
    """Определяет запросы, которым обычно требуется длительная обработка."""
    normalized = normalize_command(text)
    complex_phrases = (
        "подробно проанализируй",
        "проведи анализ",
        "глубокий анализ",
        "сравни подробно",
        "составь подробный план",
        "разработай стратегию",
        "подготовь исследование",
        "напиши большую программу",
        "разбери по пунктам",
        "максимально подробно",
        "подробно расскажи",
        "расскажи подробно",
        "объясни подробно",
        "детально объясни",
        "развернутый ответ",
        "развернуто",
        "пошаговая инструкция",
        "подробная инструкция",
        "сравни и объясни",
        "подробн",
        "детальн",
        "развернут",
        "глубоко",
    )
    if any(phrase in normalized for phrase in complex_phrases):
        return True

    # Длинные многочастные запросы чаще требуют заметно больше времени.
    words = normalized.split()
    if len(words) >= 35 or len(normalized) >= 240:
        return True

    return False


def save_job_result(session_id, job_id, result):
    """Записывает результат и автоматически вызывает навык, если ожидание уже объявлено."""
    should_auto_deliver = False
    with jobs_lock:
        current = search_jobs.get(session_id)
        if current and current.get("job_id") == job_id:
            current.update(result)
            should_auto_deliver = (
                result.get("status") == "ready"
                and bool(current.get("auto_delivery"))
            )
            persist_job_maps()
        else:
            interrupted = interrupted_list(session_id)
            for saved_job in interrupted:
                if saved_job.get("job_id") == job_id:
                    saved_job.update(result)
                    persist_job_maps()
                    logger.info("Background result saved for an interrupted conversation")
                    return
            logger.info("Discarded result of an outdated task")
            return

    if should_auto_deliver:
        auto_deliver_ready_job(session_id, job_id)


def ask_ai(messages, timeout=25, max_tokens=1000):
    """Запрашивает ответ у RouterAPI с одной повторной попыткой при сбое."""
    if not API_KEY:
        raise RuntimeError("ROUTER_API_KEY is missing")

    last_error = None

    # Повторяем только временные сбои и ответы без текста.
    # История и результаты Tavily при этом не запрашиваются повторно.
    for attempt in range(2):
        current_max_tokens = max_tokens if attempt == 0 else max(max_tokens * 2, 4000)
        current_timeout = timeout if attempt == 0 else max(timeout, 75)

        if attempt:
            logger.warning(
                "Retrying RouterAPI request: max_tokens=%s, timeout=%s",
                current_max_tokens,
                current_timeout,
            )

        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek/deepseek-v4-pro",
                    "messages": messages,
                    "thinking": {"type": "disabled"},
                    "max_tokens": current_max_tokens,
                },
                timeout=current_timeout,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout as exc:
            last_error = exc
            logger.warning(
                "RouterAPI timeout on attempt %s/2 after %s seconds",
                attempt + 1,
                current_timeout,
            )
            if attempt == 0:
                continue
            raise

        choices = data.get("choices") if isinstance(data, dict) else None

        if not isinstance(choices, list) or not choices:
            logger.error(
                "AI response has no choices: status=%s, keys=%s",
                response.status_code,
                list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )
            last_error = RuntimeError("Empty AI response")
            if attempt == 0:
                continue
            raise last_error

        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            message = {}

        content = message.get("content", "")
        answer = ""

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_part = item.get("text")
                    if isinstance(text_part, str):
                        parts.append(text_part)
            answer = "\n".join(parts).strip()

        if answer:
            if attempt:
                logger.info("RouterAPI response recovered on retry")
            return answer

        logger.warning(
            "Empty AI content: attempt=%s/2, status=%s, finish_reason=%s, "
            "message_keys=%s, content_type=%s, content_length=%s, "
            "reasoning_present=%s, reasoning_length=%s, refusal_present=%s, "
            "tool_calls_present=%s, usage=%s",
            attempt + 1,
            response.status_code,
            choice.get("finish_reason"),
            list(message.keys()),
            type(content).__name__,
            len(content) if isinstance(content, (str, list)) else 0,
            bool(message.get("reasoning")),
            len(str(message.get("reasoning", ""))),
            bool(message.get("refusal")),
            bool(message.get("tool_calls")),
            data.get("usage"),
        )
        last_error = RuntimeError("Empty AI response")
        if attempt == 0:
            continue
        raise last_error

    raise last_error or RuntimeError("Empty AI response")

def build_chat_messages(user_text, history):
    # Передаём модели актуальную дату по московскому времени,
    # чтобы она не полагалась на устаревшие внутренние сведения.
    today_moscow = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y")
    detailed = is_complex_question(user_text)
    detail_instruction = (
        "Для этого запроса подготовь содержательный, подробный и законченный ответ "
        "примерно на 1200–1800 символов. Разбей изложение по смыслу на абзацы. "
        "Не пытайся самостоятельно делить его на голосовые части и не спрашивай "
        "о продолжении: это сделает программа."
        if detailed else
        "На обычный вопрос отвечай кратко, обычно в 1–3 предложениях."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — DeepSeek Алиса, голосовой ИИ-помощник "
                "для навыка Яндекс Алисы. "
                f"Сегодняшняя дата по московскому времени: {today_moscow}. "
                "Если пользователь спрашивает, какое сегодня число, "
                "какой сегодня день или какой сейчас год, отвечай по этой дате, "
                "не полагайся на собственные внутренние сведения о текущей дате. "
                "Если спрашивают имя, отвечай: "
                "«Я твой интеллектуальный помощник DeepSeek Алиса». "
                "Всегда говори о себе в женском роде: "
                "я рада, я поняла, я подготовила, я нашла. "
                "Твоего постоянного собеседника зовут Геннадий. "
                "Обращайся к нему на «ты» во всех ответах, никогда не переходи на «вы». "
                "Используй имя Геннадий естественно, но не слишком часто. "
                "Не называй себя ChatGPT или OpenAI. "
                "Отвечай только по-русски. "
                "Не приветствуй пользователя без повода "
                "и не представляйся повторно. "
                "Учитывай историю разговора и понимай "
                "короткие уточнения по контексту. "
                "Обычные вопросы — отвечай кратко, понятно и по существу. "
                + detail_instruction + " "
                "Если пользователь просит подробный ответ, раскрывай тему "
                "последовательно и полно. Не дели ответ на части самостоятельно "
                "и не спрашивай о продолжении: голосовая программа сделает это. "
                "Если говорят «Кратко», дай краткое изложение. "
                "Если просят подробнее раскрыть отдельный момент, "
                "сосредоточься именно на нём. "
                "Говори естественно, как в обычном разговоре. "
                "Если пользователь просит анекдот или шутку, обязательно расскажи её целиком, " "с понятной развязкой и смешным финалом. Не заканчивай на завязке. " "Не обрывай ответ на полуслове."
            ),
        }
    ]

    for item in history[-4:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            messages.append({
                "role": item["role"],
                "content": str(item.get("content", ""))[:400],
            })

    messages.append({"role": "user", "content": user_text})
    return messages


def complete_joke_if_needed(answer, messages):
    """Проверяет, закончен ли анекдот, и при необходимости дописывает финал."""
    answer = str(answer or "").strip()
    if not answer:
        return answer

    review_messages = list(messages)
    review_messages.append({"role": "assistant", "content": answer})
    review_messages.append({
        "role": "user",
        "content": (
            "Проверь предыдущий ответ. Если анекдот или шутка полностью "
            "рассказаны и имеют понятную смешную развязку, ответь ровно "
            "одним словом: COMPLETE. Если ответ оборван или в нём нет "
            "развязки, верни полный законченный вариант именно этого "
            "анекдота: сохрани завязку и допиши недостающую часть с финалом. "
            "Не объясняй юмор, не добавляй другой анекдот и не обращайся "
            "к собеседнику. Верни только готовый текст."
        ),
    })

    try:
        checked = ask_ai(review_messages, timeout=35, max_tokens=700).strip()
        if checked.upper() == "COMPLETE":
            logger.info("Joke completeness check passed")
            return answer
        if checked:
            logger.info("Incomplete joke repaired")
            return checked
    except Exception:
        logger.exception("Joke completeness check failed")

    return answer


def chat_in_background(session_id, job_id, user_text, history):
    try:
        logger.info("Background chat started")
        messages = build_chat_messages(user_text, history)
        ai_started = time.perf_counter()
        answer = ask_ai(
            messages,
            timeout=60,
            max_tokens=1800 if is_complex_question(user_text) else 1000,
        )
        if is_joke_request(user_text):
            answer = complete_joke_if_needed(answer, messages)

        logger.info(
            "Background chat AI completed in %.2f seconds",
            time.perf_counter() - ai_started,
        )

        result = {
            "status": "ready",
            "answer": answer,
        }
        save_job_result(session_id, job_id, result)
        logger.info("Background chat completed")
        return result

    except Exception:
        logger.exception("Background chat failed")
        result = {
            "status": "error",
            "answer": "",
        }
        save_job_result(session_id, job_id, result)
        return result

def complete_answer_if_cut_off(answer, messages):
    """Проверяет конец ответа и просит модель закончить оборванную мысль."""
    answer = str(answer or "").strip()
    if not answer:
        return answer

    # Закрывающие кавычки и скобки не должны мешать проверке знака конца.
    ending = answer.rstrip()
    if ending.endswith((".", "!", "?", "…", "»", "”", '"', "'", ")", "]")):
        return answer

    logger.warning("Search answer appears to be cut off; requesting completion")
    repair_messages = list(messages)
    repair_messages.append({"role": "assistant", "content": answer})
    repair_messages.append({
        "role": "user",
        "content": (
            "Предыдущий ответ оборвался до завершения последней мысли. "
            "Продолжи строго с того места, где он остановился. "
            "Не повторяй уже сказанное и не начинай новый ответ. "
            "Закончи оборванную фразу и мысль. Верни только продолжение, "
            "без вступления и без вопроса о продолжении."
        ),
    })

    try:
        continuation = ask_ai(repair_messages, timeout=35, max_tokens=700).strip()
        continuation = remove_source_links(continuation)
        if continuation:
            answer = answer.rstrip() + " " + continuation.lstrip()
            logger.info("Search answer completion added")
    except Exception:
        logger.exception("Could not complete cut-off search answer")

    # Не выдаём последнюю оборванную фразу как завершённую.
    final_check = answer.rstrip()
    if final_check and not final_check.endswith((".", "!", "?", "…", "»", "”", '"', "'", ")", "]")):
        answer = answer.rstrip(" ,;:-") + "."
    return answer


def search_and_answer(session_id, job_id, user_text, history):
    try:
        logger.info("Background search started")

        if is_weather_request(user_text):
            logger.info("Dedicated Open-Meteo weather lookup started")
            answer = get_weather_answer(user_text)
            save_job_result(session_id, job_id, {
                "status": "ready",
                "answer": answer,
            })
            logger.info("Dedicated Open-Meteo weather lookup completed")
            return {"status": "ready", "answer": answer}

        if is_currency_rate_request(user_text):
            logger.info("Dedicated CBR currency rate lookup started")
            answer = get_currency_rate_answer(user_text)
            save_job_result(session_id, job_id, {
                "status": "ready",
                "answer": answer,
            })
            logger.info("Dedicated CBR currency rate lookup completed")
            return {"status": "ready", "answer": answer}

        if is_stock_quote_request(user_text):
            logger.info("Dedicated stock quote lookup started")
            answer = get_stock_quote_answer(user_text, history)
            save_job_result(session_id, job_id, {
                "status": "ready",
                "answer": answer,
            })
            logger.info("Dedicated stock quote lookup completed")
            return {"status": "ready", "answer": answer}

        if not TAVILY_KEY:
            raise RuntimeError("TAVILY_API_KEY is missing")

        today = datetime.now().astimezone().strftime("%d.%m.%Y")
        lower_text = user_text.lower()

        oil_words = (
            "нефть", "brent", "брент", "wti",
            "даблъютиай", "дабл ю ти ай", "дабл-ю-ти-ай",
            "urals", "уралс", "юралс", "юралз",
            "марки бренд", "марки brent", "марки брент",
        )
        is_oil = any(word in lower_text for word in oil_words)

        finance_words = (
            "нефть", "brent", "брент", "wti",
            "даблъютиай", "дабл ю ти ай", "дабл-ю-ти-ай",
            "urals", "уралс", "юралс", "юралз",
            "золото", "курс", "доллар", "евро", "рубль",
            "акции", "котировк", "бирж", "биткоин",
            "криптовалют", "стоимость", "цена", "газ",
        )
        is_finance = any(word in lower_text for word in finance_words)

        if is_oil:
            search_topic = "finance"
            is_wti = any(name in lower_text for name in (
                "wti", "даблъютиай", "дабл ю ти ай", "дабл-ю-ти-ай",
            ))
            is_urals = any(name in lower_text for name in (
                "urals", "уралс", "юралс", "юралз",
            ))

            if is_urals:
                benchmark = "Urals"
                search_query = (
                    f"Urals crude oil latest price assessment today {today}. "
                    f"Find the latest available Urals export blend price in "
                    f"USD per barrel, the assessment/quotation date and time, "
                    f"geographical basis and source. Prefer reputable price "
                    f"reporting agencies and financial sources such as "
                    f"Argus, S&P Global Commodity Insights (Platts), Reuters "
                    f"and other established market data providers. "
                    f"Distinguish Urals price assessments from Brent or WTI "
                    f"futures and do not invent an exchange contract month. "
                    f"Russian query: {user_text}. "
                    f"Do not present an old article or historical quote as current."
                )
            else:
                benchmark = "WTI" if is_wti else "Brent"
                exchange_contract = (
                    "NYMEX WTI crude oil futures, front-month contract"
                    if is_wti else
                    "ICE Brent crude oil futures, front-month contract"
                )
                search_query = (
                    f"{benchmark} crude oil latest market quote today {today}. "
                    f"Find the actual quoted price for {exchange_contract}, "
                    f"contract delivery month, USD per barrel, exact quote "
                    f"date and time or last update timestamp. Prefer official "
                    f"exchange quote pages and reputable financial news such "
                    f"as ICE, CME, Reuters and major market data providers. "
                    f"Distinguish the futures contract from the general spot "
                    f"benchmark. Russian query: {user_text}. "
                    f"Do not present an old article or historical quote as current."
                )
        elif is_finance:
            search_topic = "finance"
            search_query = (
                f"{user_text}. Актуальная цена или последняя котировка "
                f"на {today}; укажи дату и время обновления данных."
            )
        else:
            search_topic = "news"
            search_query = (
                f"{user_text} самые свежие новости "
                f"и последние события на {today}"
            )

        search_payload = {
            "query": search_query,
            # Для финансовых данных сохраняем advanced-поиск;
            # для остальных запросов используем быстрый режим.
            "search_depth": "advanced" if is_finance else "fast",
            "max_results": 5,
            "topic": search_topic,
            "include_published_date": True,
        }
        if not is_finance:
            search_payload["days"] = 7

        tavily_started = time.perf_counter()
        response = requests.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {TAVILY_KEY}",
                "Content-Type": "application/json",
            },
            json=search_payload,
            timeout=20,
        )

        if not response.ok:
            logger.error(
                "Tavily error response: %s",
                response.text[:1000]
            )

        response.raise_for_status()
        logger.info(
            "Tavily search completed in %.2f seconds",
            time.perf_counter() - tavily_started,
        )
        data = response.json()
        results = data.get("results", [])

        if not results:
            raise RuntimeError("No search results")

        sources = []
        for item in results[:5]:
            title = str(item.get("title", "Без названия"))
            published_date = str(
                item.get("published_date") or "дата не указана"
            )
            url = str(item.get("url", ""))

            content = str(item.get("content", ""))
            sources.append(
                "Источник: " + title
                + "\nДомен: " + (url.split("/")[2] if "://" in url else "не указан")
                + "\nДата публикации материала: " + published_date
                + "\nURL: " + url
                + "\nФрагмент источника: " + content[:1400]
            )

        detailed_search = is_complex_question(user_text)
        search_length_instruction = (
            "Поскольку запрос требует подробного ответа, подготовь развернутое "
            "изложение примерно на 1600–2200 символов, логично разделённое на абзацы. "
            "Обязательно закончи последнюю мысль полным предложением. "
            "Не обрывай текст посреди слова или предложения. "
            "Не задавай вопрос о продолжении: программа сама разделит ответ на части. "
            if detailed_search else
            "Ответ должен быть законченным, кратким и понятным. Максимум 650 символов. "
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — DeepSeek Алиса, голосовой ИИ-помощник "
                    "для Яндекс Алисы. "
                    f"Сегодняшняя дата: {today}. "
                    "Это единственная текущая дата, на которую "
                    "нужно ориентироваться. "
                    "Отвечай только по-русски. "
                    "О себе говори только в женском роде. "
                    "Постоянного собеседника зовут Геннадий. "
                    "Обращайся к нему на «ты» во всех ответах, никогда не переходи на «вы». "
                    "Используй имя Геннадий естественно, но не слишком часто. "
                    "Не называй себя ChatGPT или OpenAI. "
                    "Используй найденные материалы для ответа. "
                    "Не выдумывай факты. "
                    "Если вопрос касается цены или котировки, обязательно "
                    "различай дату публикации статьи и дату/время самой "
                    "котировки: это не одно и то же. Дату и время котировки "
                    "называй только тогда, когда они прямо подтверждены "
                    "текстом найденного источника. "
                    "Не выдавай старую котировку за текущую. "
                    "Для нефти Brent, WTI и Urals указывай цену за баррель и валюту. "
                    "Если источник указывает месяц поставки фьючерса, "
                    "назови этот контрактный месяц. Не называй цену "
                    "конкретного фьючерсного контракта просто спотовой "
                    "ценой Brent или WTI. Если контрактный месяц не указан, "
                    "не придумывай его. "
                    "Для нефтяных котировок укажи название источника "
                    "(например, ICE, CME, Reuters или название площадки), "
                    "если оно видно в найденных материалах. "
                    "Ищи биржевую котировку или последнюю цену закрытия, "
                    "а не обзорные статьи. Если дата или время самой "
                    "котировки не подтверждены, прямо скажи об этом; "
                    "не подменяй их датой публикации статьи. "
                    "Если актуальная цена не найдена, прямо сообщи об этом. "
                    "Если биржа закрыта, укажи последнюю доступную цену закрытия. "
                    "Обязательно сверяй даты публикаций с сегодняшней датой. "
                    "Не называй событие сегодняшним, если оно произошло "
                    "раньше. Не выдавай старые новости за новые. "
                    "Если публикация старше нескольких дней, "
                    "уточняй её дату. "
                    "Не утверждай, что событие произошло, если источники "
                    "этого не подтверждают. "
                    "Отдавай предпочтение самым свежим материалам. "
                    "Если источники показывают разные цены, не выбирай "
                    "произвольно одну. Кратко объясни возможную причину: "
                    "разное время обновления, разные контрактные месяцы, "
                    "биржа или тип цены. Если причина не ясна, так и скажи "
                    "и приведи обе цифры с названиями источников. "
                    "Учитывай все найденные материалы, а не только первые. "
                    + search_length_instruction + " "
                    "Не произноси URL, адреса сайтов и ссылки. "
                    "Не предлагай пользователю источники и не спрашивай, "
                    "нужны ли ему ссылки. Давай только сам ответ. "
                    "Не обрывай предложения и не заканчивай ответ "
                    "незавершённой мыслью. "
                    "Говори естественно, как в обычном разговоре."
                ),
            },
        ]

        for item in history[-4:]:
            if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
                messages.append({
                    "role": item["role"],
                    "content": str(item.get("content", ""))[:400],
                })

        messages.append({
            "role": "system",
            "content": "Результаты интернет-поиска:\n\n" + "\n\n".join(sources),
        })

        messages.append({
            "role": "user",
            "content": user_text,
        })

        search_ai_started = time.perf_counter()
        answer = ask_ai(
            messages,
            timeout=60,
            # Обычный поисковый ответ тоже получает запас токенов:
            # RouterAPI может расходовать их на reasoning до текста ответа.
            max_tokens=5000 if detailed_search else 3500,
        )
        logger.info(
            "Search answer AI completed in %.2f seconds",
            time.perf_counter() - search_ai_started,
        )

        answer = remove_source_links(answer)
        if detailed_search:
            answer = complete_answer_if_cut_off(answer, messages)
        result = {
            "status": "ready",
            "answer": answer,
        }
        save_job_result(session_id, job_id, result)
        logger.info("Background search completed")
        return result

    except Exception:
        logger.exception("Background search failed")
        result = {"status": "error", "answer": ""}
        save_job_result(session_id, job_id, result)
        return result


def make_response(body, answer, history=None):
    response_data = {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": limit_text(answer),
        },
    }

    if history is not None:
        response_data["session_state"] = {"history": history[-4:]}
    else:
        response_data["session_state"] = {
            "history": body.get("state", {}).get("session", {}).get("history", [])[-4:]
        }

    return response_data


@app.on_event("startup")
async def startup_restore_redis_state():
    load_job_maps()


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "redis_configured": REDIS_AVAILABLE}


@app.post("/")
async def main(body: dict):
    request_data = body.get("request", {})
    user_text = str(request_data.get("original_utterance") or "").strip()

    # Пустой запуск навыка только приветствует и не затрагивает фоновые задачи.
    if not user_text:
        return make_response(
            body,
            "Привет! Я твой интеллектуальный помощник DeepSeek Алиса. Чем могу помочь?",
        )

    session = body.get("session", {})
    session_id = str(session.get("session_id", "default"))
    user_id = session.get("user", {}).get("user_id")
    job_key = str(user_id or session_id)

    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])
    wake_command = is_wake_command(user_text)
    continue_command = is_continue_command(user_text)
    repair_continue_command = is_continuation_repair_request(user_text)
    topic_return_command = is_topic_return_command(user_text)
    wait_command = is_wait_command(user_text)
    retrieval_command = (
        wake_command or wait_command or continue_command
        or repair_continue_command or topic_return_command
    )

    with jobs_lock:
        job = search_jobs.get(job_key)
        interrupted = interrupted_list(job_key)

    # Явный возврат выбирает нужную тему по словам исходного вопроса.
    # При переключении текущий ответ сохраняется, а выбранный становится активным.
    if topic_return_command:
        with jobs_lock:
            current = search_jobs.get(job_key)
            saved = interrupted_list(job_key)
            candidates = ([current] if current else []) + list(saved)
            selected = choose_interrupted_topic(user_text, candidates)

            if selected is None:
                return make_response(
                    body,
                    "Не нашла эту тему среди двух сохранённых разговоров. Назови тему ещё раз, пожалуйста.",
                )

            if current and selected.get("job_id") == current.get("job_id"):
                job = current
            else:
                if current:
                    search_jobs.pop(job_key, None)
                    parts = current.get("parts") or []
                    part_index = int(current.get("part_index", 0))
                    if current.get("status") == "pending" or (
                        current.get("status") == "ready" and len(parts) > part_index and len(parts) > 1
                    ):
                        save_interrupted_job(job_key, current)

                saved = interrupted_list(job_key)
                saved[:] = [item for item in saved if item.get("job_id") != selected.get("job_id")]
                search_jobs[job_key] = selected
                trim_interrupted_for_active(job_key)
                job = selected
                persist_job_maps()

        if job and job.get("status") == "pending":
            with jobs_lock:
                current = search_jobs.get(job_key)
                notice = pending_job_notice(current) if current else "Секундочку"
            return make_response(body, notice)
        if job and job.get("status") == "ready":
            return deliver_job_part(body, job_key, job)
        if job and job.get("status") == "error":
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(job_key, None)
                    persist_job_maps()
            return make_response(body, "Не удалось получить ответ. Попробуй повторить запрос.")

    # Обычное «Да» продолжает активную тему. Если активной нет, восстанавливаем
    # самую недавно прерванную из сохранённых.
    if not job and (continue_command or repair_continue_command):
        with jobs_lock:
            saved = interrupted_list(job_key)
            if saved:
                restored = saved.pop()
                search_jobs[job_key] = restored
                job = restored
                if saved:
                    interrupted_jobs[job_key] = saved
                else:
                    interrupted_jobs.pop(job_key, None)
                persist_job_maps()

        if job and job.get("status") == "pending":
            with jobs_lock:
                current = search_jobs.get(job_key)
                notice = pending_job_notice(current) if current else "Секундочку"
            return make_response(body, notice)
        if job and job.get("status") == "ready":
            return deliver_job_part(body, job_key, job)
        if job and job.get("status") == "error":
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(job_key, None)
                    persist_job_maps()
            return make_response(body, "Не удалось получить ответ. Попробуй повторить запрос.")

    # Отказ относится только к активному многочастному ответу.
    # Удаление задачи также не позволит позднему фоновому результату её восстановить.
    if (job or interrupted) and is_stop_command(user_text):
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and (not job or current.get("job_id") == job.get("job_id")):
                search_jobs.pop(job_key, None)
            interrupted_jobs.pop(job_key, None)
            persist_job_maps()
        return make_response(body, "Хорошо, больше не буду продолжать.")

    # Если задача ещё выполняется, команда ожидания «Алиса, я жду» / «Я жду», имя «Алиса» либо продолжения
    # сообщает только о задержке — без дополнительных инструкций.
    if job and job.get("status") == "pending":
        if retrieval_command:
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job.get("job_id"):
                    if current.get("status") == "pending":
                        notice = pending_job_notice(current)
                        job = dict(current)
                    else:
                        notice = None
                        job = dict(current)
                else:
                    notice = "Секундочку"

            if notice is not None:
                return make_response(body, notice)
            # Задача успела завершиться между проверками — ниже выдадим результат.
        else:
            # Короткая реплика может оказаться оговоркой. Сохраняем задачу,
            # чтобы последующее «Да» или пояснение вернуло пользователя к ответу.
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job.get("job_id"):
                    # Сохраняем даже при полноценном новом вопросе: это может быть
                    # переключение темы (например, с рассказа об ИИ на погоду).
                    save_interrupted_job(job_key, current)
                    search_jobs.pop(job_key, None)
                    persist_job_maps()
                    logger.info("Conversation task paused and saved before topic switch")
            job = None

    # Готовый ответ выдаём по команде «Алиса, я жду» / «Я жду» или по имени «Алиса», а следующие части —
    # по командам «Да», «Продолжи» или «Продолжай».
    if job and job.get("status") == "ready":
        if retrieval_command:
            return deliver_job_part(body, job_key, job)

        # Короткая посторонняя реплика может быть ошибкой ввода/распознавания.
        # Не уничтожаем из-за неё готовый многочастный ответ.
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                # Ответ уже готов, но ещё не выдан целиком: сохраняем его независимо
                # от длины новой реплики и даём обработать новую тему отдельно.
                parts = current.get("parts") or []
                part_index = int(current.get("part_index", 0))
                has_remaining_parts = len(parts) > part_index
                if has_remaining_parts and len(parts) > 1:
                    save_interrupted_job(job_key, current)
                    logger.info("Multi-part answer paused and saved before topic switch")
                search_jobs.pop(job_key, None)
                persist_job_maps()
        job = None

    elif job and job.get("status") == "error":
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
                persist_job_maps()
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй повторить запрос.",
        )

    # Если команды получения результата нет и фоновой задачи тоже нет,
    # не отправляем имя помощника модели как обычный вопрос.
    if retrieval_command:
        return make_response(body, "Я слушаю. Чем могу помочь?")

    # Благодарности и приветственные вопросы обрабатываем мгновенно,
    # не запуская фоновую задачу и не обращаясь к нейросети.
    social_reply = get_social_reply(user_text)
    if social_reply:
        return make_response(body, social_reply)

    # Интернет-поиск выполняется в фоне. Результат забирается командой «Алиса, я жду» / «Я жду».
    if needs_search(user_text):
        job_id = uuid.uuid4().hex
        with jobs_lock:
            search_jobs[job_key] = {
                "job_id": job_id,
                "kind": "search",
                "status": "pending",
                "pending_notice_sent": True,
                "auto_delivery": True,
                "auto_wait_started_at": time.time(),
                "auto_wait_last_sent_at": 0,
                "answer": "",
                "parts": [],
                "part_index": 0,
                "user_text": user_text,
                "history": history[-4:],
            }
            trim_interrupted_for_active(job_key)
            persist_job_maps()

        executor.submit(
            search_and_answer,
            job_key,
            job_id,
            user_text,
            history[-4:],
        )
        monitor_delayed_job(job_key, job_id)
        logger.info("Search task submitted; returning immediately to Alice")
        return make_response(body, "Сейчас посмотрю.")

    # Обычный разговор: простые вопросы ждут до 3,35 секунды,
    # сложные быстро переходят в фоновую обработку.
    complex_question = is_complex_question(user_text)
    wait_timeout = 0.35 if complex_question else 3.35
    logger.info(
        "Chat question classified as %s; wait timeout %.2f seconds",
        "complex" if complex_question else "simple",
        wait_timeout,
    )
    job_id = uuid.uuid4().hex
    with jobs_lock:
        search_jobs[job_key] = {
            "job_id": job_id,
            "kind": "chat",
            "status": "pending",
            "pending_notice_sent": False,
            "answer": "",
            "parts": [],
            "part_index": 0,
            "user_text": user_text,
            "history": history[-4:],
        }
        trim_interrupted_for_active(job_key)
        persist_job_maps()

    future = executor.submit(
        chat_in_background,
        job_key,
        job_id,
        user_text,
        history[-4:],
    )

    try:
        wrapped_future = asyncio.shield(asyncio.wrap_future(future))
        result = await asyncio.wait_for(wrapped_future, timeout=wait_timeout)

        if result.get("status") == "ready":
            answer = str(result.get("answer", "")).strip()
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job_id:
                    current["status"] = "ready"
                    prepare_job_parts(current, answer)
                    persist_job_maps()
                    current_job = dict(current)
                else:
                    current_job = None

            if current_job is None:
                return make_response(body, answer)

            parts = current_job.get("parts", [answer])
            if len(parts) > 1:
                return deliver_job_part(body, job_key, current_job)

            # Короткий ответ выдаём сразу и очищаем задачу.
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job_id:
                    search_jobs.pop(job_key, None)
                    persist_job_maps()
            new_history = history[-3:] + [
                {"role": "user", "content": user_text[:400]},
                {"role": "assistant", "content": answer[:400]},
            ]
            return make_response(body, answer, new_history)

        raise RuntimeError("Background chat returned no answer")

    except asyncio.TimeoutError:
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job_id:
                current["pending_notice_sent"] = True
                current["auto_delivery"] = True
                current["auto_wait_started_at"] = time.time()
                current["auto_wait_last_sent_at"] = 0
                result_already_ready = current.get("status") == "ready"
                persist_job_maps()
            else:
                result_already_ready = False

        monitor_delayed_job(job_key, job_id)
        if result_already_ready:
            auto_deliver_ready_job(job_key, job_id)
        logger.info(
            "Chat is still running after %.2f seconds; returning control to Alice",
            wait_timeout,
        )
        return make_response(body, "Секундочку")

    except Exception:
        logger.exception("Chat request failed")
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй, пожалуйста, ещё раз.",
        )

