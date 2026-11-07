import os
import time
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_alisa")

API_URL = "https://routerapi.ru/api/v1/chat/completions"
API_KEY = os.getenv("ROUTER_API_KEY")

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

executor = ThreadPoolExecutor(max_workers=2)
search_jobs = {}
jobs_lock = threading.Lock()



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

def is_continue_command(text):
    return text.lower().strip() in (
        "продолжай",
        "продолжи",
        "продолжить",
        "готово",
    )


def is_result_check_command(text):
    text = text.lower().strip().replace("ё", "е")
    phrases = (
        "ну что там", "что там", "что нашла", "что ты нашла",
        "есть результат", "готов ответ", "получилось", "как там поиск",
        "ну как", "результат", "ты нашла", "нашла",
    )
    return text in phrases or any(text.startswith(p + " ") for p in phrases)


def needs_search(text):
    text = text.lower()

    phrases = [
        "найди в интернете",
        "поищи в интернете",
        "найди информацию",
        "поищи информацию",
        "поищи в сети",
        "последние новости",
        "свежие новости",
        "новости сегодня",
        "что сейчас происходит",
        "что происходит сейчас",
        "на сегодняшний день",
        "актуальная информация",
        "актуальные данные",
        "последние события",
        "последние изменения",
        "сколько сейчас стоит",
        "какая сейчас цена",
    ]

    if any(phrase in text for phrase in phrases):
        return True

    # Автоматически ищем свежие рыночные данные, даже если пользователь
    # не произнёс фразу «найди в интернете».
    market_terms = (
        "brent", "брент", "wti", "нефть", "курс", "котировк",
        "акции", "золото", "биткоин", "криптовалют", "цена", "стоимость",
    )
    current_terms = (
        "сейчас", "сегодня", "текущ", "актуальн", "последн", "на данный момент",
    )
    return (
        any(term in text for term in market_terms)
        and any(term in text for term in current_terms)
    )


def save_job_result(session_id, job_id, result):
    """Записывает результат, только если это всё ещё актуальная задача."""
    with jobs_lock:
        current = search_jobs.get(session_id)

        if not current or current.get("job_id") != job_id:
            logger.info("Discarded result of an outdated task")
            return

        current.update(result)

def ask_ai(messages, timeout=25):
    if not API_KEY:
        raise RuntimeError("ROUTER_API_KEY is missing")

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
            "max_tokens": 1000,
        },
        timeout=timeout,
    )

    response.raise_for_status()
    data = response.json()

    choices = data.get("choices") if isinstance(data, dict) else None

    if not isinstance(choices, list) or not choices:
        logger.error(
            "AI response has no choices: status=%s, keys=%s",
            response.status_code,
            list(data.keys()) if isinstance(data, dict) else type(data).__name__,
        )
        raise RuntimeError("Empty AI response")

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

    if not answer:
        logger.error(
            "Empty AI content: status=%s, finish_reason=%s, "
            "message_keys=%s, content_type=%s, "
            "content_length=%s, reasoning_present=%s, "
            "reasoning_length=%s, refusal_present=%s, "
            "tool_calls_present=%s, usage=%s",
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

        raise RuntimeError("Empty AI response")

    return answer

def build_chat_messages(user_text, history):
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — DeepSeek Alisa, голосовая ИИ-помощница "
                "женского рода для навыка Яндекс Алисы. "
                "Если спрашивают имя, отвечай: "
                "«Я DeepSeek Alisa, твоя интеллектуальная помощница». "
                "Всегда говори о себе в женском роде: "
                "я рада, я поняла, я подготовила, я нашла. "
                "Твоего постоянного собеседника зовут Геннадий. "
                "Обращайся к нему по имени естественно, но не слишком часто. "
                "Не называй себя ChatGPT или OpenAI. "
                "Отвечай только по-русски. "
                "Не приветствуй пользователя без повода "
                "и не представляйся повторно. "
                "Учитывай историю разговора и понимай "
                "короткие уточнения по контексту. "
                "Обычные вопросы — отвечай кратко, понятно и по существу, "
                "обычно в 1–3 предложениях. "
                "Если просят рассказать подробно, раздели рассказ "
                "на последовательные законченные части по одной теме. "
                "Размер одной части — примерно 250–350 символов. "
                "В конце предложи продолжить. "
                "Если пользователь говорит «Продолжай» или «Продолжи», "
                "переходи к следующей части без повторов. "
                "Сохраняй последовательность рассказа. "
                "Если говорят «Кратко», дай краткое изложение. "
                "Если просят подробнее раскрыть отдельный момент, "
                "сосредоточься именно на нём. "
                "Говори естественно, как в обычном разговоре. "
                "Не обрывай ответ на полуслове."
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


def chat_in_background(session_id, job_id, user_text, history):
    try:
        logger.info("Background chat started")
        messages = build_chat_messages(user_text, history)
        answer = ask_ai(messages, timeout=60)

        save_job_result(
            session_id,
            job_id,
            {
                "status": "ready",
                "answer": answer,
            },
        )
        logger.info("Background chat completed")

    except Exception:
        logger.exception("Background chat failed")
        save_job_result(
            session_id,
            job_id,
            {
                "status": "error",
                "answer": "",
            },
        )

def search_and_answer(session_id, job_id, user_text, history):
    try:
        logger.info("Background search started")

        if not TAVILY_KEY:
            raise RuntimeError("TAVILY_API_KEY is missing")

        today = datetime.now().astimezone().strftime("%d.%m.%Y")
        lower_text = user_text.lower()

        oil_words = (
            "нефть", "brent", "брент", "wti", "марки бренд",
            "марки brent", "марки брент",
        )
        is_oil = any(word in lower_text for word in oil_words)

        finance_words = (
            "нефть", "brent", "брент", "wti", "золото", "курс",
            "доллар", "евро", "рубль", "акции", "котировк", "бирж",
            "биткоин", "криптовалют", "стоимость", "цена", "газ",
        )
        is_finance = any(word in lower_text for word in finance_words)

        if is_oil:
            search_topic = "finance"
            is_wti = "wti" in lower_text
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
            # Tavily supports the finance topic only with advanced search.
            "search_depth": "advanced" if is_finance else "fast",
            "max_results": 7,
            "topic": search_topic,
            "include_published_date": True,
        }
        if not is_finance:
            search_payload["days"] = 7

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
        data = response.json()
        results = data.get("results", [])

        if not results:
            raise RuntimeError("No search results")

        sources = []
        source_details = []

        for item in results[:7]:
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

            source_details.append({
                "title": title,
                "date": published_date,
                "url": url,
            })

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — DeepSeek Alisa, голосовая ИИ-помощница "
                    "женского рода для Яндекс Алисы. "
                    f"Сегодняшняя дата: {today}. "
                    "Это единственная текущая дата, на которую "
                    "нужно ориентироваться. "
                    "Отвечай только по-русски. "
                    "О себе говори только в женском роде. "
                    "Постоянного собеседника зовут Геннадий. "
                    "Обращайся к нему естественно, но не слишком часто. "
                    "Не называй себя ChatGPT или OpenAI. "
                    "Используй найденные материалы для ответа. "
                    "Не выдумывай факты. "
                    "Если вопрос касается цены или котировки, обязательно "
                    "различай дату публикации статьи и дату/время самой "
                    "котировки: это не одно и то же. Дату и время котировки "
                    "называй только тогда, когда они прямо подтверждены "
                    "текстом найденного источника. "
                    "Не выдавай старую котировку за текущую. "
                    "Для нефти Brent и WTI указывай цену за баррель и валюту. "
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
                    "Ответ должен быть законченным, кратким и понятным. "
                    "Максимум 650 символов. "
                    "В основном ответе можно назвать один-два источника "
                    "словами, но не произноси URL и длинные списки ссылок. "
                    "Не сообщай адреса сайтов. "
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

        answer = ask_ai(messages, timeout=60)

        save_job_result(
            session_id,
            job_id,
            {
                "status": "ready",
                "answer": answer,
                "sources": source_details,
            },
        )

        logger.info("Background search completed")

    except Exception:
        logger.exception("Background search failed")
        save_job_result(
            session_id,
            job_id,
            {
                "status": "error",
                "answer": "",
            },
        )


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


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.post("/")
async def main(body: dict):
    request_data = body.get("request", {})
    user_text = str(request_data.get("original_utterance") or "").strip()
    text_lower = user_text.lower().strip().replace("ё", "е")

    if not user_text:
        return make_response(body, "Привет! Я DeepSeek Alisa. Чем могу помочь?")

    session_id = str(body.get("session", {}).get("session_id", "default"))
    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])
    check_result = is_continue_command(text_lower) or is_result_check_command(text_lower)

    with jobs_lock:
        job = search_jobs.get(session_id)

    if job and job.get("status") == "pending":
        if check_result:
            message = (
                "Я ещё проверяю информацию. Спроси меня «Ну что там?» немного позже."
                if job.get("kind") == "search" else
                "Я ещё готовлю ответ. Спроси меня «Ну что там?» немного позже."
            )
            return make_response(body, message)
        with jobs_lock:
            current = search_jobs.get(session_id)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(session_id, None)
        job = None

    if job and job.get("status") == "ready":
        if check_result:
            answer = str(job.get("answer", "")).strip()
            original_text = job.get("user_text", "")
            original_history = job.get("history", history)
            new_history = original_history[-3:] + [
                {"role": "user", "content": str(original_text)[:400]},
                {"role": "assistant", "content": answer[:400]},
            ]
            with jobs_lock:
                current = search_jobs.get(session_id)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(session_id, None)
            return make_response(body, answer or "Не удалось получить готовый ответ.", new_history)
        with jobs_lock:
            current = search_jobs.get(session_id)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(session_id, None)
        job = None

    if job and job.get("status") == "error":
        with jobs_lock:
            current = search_jobs.get(session_id)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(session_id, None)
        return make_response(body, "Не удалось получить ответ. Попробуй повторить запрос.")

    if needs_search(user_text):
        job_id = uuid.uuid4().hex
        with jobs_lock:
            search_jobs[session_id] = {
                "job_id": job_id, "kind": "search", "status": "pending",
                "answer": "", "user_text": user_text, "history": history[-4:],
            }
        executor.submit(search_and_answer, session_id, job_id, user_text, history[-4:])
        return make_response(
            body,
            "Сейчас проверю информацию. Когда захочешь узнать результат, спроси: «Ну что там?»",
        )

    # Обычные вопросы пробуем обработать сразу, чтобы Алиса не произносила
    # фразу ожидания для каждого запроса. При долгом ответе используем фон.
    try:
        messages = build_chat_messages(user_text, history)
        answer = ask_ai(messages, timeout=3.2)
        new_history = history[-3:] + [
            {"role": "user", "content": user_text[:400]},
            {"role": "assistant", "content": answer[:400]},
        ]
        return make_response(body, answer, new_history)
    except Exception:
        logger.exception("Synchronous chat failed; switching to background")

    job_id = uuid.uuid4().hex
    with jobs_lock:
        search_jobs[session_id] = {
            "job_id": job_id, "kind": "chat", "status": "pending",
            "answer": "", "user_text": user_text, "history": history[-4:],
        }
    executor.submit(chat_in_background, session_id, job_id, user_text, history[-4:])
    return make_response(
        body,
        "Мне нужно немного больше времени. Спроси меня «Ну что там?» чуть позже.",
    )

