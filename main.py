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

    return any(phrase in text for phrase in phrases)


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

        response = requests.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {TAVILY_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": user_text[:350],
                "search_depth": "fast",
                "max_results": 7,
                "topic": "news",
                "days": 7,
                "include_published_date": True,
            },
            timeout=20,
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

            sources.append(
                "Название: " + title
                + "\nДата публикации: " + published_date
                + "\nСсылка: " + url
                + "\nСодержание: " + str(item.get("content", ""))[:1000]
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
                    "Обязательно сверяй даты публикаций с сегодняшней датой. "
                    "Не называй событие сегодняшним, если оно произошло "
                    "раньше. Не выдавай старые новости за новые. "
                    "Если публикация старше нескольких дней, "
                    "уточняй её дату. "
                    "Не утверждай, что событие произошло, если источники "
                    "этого не подтверждают. "
                    "Отдавай предпочтение самым свежим материалам. "
                    "Если источники противоречат друг другу, сообщи об этом. "
                    "Учитывай все найденные материалы, а не только первые. "
                    "Ответ должен быть законченным, кратким и понятным. "
                    "Максимум 650 символов. "
                    "Не перечисляй источники, даты и ссылки в основном ответе. "
                    "Не сообщай адреса сайтов. "
                    "Не задавай вопрос о ссылках на источники: "
                    "это сделает обработчик навыка отдельно. "
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
                "waiting_for_sources": True,
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
    text_lower = user_text.lower().strip()

    # Запуск навыка без произнесённого вопроса: только приветствуем.
    # Важно сделать это до проверки фоновых задач, чтобы пустой запуск
    # не отменял уже выполняющийся запрос.
    if not user_text:
        return make_response(
            body,
            "Привет! Я DeepSeek Alisa. Чем могу помочь?",
        )

    session_id = str(
        body.get("session", {}).get("session_id", "default")
    )

    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])

    with jobs_lock:
        job = search_jobs.get(session_id)

    # Если предыдущая задача ещё выполняется, не запускаем вторую
    # поверх неё. Команда продолжения проверяет готовность результата.
    if job and job.get("status") == "pending":
        if is_continue_command(text_lower):
            if job.get("kind") == "search":
                message = (
                    "Я ещё ищу информацию. "
                    "Попробуй сказать «Продолжи» немного позже."
                )
            else:
                message = (
                    "Я ещё готовлю ответ. "
                    "Попробуй сказать «Продолжи» немного позже."
                )
            return make_response(body, message)

        # Новый вопрос отменяет ожидание старого результата.
        with jobs_lock:
            current = search_jobs.get(session_id)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(session_id, None)
        job = None

    # Обрабатываем готовую задачу
    if job and job.get("status") == "ready":
        if job.get("kind") == "search":
            if job.get("waiting_for_sources") and text_lower in (
                "да", "нужны", "давай", "покажи", "конечно"
            ):
                source_list = job.get("sources", [])
                answer = "Вот источники с датами публикаций:\n"

                for i, source in enumerate(source_list, 1):
                    entry = (
                        f"\n{i}. {source['title']}"
                        f"\nДата: {source['date']}"
                        f"\nСсылка: {source['url']}\n"
                    )
                    if len(answer) + len(entry) > 1000:
                        answer += "\nОстальные ссылки не поместились в один ответ."
                        break
                    answer += entry

                with jobs_lock:
                    search_jobs.pop(session_id, None)

                return make_response(body, answer)

            if job.get("waiting_for_sources") and text_lower in (
                "нет", "не нужно", "не надо", "не нужны"
            ):
                with jobs_lock:
                    search_jobs.pop(session_id, None)
                return make_response(body, "Хорошо, продолжим без ссылок.")

            if is_continue_command(text_lower):
                answer = (
                    job["answer"].rstrip()
                    + "\n\nНужны ссылки на источники с датами публикаций?"
                )
                with jobs_lock:
                    current = search_jobs.get(session_id)
                    if current and current.get("job_id") == job.get("job_id"):
                        current["waiting_for_sources"] = True
                return make_response(body, answer)

            # Иное сообщение — начинаем обработку нового запроса.
            with jobs_lock:
                current = search_jobs.get(session_id)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(session_id, None)

        elif job.get("kind") == "chat":
            if is_continue_command(text_lower):
                answer = job.get("answer", "")
                original_text = job.get("user_text", "")
                original_history = job.get("history", [])
                new_history = original_history[-3:] + [
                    {"role": "user", "content": original_text[:400]},
                    {"role": "assistant", "content": answer[:400]},
                ]

                with jobs_lock:
                    current = search_jobs.get(session_id)
                    if current and current.get("job_id") == job.get("job_id"):
                        search_jobs.pop(session_id, None)

                return make_response(body, answer, new_history)

            # Пользователь задал новый вопрос — удаляем старый результат.
            with jobs_lock:
                current = search_jobs.get(session_id)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(session_id, None)

    elif job and job.get("status") == "error":
        with jobs_lock:
            current = search_jobs.get(session_id)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(session_id, None)

        return make_response(
            body,
            "Не удалось получить ответ. Попробуй повторить запрос.",
        )

    # Запускаем интернет-поиск в фоновом потоке
    if needs_search(user_text):
        job_id = uuid.uuid4().hex

        with jobs_lock:
            search_jobs[session_id] = {
                "job_id": job_id,
                "kind": "search",
                "status": "pending",
                "answer": "",
                "user_text": user_text,
                "history": history[-4:],
            }

        executor.submit(
            search_and_answer,
            session_id,
            job_id,
            user_text,
            history[-4:],
        )

        return make_response(
            body,
            "Хорошо, сейчас поищу информацию в интернете. "
            "Скажи «Продолжи» через несколько секунд.",
        )

    # Обычный запрос тоже обрабатывается в фоне
    job_id = uuid.uuid4().hex

    with jobs_lock:
        search_jobs[session_id] = {
            "job_id": job_id,
            "kind": "chat",
            "status": "pending",
            "answer": "",
            "user_text": user_text,
            "history": history[-4:],
        }

    executor.submit(
        chat_in_background,
        session_id,
        job_id,
        user_text,
        history[-4:],
    )

    return make_response(
        body,
        "Хорошо, готовлю ответ. Скажи «Продолжи» через несколько секунд.",
    )
