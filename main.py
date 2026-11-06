
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI, HTTPException

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

    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]

    return shortened.rstrip(" ,;:-") + "..."
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
        "какая сейчас цена"
    ]

    return any(phrase in text for phrase in phrases)


def search_and_answer(session_id, user_text, history):
    try:
        logger.info("Background search started")

        if not TAVILY_KEY:
            raise Exception("TAVILY_API_KEY is missing")

        response = requests.post(
            TAVILY_URL,
            headers={
                "Authorization": f"Bearer {TAVILY_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "query": user_text[:350],
                "search_depth": "fast",
                "max_results": 3,
                "topic": "news",
                "days": 7,
                "include_published_date": True
            },
            timeout=20
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])

        if not results:
            raise Exception("No search results")

        sources = []
        source_details = []

        for item in results[:3]:
            title = str(item.get("title", "Без названия"))
            published_date = str(
            item.get("published_date") or "дата не указана"
            )
            url = str(item.get("url", ""))

            sources.append(
                "Название: " + title +
                "\nДата публикации: " + published_date +
                "\nСсылка: " + url +
                "\nСодержание: " + str(item.get("content", ""))[:1000]
            )

            source_details.append({
                "title": title,
                "date": published_date,
                "url": url
            })

        web_context = "\n\n".join(sources)

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — DeepSeek Alisa, голосовая ИИ-помощница "
                    "женского рода для Яндекс Алисы. "
                    "Отвечай только по-русски. "
                    "О себе говори только в женском роде. "
                    "Постоянного собеседника зовут Геннадий. "
                    "Обращайся к нему естественно, но не слишком часто. "
                    "Не называй себя ChatGPT или OpenAI. "
                    "Используй найденные материалы для ответа. "
                    "Не выдумывай факты. "
                    "Учитывай даты публикаций и отдавай предпочтение "
                    "самым свежим материалам. "
                    "Если источники противоречат друг другу, "
                    "сообщи об этом. "
                    "Ответ должен быть законченным, кратким и понятным. "
                    "Максимум 750 символов. "
                    "Не перечисляй источники, даты и ссылки в основном ответе. "
                    "Не сообщай адреса сайтов. "
                    "В конце основного ответа обязательно спроси: "
                   "«Нужны ссылки на источники с датами публикаций?» "
                   "Ответ должен быть законченным и не обрываться. "
                   "Говори естественно, как в обычном разговоре."
                )
            }
        ]

        for item in history[-4:]:
            if isinstance(item, dict) and item.get("role") in (
                "user", "assistant"
            ):
                messages.append({
                    "role": item["role"],
                    "content": str(item.get("content", ""))[:400]
                })

        messages.append({
            "role": "system",
            "content": (
                "Результаты интернет-поиска:\n\n"
                + web_context
            )
        })

        messages.append({
            "role": "user",
            "content": user_text
        })

        ai_response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-v4-pro",
                "messages": messages,
                "thinking": {
                    "type": "disabled"
                },
                "max_tokens": 350
            },
            timeout=20
        )

        ai_response.raise_for_status()
        ai_data = ai_response.json()

        answer = (
            ai_data["choices"][0]
            .get("message", {})
            .get("content", "")
        )

        if not answer:
            raise Exception("Empty AI response")

        with jobs_lock:
            search_jobs[session_id] = {
                "status": "ready",
                "answer": answer,
                "sources": source_details,
                "waiting_for_sources": True
            }

        logger.info("Background search completed")

    except Exception:
        logger.exception("Background search failed")

        with jobs_lock:
            search_jobs[session_id] = {
                "status": "error",
                "answer": ""
            }


def ask_ai(messages):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-v4-pro",
            "messages": messages,
            "thinking": {
                "type": "disabled"
            },
            "max_tokens": 350
        },
        timeout=3.5
    )

    response.raise_for_status()
    data = response.json()

    return (
        data["choices"][0]
        .get("message", {})
        .get("content", "")
    )


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.post("/")
async def main(body: dict):
    user_text = body["request"]["original_utterance"]
    text_lower = user_text.lower().strip()

    session_id = str(
        body.get("session", {}).get("session_id", "default")
    )

    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])

    # Проверяем состояние поиска и ожидание ответа о ссылках

    with jobs_lock:
        job = search_jobs.get(session_id)

    if job and job.get("status") == "ready":

        # Пользователь согласился получить источники

        if job.get("waiting_for_sources") and text_lower in (
            "да", "нужны", "давай", "покажи", "конечно"
        ):
            source_list = job.get("sources", [])

            answer = "Вот источники с датами публикаций:\n"

            for i, source in enumerate(source_list, 1):
                answer += (
                    f"\n{i}. {source['title']}"
                    f"\nДата: {source['date']}"
                    f"\nСсылка: {source['url']}\n"
                )

            with jobs_lock:
                search_jobs.pop(session_id, None)

        # Пользователь отказался от ссылок

        elif job.get("waiting_for_sources") and text_lower in (
            "нет", "не нужно", "не надо", "не нужны"
        ):
            with jobs_lock:
                search_jobs.pop(session_id, None)

            answer = "Хорошо, продолжим без ссылок."

        # Получаем готовый ответ после команды Продолжай

        elif text_lower == "продолжай":
            answer = (
                job["answer"].rstrip()
                + "\n\nНужны ссылки на источники с датами публикаций?"
            )

            with jobs_lock:
                job["waiting_for_sources"] = True

        else:
            answer = None

        if answer is not None:
            return {
                "version": body["version"],
                "session": body["session"],
                "response": {
                    "end_session": False,
                    "text": limit_text(answer)
                },
                "session_state": {
                    "history": history[-4:]
                }
            }

    elif job and job.get("status") == "error":
        with jobs_lock:
            search_jobs.pop(session_id, None)

        answer = (
            "Не удалось получить результаты поиска. "
            "Попробуй повторить запрос."
        )

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": limit_text(answer)
            },
            "session_state": {
                "history": history[-4:]
            }
        }

    elif job and job.get("status") == "pending":

        if text_lower in ("да", "продолжай", "продолжи", "продолжить", "готово"):
            answer = (
                "Я ещё ищу информацию. "
                "Попробуй сказать «Продолжай» немного позже."
            )

            return {
                "version": body["version"],
                "session": body["session"],
                "response": {
                    "end_session": False,
                    "text": limit_text(answer)
                },
                "session_state": {
                    "history": history[-4:]
                }
            }

    # Запускаем интернет-поиск в фоновом потоке
    if needs_search(user_text):
        with jobs_lock:
            existing_job = search_jobs.get(session_id)

            if existing_job and existing_job["status"] == "pending":
                answer = (
                    "Предыдущий поиск ещё выполняется. "
                    "Скажи «Продолжай» немного позже."
                )
            else:
                search_jobs[session_id] = {
                    "status": "pending",
                    "answer": ""
                }

                executor.submit(
                    search_and_answer,
                    session_id,
                    user_text,
                    history[-4:]
                )

                answer = (
                    "Хорошо, сейчас поищу информацию в интернете. "
                    "Скажи «Продолжай» через несколько секунд."
                )

        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": limit_text(answer)
            },
            "session_state": {
                "history": history[-4:]
            }
        }

    # Обычный режим работы Алисы

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
                "Обращайся к нему по имени естественно, "
                "но не слишком часто. "

                "Не называй себя ChatGPT или OpenAI. "
                "Отвечай только по-русски. "
                "Не приветствуй пользователя без повода "
                "и не представляйся повторно. "

                "Учитывай историю разговора и понимай "
                "короткие уточнения по контексту. "

                "Обычные вопросы — отвечай кратко, "
                "понятно и по существу, обычно в 1–3 предложениях. "

                "Если просят рассказать подробно, "
                "раздели рассказ на последовательные части. "
                "Каждая часть должна быть законченной, "
                "содержательной и посвящённой одной теме. "

                "Размер одной части — примерно 250–350 символов. "
                "Не пытайся рассказать всё сразу. "
                "В конце предложи продолжить. "

                "Если пользователь говорит «Продолжай», "
                "переходи к следующей части без повторов. "
                "Сохраняй последовательность рассказа. "

                "Если говорят «Кратко», дай краткое изложение. "
                "Если просят подробнее раскрыть отдельный момент, "
                "сосредоточься именно на нём. "

                "Говори естественно, как в обычном разговоре. "
                "Не обрывай ответ на полуслове."
            )
        }
    ]

    for item in history:
        if isinstance(item, dict) and item.get("role") in (
            "user", "assistant"
        ):
            messages.append({
                "role": item["role"],
                "content": str(item.get("content", ""))[:400]
            })

    messages.append({
        "role": "user",
        "content": user_text
    })

    start_time = time.perf_counter()

    try:
        answer = ask_ai(messages)

        elapsed = round(time.perf_counter() - start_time, 2)

        logger.info(
            "API response: time=%s sec, chars=%s",
            elapsed,
            len(answer)
        )

        if not answer:
            answer = "Не удалось сформировать ответ. Попробуй ещё раз."

    except requests.exceptions.Timeout:
        logger.warning("API timeout")
        answer = "Ответ задерживается. Попробуй повторить вопрос."

    except requests.exceptions.RequestException:
        logger.exception("API connection error")
        answer = "Сейчас не удаётся связаться с помощником. Попробуй позже."

    except Exception:
        logger.exception("Unexpected error")
        answer = "Произошла ошибка. Попробуй ещё раз."

    new_history = history[-3:] + [
        {"role": "user", "content": user_text[:400]},
        {"role": "assistant", "content": answer[:400]}
    ]

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": limit_text(answer)
        },
        "session_state": {
            "history": new_history[-4:]
        }
    }
