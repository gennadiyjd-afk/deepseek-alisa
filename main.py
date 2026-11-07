import os
import time
import asyncio
import uuid
import logging
import threading
import re
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


def is_status_command(text):
    return normalize_command(text) in (
        "что там",
        "ну что там",
        "что там у нас",
        "ну и что там",
    )


def is_continue_command(text):
    return normalize_command(text) in (
        "продолжай",
        "продолжи",
        "продолжить",
        "готово",
    ) or is_status_command(text)


def needs_search(text):
    text = str(text or "").lower().replace("ё", "е")
    return (
        "найди в интернете" in text
        or "посмотри в интернете" in text
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
        ai_started = time.perf_counter()
        answer = ask_ai(messages, timeout=60)
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
        answer = ask_ai(messages, timeout=60)
        logger.info(
            "Search answer AI completed in %.2f seconds",
            time.perf_counter() - search_ai_started,
        )

        answer = remove_source_links(answer)
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

    session = body.get("session", {})
    session_id = str(session.get("session_id", "default"))
    # user_id persists across separate Alice sessions; session_id may change
    # between the initial request and the follow-up phrase "Что там?".
    user_id = session.get("user", {}).get("user_id")
    job_key = str(user_id or session_id)

    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])

    with jobs_lock:
        job = search_jobs.get(job_key)

    # Если предыдущая задача ещё выполняется, не запускаем вторую
    # поверх неё. Команда продолжения проверяет готовность результата.
    if job and job.get("status") == "pending":
        if is_continue_command(text_lower):
            if job.get("kind") == "search":
                # Не подсказываем пользователю команду повторно:
                # он сам спросит «Что там?», когда захочет получить результат.
                message = "Поиск ещё выполняется. Попробуй немного позже."
            else:
                message = "Ответ ещё готовится. Попробуй немного позже."
            return make_response(body, message)

        # Новый вопрос отменяет ожидание старого результата.
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
        job = None

    # Обрабатываем готовую задачу
    if job and job.get("status") == "ready":
        if job.get("kind") == "search":
            if is_continue_command(text_lower):
                answer = job.get("answer", "")
                with jobs_lock:
                    current = search_jobs.get(job_key)
                    if current and current.get("job_id") == job.get("job_id"):
                        search_jobs.pop(job_key, None)
                return make_response(body, answer)

            # Иное сообщение — начинаем обработку нового запроса.
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(job_key, None)

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
                    current = search_jobs.get(job_key)
                    if current and current.get("job_id") == job.get("job_id"):
                        search_jobs.pop(job_key, None)

                return make_response(body, answer, new_history)

            # Пользователь задал новый вопрос — удаляем старый результат.
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job.get("job_id"):
                    search_jobs.pop(job_key, None)

    elif job and job.get("status") == "error":
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)

        return make_response(
            body,
            "Не удалось получить ответ. Попробуй повторить запрос.",
        )

    # A status phrase is a retrieval command, not a new question for the AI.
    if is_status_command(user_text):
        return make_response(
            body,
            "Я пока не вижу сохранённого результата поиска. Повтори вопрос "
            "с фразой «найди в интернете» или «посмотри в интернете».",
        )

    # Запускаем интернет-поиск в фоновом потоке
    if needs_search(user_text):
        job_id = uuid.uuid4().hex

        with jobs_lock:
            search_jobs[job_key] = {
                "job_id": job_id,
                "kind": "search",
                "status": "pending",
                "answer": "",
                "user_text": user_text,
                "history": history[-4:],
            }

        future = executor.submit(
            search_and_answer,
            job_key,
            job_id,
            user_text,
            history[-4:],
        )

        # Не ждём завершения поиска в текущем вебхуке: Алиса сразу
        # отвечает пользователю, а Tavily и DeepSeek работают в фоне.
        # Готовый результат пользователь заберёт командой «Что там?».
        logger.info("Search task submitted; returning immediately to Alice")
        return make_response(body, "Сейчас посмотрю.")

    # Обычный разговор: даём модели короткое время на быстрый ответ.
    # Если ответ не готов за 2 секунды, продолжаем запрос в фоне,
    # чтобы не удерживать вебхук Алисы до тайм-аута.
    job_id = uuid.uuid4().hex
    with jobs_lock:
        search_jobs[job_key] = {
            "job_id": job_id,
            "kind": "chat",
            "status": "pending",
            "answer": "",
            "user_text": user_text,
            "history": history[-4:],
        }

    future = executor.submit(
        chat_in_background,
        job_key,
        job_id,
        user_text,
        history[-4:],
    )

    try:
        # shield не позволяет тайм-ауту ожидания отменить фоновый запрос.
        wrapped_future = asyncio.shield(asyncio.wrap_future(future))
        result = await asyncio.wait_for(wrapped_future, timeout=2.0)

        if result.get("status") == "ready":
            answer = result.get("answer", "")
            new_history = history[-3:] + [
                {"role": "user", "content": user_text[:400]},
                {"role": "assistant", "content": answer[:400]},
            ]
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job_id:
                    search_jobs.pop(job_key, None)
            return make_response(body, answer, new_history)

        raise RuntimeError("Background chat returned no answer")

    except asyncio.TimeoutError:
        logger.info("Chat is still running after 2 seconds; returning control to Alice")
        return make_response(
            body,
            "Ответ ещё готовится. Скажи «Что там?» немного позже.",
        )

    except Exception:
        logger.exception("Chat request failed")
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй, пожалуйста, ещё раз.",
        )
