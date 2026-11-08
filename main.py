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


def is_wake_command(text):
    """Команда, которой пользователь забирает готовый ответ у Алисы."""
    return normalize_command(text) == "алиса"


def is_continue_command(text):
    """Команды перехода к следующей части длинного ответа."""
    return normalize_command(text) in (
        "да",
        "продолжай",
        "продолжи",
        "продолжить",
    )


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
            return make_response(body, "На этом всё.")

        answer = parts[index]
        is_last = index == len(parts) - 1
        if not is_last:
            answer += " Продолжить?"
            current["part_index"] = index + 1
        else:
            original_text = current.get("user_text", "")
            original_history = current.get("history", [])
            complete_answer = current.get("answer", "")
            new_history = original_history[-3:] + [
                {"role": "user", "content": original_text[:400]},
                {"role": "assistant", "content": complete_answer[:400]},
            ]
            search_jobs.pop(job_key, None)
            return make_response(body, answer, new_history)

    return make_response(body, answer)

def needs_search(text):
    text = str(text or "").lower().replace("ё", "е")
    return (
        "найди в интернете" in text
        or "посмотри в интернете" in text
    )


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
    """Записывает результат, только если это всё ещё актуальная задача."""
    with jobs_lock:
        current = search_jobs.get(session_id)

        if not current or current.get("job_id") != job_id:
            logger.info("Discarded result of an outdated task")
            return

        current.update(result)

def ask_ai(messages, timeout=25, max_tokens=1000):
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
            "max_tokens": max_tokens,
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
                "Обычные вопросы — отвечай кратко, понятно и по существу. "
                + detail_instruction + " "
                "Если пользователь просит подробный ответ, раскрывай тему "
                "последовательно и полно. Не дели ответ на части самостоятельно "
                "и не спрашивай о продолжении: голосовая программа сделает это. "
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
        answer = ask_ai(
            messages,
            timeout=60,
            max_tokens=1800 if is_complex_question(user_text) else 1000,
        )
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
            max_tokens=2600 if detailed_search else 1000,
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


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.post("/")
async def main(body: dict):
    request_data = body.get("request", {})
    user_text = str(request_data.get("original_utterance") or "").strip()

    # Пустой запуск навыка только приветствует и не затрагивает фоновые задачи.
    if not user_text:
        return make_response(
            body,
            "Привет! Я DeepSeek Alisa. Чем могу помочь?",
        )

    session = body.get("session", {})
    session_id = str(session.get("session_id", "default"))
    user_id = session.get("user", {}).get("user_id")
    job_key = str(user_id or session_id)

    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])
    wake_command = is_wake_command(user_text)
    continue_command = is_continue_command(user_text)
    retrieval_command = wake_command or continue_command

    with jobs_lock:
        job = search_jobs.get(job_key)

    # Если задача ещё выполняется, команда «Алиса» или продолжения
    # сообщает только о задержке — без дополнительных инструкций.
    if job and job.get("status") == "pending":
        if retrieval_command:
            return make_response(body, "Ответ задерживается")

        # Новый вопрос отменяет старую задачу.
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
        job = None

    # Готовый ответ выдаём по слову «Алиса», а следующие части —
    # по командам «Да», «Продолжи» или «Продолжай».
    if job and job.get("status") == "ready":
        if retrieval_command:
            return deliver_job_part(body, job_key, job)

        # Иное сообщение считается новым запросом.
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
        job = None

    elif job and job.get("status") == "error":
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй повторить запрос.",
        )

    # Если команды получения результата нет и фоновой задачи тоже нет,
    # не отправляем слово «Алиса» модели как обычный вопрос.
    if retrieval_command:
        return make_response(body, "Я слушаю. Чем могу помочь?")

    # Интернет-поиск выполняется в фоне. Результат забирается командой «Алиса».
    if needs_search(user_text):
        job_id = uuid.uuid4().hex
        with jobs_lock:
            search_jobs[job_key] = {
                "job_id": job_id,
                "kind": "search",
                "status": "pending",
                "answer": "",
                "parts": [],
                "part_index": 0,
                "user_text": user_text,
                "history": history[-4:],
            }

        executor.submit(
            search_and_answer,
            job_key,
            job_id,
            user_text,
            history[-4:],
        )
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
            "answer": "",
            "parts": [],
            "part_index": 0,
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
        wrapped_future = asyncio.shield(asyncio.wrap_future(future))
        result = await asyncio.wait_for(wrapped_future, timeout=wait_timeout)

        if result.get("status") == "ready":
            answer = str(result.get("answer", "")).strip()
            with jobs_lock:
                current = search_jobs.get(job_key)
                if current and current.get("job_id") == job_id:
                    current["status"] = "ready"
                    prepare_job_parts(current, answer)
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
            new_history = history[-3:] + [
                {"role": "user", "content": user_text[:400]},
                {"role": "assistant", "content": answer[:400]},
            ]
            return make_response(body, answer, new_history)

        raise RuntimeError("Background chat returned no answer")

    except asyncio.TimeoutError:
        logger.info(
            "Chat is still running after %.2f seconds; returning control to Alice",
            wait_timeout,
        )
        return make_response(body, "Ответ задерживается")

    except Exception:
        logger.exception("Chat request failed")
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй, пожалуйста, ещё раз.",
        )

