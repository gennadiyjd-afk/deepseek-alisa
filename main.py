import os
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
    """Имя помощника, которым можно запросить готовый ответ."""
    return normalize_command(text) in ("алиса")


def is_wait_command(text):
    """Команда получения ответа после ожидания; учитываем перехват слова «Алиса» Яндексом."""
    command = normalize_command(text).replace(",", " ")
    command = " ".join(command.split())
    return command in ("алиса я жду", "я жду")


def is_continue_command(text):
    """Команды перехода к следующей части длинного ответа."""
    return normalize_command(text) in (
        "да",
        "продолжай",
        "продолжи",
        "продолжить",
    )


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


def needs_search(text):
    text = str(text or "").lower().replace("ё", "е")
    return (
        "найди в интернете" in text
        or "посмотри в интернете" in text
        or is_stock_quote_request(text)
        or is_currency_rate_request(text)
        or is_weather_request(text)
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


def fetch_cbr_currency_rate(currency_code):
    """Получает официальный курс ЦБ РФ на текущую дату (с учётом выходных)."""
    today_moscow = datetime.now(ZoneInfo("Europe/Moscow")).date()
    response = requests.get(
        "https://www.cbr.ru/scripts/XML_daily.asp",
        params={"date_req": today_moscow.strftime("%d/%m/%Y")},
        headers={"User-Agent": "Mozilla/5.0 DeepSeekAlice/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
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
        date_text = root.attrib.get("Date", "")
        if date_text:
            date_text = datetime.strptime(date_text, "%d.%m.%Y").strftime("%d.%m.%Y")
        return {
            "code": code,
            "name": (item.findtext("Name") or code).strip(),
            "value": value / nominal,
            "nominal": nominal,
            "date": date_text,
        }
    raise LookupError(f"В ответе Банка России нет курса {currency_code}")


def get_currency_rate_answer(user_text):
    code, currency_singular, currency_plural = extract_currency_code(user_text)
    if not code:
        return "Уточни, курс какой валюты тебя интересует."
    try:
        rate = fetch_cbr_currency_rate(code)
        value = f"{rate['value']:,.4f}".replace(",", " ").replace(".", ",")
        nominal = int(rate["nominal"])
        nominal_phrase = f"за 1 {currency_singular}" if nominal == 1 else f"за {nominal} {currency_plural}"
        date_phrase = f" на {rate['date']}" if rate.get("date") else ""
        return (
            f"Официальный курс Банка России: {value} рубля {nominal_phrase}. "
            f"Курс установлен{date_phrase}."
        )
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
            response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": query_variant,
                    "count": 10,
                    "language": language,
                    "format": "json",
                },
                headers={"User-Agent": "Mozilla/5.0 DeepSeekAlice/1.0"},
                timeout=12,
            )
            response.raise_for_status()
            for item in response.json().get("results") or []:
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
        "timezone": "auto",
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


def get_weather_answer(user_text):
    """Получает актуальные условия и прогноз через Open-Meteo."""
    try:
        location_query = extract_weather_location(user_text)
        location = resolve_weather_location(location_query)
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
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
            },
            headers={"User-Agent": "Mozilla/5.0 DeepSeekAlice/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
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
            return f"Прогноз на неделю для {place}. " + ". ".join(day_lines) + "."

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
                return f"Прогноз на выходные. Населённый пункт: {place}. " + "; ".join(day_lines) + ". Данные прогноза — Open-Meteo."

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
            return answer + ". Данные прогноза — Open-Meteo."
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
        return answer + ". Данные прогноза — Open-Meteo."

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
                "Ты — DeepSeek Алиса, голосовой ИИ-помощник "
                "для навыка Яндекс Алисы. "
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
    wait_command = is_wait_command(user_text)
    retrieval_command = wake_command or wait_command or continue_command

    with jobs_lock:
        job = search_jobs.get(job_key)

    # Отказ относится только к активному многочастному ответу.
    # Удаление задачи также не позволит позднему фоновому результату её восстановить.
    if job and is_stop_command(user_text):
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
        return make_response(body, "Хорошо, больше не буду продолжать.")

    # Если задача ещё выполняется, команда ожидания «Алиса, я жду» / «Я жду», имя «Алиса» либо продолжения
    # сообщает только о задержке — без дополнительных инструкций.
    if job and job.get("status") == "pending":
        if retrieval_command:
            return make_response(body, "Секундочку")

        # Новый вопрос отменяет старую задачу.
        with jobs_lock:
            current = search_jobs.get(job_key)
            if current and current.get("job_id") == job.get("job_id"):
                search_jobs.pop(job_key, None)
        job = None

    # Готовый ответ выдаём по команде «Алиса, я жду» / «Я жду» или по имени «Алиса», а следующие части —
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
            "Chat is still running after %.2f seconds; returning control to Marusya",
            wait_timeout,
        )
        return make_response(body, "Секундочку")

    except Exception:
        logger.exception("Chat request failed")
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй, пожалуйста, ещё раз.",
        )

