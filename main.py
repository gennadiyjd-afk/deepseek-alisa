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
from datetime import datetime, timezone
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


def get_social_reply(text):
    """Мгновенные ответы на благодарность и вопрос о самочувствии."""
    normalized = normalize_command(text)
    normalized = " ".join(re.sub(r"[^a-zа-я0-9 ]+", " ", normalized).split())

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
    if normalized in thanks_phrases:
        return "Пожалуйста! Рада была помочь!"

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
    if normalized in how_are_you_phrases:
        return "У меня всё отлично! Спасибо, что интересуешься!"

    return None


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


def needs_search(text):
    text = str(text or "").lower().replace("ё", "е")
    return (
        "найди в интернете" in text
        or "посмотри в интернете" in text
        or is_stock_quote_request(text)
        or is_currency_rate_request(text)
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
    headers = {"User-Agent": "Mozilla/5.0 DeepSeekAlisa/1.0"}
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
        headers={"User-Agent": "Mozilla/5.0 DeepSeekAlisa/1.0"},
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
                "Ты — DeepSeek Alisa, голосовой ИИ-помощник "
                "для навыка Яндекс Алисы. "
                "Если спрашивают имя, отвечай: "
                "«Я твой интеллектуальный помощник DeepSeek Alisa». "
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
                    "Ты — DeepSeek Alisa, голосовой ИИ-помощник "
                    "для Яндекс Алисы. "
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
            "Привет! Я твой интеллектуальный помощник DeepSeek Alisa. Чем могу помочь?",
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
            return make_response(body, "Секундочку")

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

    # Благодарности и приветственные вопросы обрабатываем мгновенно,
    # не запуская фоновую задачу и не обращаясь к нейросети.
    social_reply = get_social_reply(user_text)
    if social_reply:
        return make_response(body, social_reply)

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
        return make_response(body, "Секундочку")

    except Exception:
        logger.exception("Chat request failed")
        return make_response(
            body,
            "Не удалось получить ответ. Попробуй, пожалуйста, ещё раз.",
        )

