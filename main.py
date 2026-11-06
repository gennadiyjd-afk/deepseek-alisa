
import os
import time
import logging
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek_alisa")

API_URL = "https://routerapi.ru/api/v1/chat/completions"
API_KEY = os.getenv("ROUTER_API_KEY")


@app.post("/")
async def main(body: dict):
    user_text = body["request"]["original_utterance"]

    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])


messages = [
    {
        "role": "system",
        "content": (
            "Ты — DeepSeek Alisa, голосовая ИИ-помощница "
            "женского рода для навыка Яндекс Алисы. "

            "Если спрашивают имя, отвечай: "
            "«Я DeepSeek Alisa, твоя интеллектуальная помощница». "

            "Ты всегда говоришь о себе исключительно в женском роде. "
            "Используй формы: я рада, я поняла, я подготовила, "
            "я нашла, я уверена, я готова. "
            "Никогда не употребляй мужской род по отношению к себе. "
            "Соблюдай это правило во всех ответах без исключений. "

            "Твоего постоянного собеседника зовут Геннадий. "
            "Обращайся к нему по имени естественно и дружелюбно, "
            "но не начинай каждое сообщение с его имени "
            "и не повторяй имя слишком часто. "

            "Не называй себя ChatGPT или OpenAI. "
            "Отвечай только по-русски. "
            "Не приветствуй повторно и не представляйся без необходимости. "
            "Учитывай предыдущие реплики и короткие уточнения. "
            "Отвечай понятно, дружелюбно и по существу. "
            "Для голосового общения используй 1–3 законченных предложения. "
            "Не обрывай ответ на полуслове."
        )
    }
]

for item in history[-4:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            messages.append({
                "role": item["role"],
                "content": str(item.get("content", ""))[:170]
            })

    messages.append({"role": "user", "content": user_text})

    start_time = time.perf_counter()

    try:
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

        elapsed = round(time.perf_counter() - start_time, 2)
        response.raise_for_status()
        data = response.json()

        if not data.get("choices"):
            logger.error("API returned no choices")
            raise ValueError("No choices")

        choice = data["choices"][0]
        message = choice.get("message", {})
        answer = message.get("content") or ""
        finish_reason = choice.get("finish_reason")

        usage = data.get("usage", {})
        logger.info(
            "API response: time=%s sec, finish_reason=%s, "
            "chars=%s, tokens=%s",
            elapsed,
            finish_reason,
            len(answer),
            usage.get("completion_tokens", "unknown")
        )

        if not answer.strip():
            answer = "Не удалось сформировать ответ. Попробуй ещё раз."

        elif finish_reason == "length":
            answer += " Продолжить?"

    except requests.exceptions.Timeout:
        logger.warning("API timeout")
        answer = "Ответ задерживается. Попробуй повторить вопрос."

    except requests.exceptions.RequestException:
        logger.exception("Connection error")
        answer = "Сейчас не удаётся связаться с помощником. Попробуй позже."

    except (ValueError, KeyError, IndexError):
        logger.exception("Invalid API response")
        answer = "Получен некорректный ответ. Попробуй ещё раз."

    new_history = history[-2:] + [
        {"role": "user", "content": user_text[:170]},
        {"role": "assistant", "content": answer[:170]}
    ]

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": answer
        },
        "session_state": {
            "history": new_history[-4:]
        }
    }
