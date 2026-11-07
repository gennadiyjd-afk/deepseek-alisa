
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


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


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
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
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
        data = response.json()

        if response.status_code != 200 or "choices" not in data:
            logger.error(
                "API error: status=%s, time=%s",
                response.status_code,
                elapsed
            )
            raise HTTPException(
                status_code=502,
                detail="Ошибка ответа API"
            )

        choice = data["choices"][0]
        answer = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason")

        logger.info(
            "API response: time=%s sec, finish_reason=%s, chars=%s",
            elapsed,
            finish_reason,
            len(answer)
        )

        if not answer:
            answer = "Не удалось сформировать ответ. Попробуй ещё раз."

    except requests.exceptions.Timeout:
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.warning("API timeout after %s sec", elapsed)
        answer = "Ответ задерживается. Попробуй повторить вопрос."

    except requests.exceptions.RequestException:
        logger.exception("Connection error")
        answer = "Сейчас не удаётся связаться с помощником. Попробуй позже."

    except ValueError:
        logger.exception("Invalid API response")
        answer = "Получен некорректный ответ. Попробуй ещё раз."

    new_history = history[-3:] + [
        {"role": "user", "content": user_text[:400]},
        {"role": "assistant", "content": answer[:400]}
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
