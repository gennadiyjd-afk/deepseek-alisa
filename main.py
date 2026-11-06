import os
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

API_URL = "https://routerapi.ru/api/v1/chat/completions"
API_KEY = os.getenv("ROUTER_API_KEY")


@app.post("/")
async def main(body: dict):
    user_text = body["request"]["original_utterance"]

    # Получаем историю предыдущих реплик
    state = body.get("state", {}).get("session", {})
    history = state.get("history", [])

    # Оставляем только последние сообщения
    messages = [
        {
            "role": "system",
            "content": (
                "Ты — DeepSeek Alisa, голосовой ИИ-помощник "
                "для навыка Яндекс Алисы. "
                "Если спрашивают твоё имя, отвечай: "
                "«Я DeepSeek Alisa, твой интеллектуальный помощник». "
                "Не называй себя ChatGPT или OpenAI. "
                "Отвечай только по-русски. "
                "Не начинай ответ с приветствия, если пользователь "
                "сам не поздоровался. Не представляйся повторно. "
                "Учитывай предыдущие реплики и понимай короткие "
                "уточнения по контексту разговора. "
                "Отвечай понятно, дружелюбно и по существу. "
                "Для голосового общения предпочитай короткие ответы "
                "из 2–4 предложений."
            )
        }
    ]

    for item in history:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            messages.append({
                "role": item["role"],
                "content": str(item.get("content", ""))[:170]
            })

    messages.append({"role": "user", "content": user_text})

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
                "max_tokens": 220
            },
            timeout=3.5
        )

        data = response.json()

    except requests.exceptions.Timeout:
        return {
            "version": body["version"],
            "session": body["session"],
            "response": {
                "end_session": False,
                "text": "Ответ задерживается. Попробуй повторить вопрос."
            },
            "session_state": {"history": history[-4:]}
        }

    except (requests.exceptions.RequestException, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Не удалось получить ответ от API"
        )

    if response.status_code != 200 or "choices" not in data:
        raise HTTPException(
            status_code=502,
            detail="Ошибка ответа API"
        )

    answer = data["choices"][0]["message"]["content"]

    # Сохраняем последние две пары реплик
    new_history = history[-3:] + [
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
