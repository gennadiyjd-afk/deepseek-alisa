import os
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

API_URL = "https://routerapi.ru/api/v1/chat/completions"
API_KEY = os.getenv("ROUTER_API_KEY")


@app.post("/")
async def main(body: dict):
    user_text = body["request"]["original_utterance"]

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": user_text}
            ]
        },
        timeout=60
    )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="API вернул некорректный ответ")

    if response.status_code != 200 or "choices" not in data:
        raise HTTPException(
            status_code=502,
            detail={
                "api_status": response.status_code,
                "api_error": data
            }
        )

    answer = data["choices"][0]["message"]["content"]

    return {
        "version": body["version"],
        "session": body["session"],
        "response": {
            "end_session": False,
            "text": answer
        }
    }
