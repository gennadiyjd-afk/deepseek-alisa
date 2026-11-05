import os
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


@app.post("/")
async def main(body: dict):
    user_text = body["request"]["original_utterance"]

    response = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": user_text}],
        },
        timeout=60
    )

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek вернул не JSON. HTTP {response.status_code}"
        )

    if response.status_code != 200 or "choices" not in data:
        raise HTTPException(
            status_code=502,
            detail={
                "deepseek_status": response.status_code,
                "deepseek_error": data
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
