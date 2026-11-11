import logging
import os
import re
import secrets
import threading
import time
import traceback
from typing import Any

import requests
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI(title="Yandex Alice TTS test")
security = HTTPBasic()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alice_tts_test")

PANEL_PASSWORD = os.getenv("TTS_TEST_PASSWORD", "")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CLIENT_ID = "c0ebe342af7d48fbbbfcf2d2eedb8f9e"
CLIENT_SECRET = "ad0a908f0aa341a182a37ecd75bc319e"
QUASAR_API = "https://iot.quasar.yandex.ru/m"
PERSONAL_STATION_NAME = "Станция Геннадия"

_guard = threading.RLock()
_state: dict[str, Any] = {
    "status": "idle",
    "message": "Авторизация ещё не запущена",
    "link": None,
    "session": None,
    "csrf": None,
    "devices": [],
    "error": None,
}


def require_password(credentials: HTTPBasicCredentials = Depends(security)):
    if not PANEL_PASSWORD:
        raise HTTPException(503, "На Render не задана переменная TTS_TEST_PASSWORD")
    if not secrets.compare_digest(credentials.password, PANEL_PASSWORD):
        raise HTTPException(
            401, "Неверный пароль", headers={"WWW-Authenticate": "Basic"}
        )
    return True
def response_json(response: requests.Response, label: str):
    try:
        data = response.json()
    except Exception:
        raise RuntimeError(
            f"{label}: сервер вернул не JSON (HTTP {response.status_code})"
        )

    if response.status_code >= 400:
        message = data.get("message") or data.get("error") or response.text[:300]
        raise RuntimeError(f"{label}: HTTP {response.status_code}: {message}")

    return data


def yandex_login():
    passport = requests.Session()
    passport.headers.update({"User-Agent": USER_AGENT})

    page = passport.get(
        "https://passport.yandex.ru/pwl-yandex",
        timeout=20,
    )
    page.raise_for_status()

    csrf_match = re.search(
        r'name="csrf_token"\s+value="([^"]+)"',
        page.text,
    )
    if not csrf_match:
        csrf_match = re.search(
            r'"csrfToken"\s*:\s*"([^"]+)"',
            page.text,
        )
    if not csrf_match:
        raise RuntimeError("Не удалось получить CSRF-токен для QR-входа")

    passport_csrf = csrf_match.group(1)

    start = response_json(
        passport.post(
            "https://passport.yandex.ru/pwl-yandex/api/passport/auth/password/submit",
            data={"csrf_token": passport_csrf},
            timeout=20,
        ),
        "Запуск QR-авторизации",
    )

    magic = response_json(
        passport.post(
            "https://passport.yandex.ru/pwl-yandex/api/passport/auth/magic/code",
            data={"csrf_token": passport_csrf},
            timeout=20,
        ),
        "Создание QR-кода",
    )

    qr_link = (
        magic.get("link")
        or magic.get("qr_link")
        or magic.get("url")
    )
    if not qr_link:
        raise RuntimeError("Яндекс не вернул ссылку для QR-авторизации")

    with _guard:
        _state.update(
            status="waiting",
            message="Открой ссылку и подтверди вход в приложении Яндекса",
            link=qr_link,
            error=None,
        )

    deadline = time.time() + 180
    auth_status = {}

    while time.time() < deadline:
        time.sleep(3)
        auth_status = response_json(
            passport.post(
                "https://passport.yandex.ru/pwl-yandex/api/passport/auth/magic/code/status",
                data={
                    "csrf_token": passport_csrf,
                    "track_id": magic.get("track_id"),
                },
                timeout=20,
            ),
            "Проверка QR-авторизации",
        )

        if auth_status.get("status") == "otp_auth_finished":
            break
    else:
        raise RuntimeError("Время ожидания подтверждения входа истекло")
    session_data = response_json(
        passport.post(
            "https://passport.yandex.ru/pwl-yandex/api/passport/auth/sessions/get_session",
            data={
                "csrf_token": passport_csrf,
                "track_id": magic.get("track_id"),
            },
            timeout=20,
        ),
        "Получение сессии Яндекса",
    )

    cookies = "; ".join(
        f"{cookie.name}={cookie.value}"
        for cookie in passport.cookies
    )
    if not cookies:
        raise RuntimeError("Яндекс не выдал cookies после подтверждения входа")

    token_response = response_json(
        passport.post(
            "https://mobileproxy.passport.yandex.net/1/bundle/oauth/token_by_sessionid",
            data={"sessionid": cookies},
            timeout=20,
        ),
        "Получение OAuth-токена",
    )

    x_token = token_response.get("access_token") or token_response.get("x_token")
    if not x_token:
        raise RuntimeError("Не удалось получить токен доступа Яндекса")

    token_exchange = response_json(
        passport.post(
            "https://mobileproxy.passport.yandex.net/1/bundle/auth/x_token/",
            data={
                "type": "x-token",
                "retpath": "https://www.yandex.ru",
            },
            headers={
                "Ya-Consumer-Authorization": f"OAuth {x_token}",
            },
            timeout=20,
        ),
        "Обмен токена",
    )

    passport_host = token_exchange.get("passport_host")
    track_id = token_exchange.get("track_id")
    if not passport_host or not track_id:
        raise RuntimeError("Яндекс не вернул данные для завершения авторизации")

    quasar = requests.Session()
    quasar.headers.update({"User-Agent": USER_AGENT})

    # Переносим cookies авторизованной сессии в отдельную Quasar-сессию.
    for cookie in passport.cookies:
        quasar.cookies.set(
            cookie.name,
            cookie.value,
            domain=cookie.domain,
            path=cookie.path,
        )

    finish = quasar.get(
        f"{passport_host}/auth/session/",
        params={"track_id": track_id},
        allow_redirects=False,
        timeout=20,
    )
    log.info("Quasar finish response: HTTP %s", finish.status_code)
    # Получаем CSRF-токен со страницы Яндекса.
    iot_page = quasar.get(
        "https://yandex.ru/iot",
        timeout=20,
    )
    iot_page.raise_for_status()

    csrf_match = re.search(
        r'"csrfToken2"\s*:\s*"([^"]+)"',
        iot_page.text,
    )
    if not csrf_match:
        raise RuntimeError(
            "Не удалось найти csrfToken2 на странице Яндекса /iot"
        )

    csrf_token = csrf_match.group(1)
    quasar.headers.update({"x-csrf-token": csrf_token})

    devices_data = response_json(
        quasar.get(
            f"{QUASAR_API}/v3/user/devices",
            timeout=20,
        ),
        "Получение списка устройств",
    )

    all_devices = devices_data.get("devices", [])
    if not isinstance(all_devices, list):
        raise RuntimeError("Яндекс вернул список устройств в неожиданном формате")

    personal_speakers = []

    for device in all_devices:
        quasar_info = device.get("quasar_info") or {}

        if device.get("name") != PERSONAL_STATION_NAME:
            continue
        if not device.get("id"):
            continue
        if not device.get("capabilities"):
            continue
        if not quasar_info:
            continue

        personal_speakers.append(device)

    log.info(
        "Найдена личная станция: %s",
        [device.get("name") for device in personal_speakers],
    )

    if len(personal_speakers) != 1:
        raise RuntimeError(
            "Не удалось однозначно определить личную станцию "
            f"«{PERSONAL_STATION_NAME}». Найдено: {len(personal_speakers)}"
        )

    with _guard:
        _state.update(
            status="ready",
            message="Яндекс подключён. Можно отправить тестовую фразу на личную станцию.",
            link=None,
            session=quasar,
            csrf=csrf_token,
            devices=personal_speakers,
            error=None,
        )
def login_worker():
    try:
        yandex_login()
    except Exception as exc:
        log.exception("Ошибка авторизации Яндекса")
        with _guard:
            _state.update(
                status="error",
                message="Не удалось завершить подключение к Яндексу",
                link=None,
                session=None,
                csrf=None,
                devices=[],
                error=f"{type(exc).__name__}: {exc}",
            )


@app.get("/", response_class=HTMLResponse)
def index(_: bool = Depends(require_password)):
    return HTML_PAGE


@app.post("/api/start")
def start_login(_: bool = Depends(require_password)):
    with _guard:
        if _state["status"] in {"waiting", "working"}:
            return {"ok": True, "message": _state["message"]}

        _state.update(
            status="working",
            message="Подготавливаю подключение к Яндексу",
            link=None,
            session=None,
            csrf=None,
            devices=[],
            error=None,
        )

    threading.Thread(target=login_worker, daemon=True).start()
    return {"ok": True, "message": "Авторизация запущена"}


@app.get("/api/status")
def get_status(_: bool = Depends(require_password)):
    with _guard:
        return {
            "status": _state["status"],
            "message": _state["message"],
            "link": _state["link"],
            "devices": [
                {"id": device["id"], "name": device.get("name", "Станция")}
                for device in _state["devices"]
            ],
            "error": _state["error"],
        }


@app.post("/api/say")
def say(_: bool = Depends(require_password)):
    phrase = "Проверка связи. Я могу говорить с вами напрямую."

    with _guard:
        session = _state.get("session")
        csrf = _state.get("csrf")
        devices = _state.get("devices") or []

        if _state["status"] != "ready" or session is None or not csrf:
            raise HTTPException(409, "Сначала подключи Яндекс")
        if len(devices) != 1:
            raise HTTPException(409, "Личная станция не определена")

        device_id = str(devices[0]["id"])

    try:
        response = session.post(
            f"{QUASAR_API}/user/devices/{device_id}/actions",
            json={
                "actions": [
                    {
                        "type": "devices.capabilities.quasar.server_action",
                        "state": {
                            "instance": "phrase_action",
                            "value": phrase,
                        },
                    }
                ]
            },
            headers={"x-csrf-token": csrf},
            timeout=20,
        )

        response_json(response, "Отправка тестовой фразы")
        return {"ok": True, "message": "Команда отправлена"}
    except Exception as exc:
        log.exception("Ошибка отправки тестовой фразы")
        raise HTTPException(
            502,
            f"Не удалось отправить фразу: {type(exc).__name__}",
        )


HTML_PAGE = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка Яндекс Станции</title>
<style>
body {
    font-family: Arial, sans-serif;
    max-width: 560px;
    margin: 30px auto;
    padding: 0 16px;
    line-height: 1.5;
}
button {
    width: 100%;
    padding: 14px;
    margin: 8px 0;
    font-size: 16px;
}
#status {
    margin: 18px 0;
    white-space: pre-wrap;
}
a { overflow-wrap: anywhere; }
</style>
</head>
<body>
<h2>Подключение Яндекс Станции</h2>
<p>Будет выбрана только личная станция.</p>

<button id="start">Подключить Яндекс</button>
<div id="status">Ожидание запуска</div>
<div id="link"></div>
<button id="say" hidden>Произнести тестовую фразу</button>

<script>
const statusBox = document.getElementById("status");
const linkBox = document.getElementById("link");
const sayButton = document.getElementById("say");

async function api(path, method = "GET") {
    const response = await fetch(path, {method});
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail || "Ошибка запроса");
    }
    return data;
}

async function refresh() {
    try {
        const data = await api("/api/status");
        statusBox.textContent = data.message || data.status;
        linkBox.innerHTML = "";

        if (data.link) {
            const link = document.createElement("a");
            link.href = data.link;
            link.textContent = "Открыть подтверждение входа";
            link.target = "_blank";
            link.rel = "noopener";
            linkBox.appendChild(link);
        }

        if (data.error) {
            statusBox.textContent += "\n\n" + data.error;
        }

        sayButton.hidden = data.status !== "ready";
        if (data.status === "waiting" || data.status === "working") {
            setTimeout(refresh, 2500);
        }
    } catch (error) {
        statusBox.textContent = error.message;
    }
}

document.getElementById("start").addEventListener("click", async () => {
    statusBox.textContent = "Запускаю авторизацию...";
    sayButton.hidden = true;
    linkBox.innerHTML = "";

    try {
        await api("/api/start", "POST");
        refresh();
    } catch (error) {
        statusBox.textContent = error.message;
    }
});

sayButton.addEventListener("click", async () => {
    sayButton.disabled = true;
    statusBox.textContent = "Отправляю фразу на личную станцию...";

    try {
        const data = await api("/api/say", "POST");
        statusBox.textContent = data.message;
    } catch (error) {
        statusBox.textContent = error.message;
    } finally {
        sayButton.disabled = false;
    }
});

refresh();
</script>
</body>
</html>
"""
