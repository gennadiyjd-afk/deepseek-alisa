import logging
import os
import re
import secrets
import threading
import time
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


def response_json(response: requests.Response, stage: str) -> dict:
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(f"{stage}: Яндекс вернул неожиданный ответ") from None
    if not isinstance(data, dict):
        raise RuntimeError(f"{stage}: неверный формат ответа Яндекса")
    if data.get("status") not in (None, "ok"):
        raise RuntimeError(f"{stage}: Яндекс сообщил об ошибке")
    return data


def yandex_login():
    passport = requests.Session()
    passport.headers.update({"User-Agent": USER_AGENT})

    try:
        page = passport.get("https://passport.yandex.ru/pwl-yandex", timeout=20)
        page.raise_for_status()
        match = re.search(r'__CSRF__\s*=\s*"([^"]+)', page.text)
        if not match:
            raise RuntimeError("Не удалось получить защитный токен страницы входа")

        csrf_login = match.group(1)
        login_headers = {"X-CSRF-Token": csrf_login}

        start = response_json(
            passport.post(
                "https://passport.yandex.ru/pwl-yandex/api/passport/auth/password/submit",
                json={"retpath": "https://passport.yandex.ru/"},
                headers=login_headers,
                timeout=20,
            ),
            "Создание QR-входа",
        )
        track_id = start.get("track_id")
        if not track_id:
            raise RuntimeError("Яндекс не создал сессию QR-входа")

        qr = response_json(
            passport.post(
                "https://passport.yandex.ru/pwl-yandex/api/passport/auth/magic/code",
                data={
                    "location_id": "0",
                    "magic_track_id": track_id,
                    "track_id": "",
                },
                headers=login_headers,
                timeout=20,
            ),
            "Получение QR-ссылки",
        )
        qr_link = qr.get("link")
        if not qr_link:
            raise RuntimeError("Яндекс не вернул ссылку для подтверждения")

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
                    json=start,
                    headers=login_headers,
                    timeout=20,
                ),
                "Проверка QR-входа",
            )
            if auth_status.get("state") == "otp_auth_finished":
                break
        else:
            raise RuntimeError("Время ожидания подтверждения истекло")

        session_track = auth_status.get("trackId")
        if not session_track:
            raise RuntimeError("Яндекс не подтвердил завершение входа")

        response_json(
            passport.post(
                "https://passport.yandex.ru/pwl-yandex/api/passport/sessions/get_session",
                data={"track_id": session_track},
                headers=login_headers,
                timeout=20,
            ),
            "Завершение входа",
        )

        cookie_header = "; ".join(
            f"{cookie.name}={cookie.value}"
            for cookie in passport.cookies
            if (cookie.domain or "").endswith("yandex.ru")
        )
        token_data = response_json(
            passport.post(
                "https://mobileproxy.passport.yandex.net/1/bundle/oauth/token_by_sessionid",
                data={
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                },
                headers={
                    "Ya-Client-Host": "passport.yandex.ru",
                    "Ya-Client-Cookie": cookie_header,
                },
                timeout=20,
            ),
            "Получение токена",
        )
        x_token = token_data.get("access_token")
        if not x_token:
            raise RuntimeError("Яндекс не выдал токен доступа")

        # Получаем сессию Quasar. Токен остаётся только в памяти процесса.
        quasar = requests.Session()
        quasar.headers.update({"User-Agent": USER_AGENT})
        auth = response_json(
            quasar.post(
                "https://mobileproxy.passport.yandex.net/1/bundle/auth/x_token/",
                data={"type": "x-token", "retpath": "https://www.yandex.ru"},
                headers={"Ya-Consumer-Authorization": f"OAuth {x_token}"},
                timeout=20,
            ),
            "Обновление сессии Яндекса",
        )
        passport_host = auth.get("passport_host")
        auth_track = auth.get("track_id")
        if not passport_host or not auth_track:
            raise RuntimeError("Яндекс не вернул параметры сессии Quasar")

        finish = quasar.get(
            f"{passport_host}/auth/session/",
            params={"track_id": auth_track},
            allow_redirects=False,
            timeout=20,
        )
        if finish.status_code not in (200, 302, 303):
            finish.raise_for_status()

        quasar_page = quasar.get("https://yandex.ru/quasar", timeout=20)
        quasar_page.raise_for_status()
        csrf_match = re.search(r'"csrfToken2"\s*:\s*"([^"]+)"', quasar_page.text)
        if not csrf_match:
            raise RuntimeError("Не удалось получить защитный токен Quasar")
        csrf = csrf_match.group(1)
        quasar.headers.update({"x-csrf-token": csrf})

        device_data = response_json(
            quasar.get(f"{QUASAR_API}/v3/user/devices", timeout=20),
            "Получение списка устройств",
        )
        devices = device_data.get("devices", [])
        if not isinstance(devices, list):
            raise RuntimeError("Яндекс вернул неизвестный формат списка устройств")

        speakers = []
        for device in devices:
            capabilities = device.get("capabilities") or []
            supports_phrase = any(
                item.get("type") == "devices.capabilities.quasar.server_action"
                for item in capabilities
                if isinstance(item, dict)
            )
            if supports_phrase and device.get("id"):
                speakers.append(
                    {
                        "id": str(device["id"]),
                        "name": str(device.get("name") or "Колонка Алисы"),
                    }
                )

        if not speakers:
            raise RuntimeError("Не найдена колонка с поддержкой прямой озвучки")

        with _guard:
            _state.update(
                status="ready",
                message="Яндекс подключён. Выбери колонку и отправь тестовую фразу.",
                link=None,
                session=quasar,
                csrf=csrf,
                devices=speakers,
                error=None,
            )

    except Exception as exc:
        # Не записываем токены, cookies и тела ответов в журнал.
        detail = str(exc).split(" for url:")[0].split(" with url:")[0][:180]
        log.warning("Авторизация не завершилась: %s: %s", type(exc).__name__, detail)
        with _guard:
            _state.update(
                status="error",
                message="Не удалось завершить вход. Проверь сообщение об ошибке ниже.",
                link=None,
                session=None,
                csrf=None,
                devices=[],
                error=str(exc)[:240],
            )


@app.get("/", response_class=HTMLResponse)
def index(_: bool = Depends(require_password)):
    return HTML_PAGE


@app.post("/api/start")
def start_login(_: bool = Depends(require_password)):
    with _guard:
        if _state["status"] in ("starting", "waiting"):
            return {
                "status": _state["status"],
                "message": _state["message"],
                "link": _state["link"],
            }
        _state.update(
            status="starting",
            message="Готовлю вход в Яндекс…",
            link=None,
            session=None,
            csrf=None,
            devices=[],
            error=None,
        )
    threading.Thread(target=yandex_login, daemon=True).start()
    return {"status": "starting", "message": "Готовлю вход в Яндекс…", "link": None}


@app.get("/api/status")
def get_status(_: bool = Depends(require_password)):
    with _guard:
        return {
            key: _state[key]
            for key in ("status", "message", "link", "devices", "error")
        }


@app.post("/api/say")
def say(payload: dict, _: bool = Depends(require_password)):
    device_id = str(payload.get("device_id") or "")
    phrase = "Проверка связи. Я могу говорить с вами напрямую."

    with _guard:
        session = _state.get("session")
        csrf = _state.get("csrf")
        allowed_ids = {device["id"] for device in _state.get("devices", [])}
        if _state["status"] != "ready" or session is None:
            raise HTTPException(409, "Сначала подключи Яндекс")
        if device_id not in allowed_ids:
            raise HTTPException(400, "Неизвестная колонка")

    try:
        result = session.post(
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
        response_json(result, "Отправка фразы")
        return {"status": "ok", "message": "Команда отправлена"}
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Отправка тестовой фразы не удалась: %s", type(exc).__name__)
        raise HTTPException(502, "Яндекс не принял тестовую фразу") from None


HTML_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Тест озвучки Алисы</title>
<style>
body{font:16px system-ui;max-width:650px;margin:28px auto;padding:0 16px;line-height:1.5}
button,select{font:inherit;padding:10px;margin:8px 0;max-width:100%}
#link{overflow-wrap:anywhere}.muted{color:#666}
</style>
</head>
<body>
<h2>Тест прямой озвучки Алисы</h2>
<p>Нажми кнопку. Затем открой ссылку Яндекса и подтверди вход в приложении Яндекса.</p>
<button id="start">Подключить Яндекс</button>
<p id="message">Авторизация ещё не запущена</p>
<p id="link"></p>
<div id="controls" hidden>
<label for="device">Выбери колонку:</label><br>
<select id="device"></select><br>
<button id="say">Произнести тестовую фразу</button>
</div>
<p class="muted">Токен не показывается на странице и хранится только в оперативной памяти тестового сервиса. После перезапуска сервиса вход потребуется повторить.</p>
<script>
const message=document.getElementById('message');
const link=document.getElementById('link');
const controls=document.getElementById('controls');
const device=document.getElementById('device');

async function callApi(url, body){
  const options={method:body?'POST':'GET'};
  if(body){options.headers={'Content-Type':'application/json'};options.body=JSON.stringify(body);}
  const response=await fetch(url,options);
  const data=await response.json();
  if(!response.ok) throw new Error(data.detail||'Ошибка запроса');
  return data;
}
async function refresh(){
  try{
    const data=await callApi('/api/status');
    message.textContent=data.message;
    link.replaceChildren();
    if(data.link){
      const a=document.createElement('a');
      a.href=data.link;a.target='_blank';a.rel='noopener';
      a.textContent='Открыть подтверждение входа Яндекса';
      link.appendChild(a);
    }
    if(data.status==='ready'){
      controls.hidden=false;
      device.replaceChildren();
      data.devices.forEach(item=>{
        const option=document.createElement('option');
        option.value=item.id;option.textContent=item.name;
        device.appendChild(option);
      });
    }
    if(data.status==='error'&&data.error) message.textContent+=' '+data.error;
    if(data.status==='starting'||data.status==='waiting') setTimeout(refresh,2500);
  }catch(error){message.textContent=error.message;}
}
document.getElementById('start').onclick=async()=>{
  try{await callApi('/api/start',{});refresh();}
  catch(error){message.textContent=error.message;}
};
document.getElementById('say').onclick=async()=>{
  try{
    const data=await callApi('/api/say',{device_id:device.value});
    message.textContent=data.message;
  }catch(error){message.textContent=error.message;}
};
</script>
</body>
</html>"""
