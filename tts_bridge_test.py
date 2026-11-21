import base64
import hashlib
import json
import logging
import os
import re
import secrets
import socket
import ssl
import struct
import threading
import time
import traceback
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Body, Depends, FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI(title="Yandex Alice TTS test")
security = HTTPBasic()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("alice_tts_test")

PANEL_PASSWORD = os.getenv("TTS_TEST_PASSWORD", "")
BRIDGE_SECRET = os.getenv("TTS_BRIDGE_SECRET", "")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CLIENT_ID = "c0ebe342af7d48fbbbfcf2d2eedb8f9e"
CLIENT_SECRET = "ad0a908f0aa341a182a37ecd75bc319e"
QUASAR_API = "https://iot.quasar.yandex.ru/m"
PERSONAL_STATION_NAME = "Станция Геннадия"

SESSION_REDIS_KEY = "deepseek-alisa:tts-bridge:production:yandex-session:v1"
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60

_guard = threading.RLock()
# После выхода блокируем новые автоматические text_action до следующего запуска навыка.
_bridge_commands_enabled = False
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


def get_session_cipher() -> Fernet:
    """Build the Fernet cipher from the private Render environment variable."""
    encryption_key = os.getenv("TTS_SESSION_ENCRYPTION_KEY", "").strip()
    if not encryption_key:
        raise RuntimeError("Ключ шифрования не настроен")
    try:
        return Fernet(encryption_key.encode("ascii"))
    except (ValueError, UnicodeError):
        raise RuntimeError("Ключ шифрования имеет неверный формат")


def redis_session_command(command: list[Any]) -> Any:
    """Execute one Upstash REST command without exposing credentials or payloads."""
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip().rstrip("/")
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if not redis_url or not redis_token:
        raise RuntimeError("Upstash Redis не настроен")

    response = requests.post(
        redis_url,
        json=command,
        headers={"Authorization": f"Bearer {redis_token}"},
        timeout=(5, 10),
    )
    if response.status_code != 200:
        raise RuntimeError(f"Upstash Redis вернул HTTP {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Upstash Redis вернул некорректный ответ")

    if not isinstance(data, dict) or "error" in data or "result" not in data:
        raise RuntimeError("Команда Upstash Redis завершилась ошибкой")
    return data["result"]


def save_yandex_session(session: requests.Session) -> None:
    """Encrypt and persist only the Yandex cookies needed to rebuild the session."""
    cookies = []
    for cookie in session.cookies:
        cookies.append({
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
            "secure": bool(cookie.secure),
            "expires": cookie.expires,
            "rest": dict(cookie._rest or {}),
            "discard": bool(cookie.discard),
            "version": int(cookie.version or 0),
        })

    if not cookies:
        raise RuntimeError("В сессии Яндекса нет cookies для сохранения")

    payload = json.dumps(
        {"version": 1, "cookies": cookies},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encrypted_payload = get_session_cipher().encrypt(payload).decode("ascii")

    result = redis_session_command([
        "SET",
        SESSION_REDIS_KEY,
        encrypted_payload,
        "EX",
        SESSION_TTL_SECONDS,
    ])
    if result != "OK":
        raise RuntimeError("Upstash Redis не подтвердил сохранение сессии")


def load_yandex_session_cookies() -> list[dict[str, Any]] | None:
    """Read and decrypt the saved cookie bundle; never log its contents."""
    encrypted_payload = redis_session_command(["GET", SESSION_REDIS_KEY])
    if encrypted_payload is None:
        return None
    if not isinstance(encrypted_payload, str):
        raise RuntimeError("Сохранённая сессия имеет неожиданный формат")

    try:
        payload = get_session_cipher().decrypt(encrypted_payload.encode("ascii"))
        data = json.loads(payload.decode("utf-8"))
    except (InvalidToken, ValueError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("Не удалось расшифровать сохранённую сессию")

    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("cookies"), list)
        or not data["cookies"]
    ):
        raise RuntimeError("Сохранённая сессия имеет неподдерживаемый формат")
    return data["cookies"]


def restore_yandex_session() -> bool:
    """Restore cookies, then validate the Yandex account and exact personal station."""
    saved_cookies = load_yandex_session_cookies()
    if saved_cookies is None:
        log.info("Сохранённая сессия Яндекса в Redis не найдена")
        return False

    quasar = requests.Session()
    quasar.headers.update({"User-Agent": USER_AGENT})

    for item in saved_cookies:
        if not isinstance(item, dict):
            raise RuntimeError("Сохранённые cookies имеют неожиданный формат")
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            raise RuntimeError("Сохранённые cookies имеют неожиданный формат")

        quasar.cookies.set(
            name,
            value,
            domain=item.get("domain") or "",
            path=item.get("path") or "/",
            secure=bool(item.get("secure", False)),
            expires=item.get("expires"),
            rest=item.get("rest") if isinstance(item.get("rest"), dict) else {},
            discard=bool(item.get("discard", False)),
            version=int(item.get("version", 0)),
        )

    # A fresh CSRF token is obtained from Yandex; it is not stored in Redis.
    iot_page = quasar.get("https://yandex.ru/iot", timeout=20)
    iot_page.raise_for_status()
    csrf_match = re.search(r'"csrfToken2"\s*:\s*"([^"]+)"', iot_page.text)
    if not csrf_match:
        raise RuntimeError("Сохранённая авторизация Яндекса больше не действует")

    csrf_token = csrf_match.group(1)
    quasar.headers.update({"x-csrf-token": csrf_token})

    devices_data = response_json(
        quasar.get(f"{QUASAR_API}/v3/user/devices", timeout=20),
        "Проверка сохранённой авторизации",
    )
    households = devices_data.get("households", [])
    if not isinstance(households, list):
        raise RuntimeError("Яндекс вернул households в неожиданном формате")

    expected_name = " ".join(PERSONAL_STATION_NAME.split()).casefold()
    personal_speakers = []
    for household in households:
        if not isinstance(household, dict):
            continue
        household_devices = household.get("all", [])
        if not isinstance(household_devices, list):
            continue
        for device in household_devices:
            if not isinstance(device, dict):
                continue
            normalized_name = " ".join(
                str(device.get("name") or "").split()
            ).casefold()
            if normalized_name != expected_name:
                continue
            if not device.get("id") or not device.get("capabilities") or not device.get("quasar_info"):
                raise RuntimeError("Данные личной станции неполные")
            personal_speakers.append(device)

    if len(personal_speakers) != 1:
        raise RuntimeError(
            "Не удалось однозначно определить личную станцию "
            f"«{PERSONAL_STATION_NAME}». Найдено: {len(personal_speakers)}"
        )

    with _guard:
        _state.update(
            status="ready",
            message="Сохранённая авторизация Яндекса восстановлена. Можно пользоваться мостом.",
            link=None,
            session=quasar,
            csrf=csrf_token,
            devices=personal_speakers,
            error=None,
        )

    log.info("Сохранённая авторизация Яндекса успешно восстановлена")
    return True


def yandex_login():
    passport = requests.Session()
    passport.headers.update({"User-Agent": USER_AGENT})

    # Возвращаем проверенный способ получения CSRF-токена,
    # который использовался в самой первой версии теста.
    page = passport.get(
        "https://passport.yandex.ru/pwl-yandex",
        timeout=20,
    )
    page.raise_for_status()

    match = re.search(r'__CSRF__\s*=\s*"([^"]+)', page.text)
    if not match:
        raise RuntimeError(
            "Не удалось получить защитный токен страницы входа (__CSRF__)"
        )

    csrf_login = match.group(1)
    login_headers = {"X-CSRF-Token": csrf_login}

    start_data = response_json(
        passport.post(
            "https://passport.yandex.ru/pwl-yandex/api/passport/auth/password/submit",
            json={"retpath": "https://passport.yandex.ru/"},
            headers=login_headers,
            timeout=20,
        ),
        "Создание QR-входа",
    )
    track_id = start_data.get("track_id")
    if not track_id:
        raise RuntimeError("Яндекс не создал сессию QR-входа")

    magic = response_json(
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
    qr_link = magic.get("link")
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
                json=start_data,
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
    if not cookie_header:
        raise RuntimeError("Яндекс не выдал cookies после подтверждения входа")

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
            secure=cookie.secure,
            expires=cookie.expires,
            rest=dict(cookie._rest or {}),
            discard=cookie.discard,
            version=cookie.version,
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

    # В ответе Quasar устройства находятся внутри household["all"],
    # а не в корневом поле "devices".
    households = devices_data.get("households", [])
    log.info(
        "Quasar devices response: top_level_keys=%s; households_type=%s; households_count=%s",
        list(devices_data.keys()),
        type(households).__name__,
        len(households) if isinstance(households, list) else None,
    )

    if not isinstance(households, list):
        raise RuntimeError("Яндекс вернул households в неожиданном формате")

    all_devices = []
    for household_index, household in enumerate(households):
        if not isinstance(household, dict):
            continue
        household_name = household.get("name", "")
        household_devices = household.get("all", [])
        log.info(
            "Quasar household[%s]: name=%r; all_type=%s; all_count=%s",
            household_index,
            household_name,
            type(household_devices).__name__,
            len(household_devices) if isinstance(household_devices, list) else None,
        )
        if not isinstance(household_devices, list):
            continue
        for device in household_devices:
            if isinstance(device, dict):
                # Сохраняем название дома только для безопасной диагностики.
                device_copy = dict(device)
                device_copy["_household_name"] = household_name
                all_devices.append(device_copy)

    personal_speakers = []
    expected_name = " ".join(PERSONAL_STATION_NAME.split()).casefold()

    # Ищем строго по точному имени личной станции.
    # Устройства из второго дома не выбираются по одному лишь типу.
    for device in all_devices:
        name = device.get("name")
        normalized_name = " ".join(str(name or "").split()).casefold()
        if normalized_name != expected_name:
            continue

        capabilities = device.get("capabilities")
        quasar_info = device.get("quasar_info") or {}
        log.info(
            "Личная станция-кандидат: household=%r; id_present=%s; "
            "capabilities_type=%s; capabilities_count=%s; quasar_info_present=%s",
            device.get("_household_name"),
            bool(device.get("id")),
            type(capabilities).__name__,
            len(capabilities) if isinstance(capabilities, (list, dict)) else None,
            bool(quasar_info),
        )

        if not device.get("id"):
            raise RuntimeError("У личной станции отсутствует id")
        if not capabilities:
            raise RuntimeError("У личной станции отсутствуют capabilities")
        if not quasar_info:
            raise RuntimeError("У личной станции отсутствует quasar_info")
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

    # Persistence is best-effort: a Redis outage must not undo a working login.
    try:
        save_yandex_session(quasar)
        log.info("Зашифрованная сессия Яндекса сохранена в Redis")
    except Exception as exc:
        log.warning(
            "Не удалось сохранить сессию Яндекса в Redis: %s",
            type(exc).__name__,
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


@app.on_event("startup")
def restore_saved_session_on_startup():
    """Try to restore the saved Yandex session before the service accepts requests."""
    try:
        restore_yandex_session()
    except Exception as exc:
        with _guard:
            _state.update(
                status="idle",
                message="Не удалось восстановить авторизацию. При необходимости подключи Яндекс заново.",
                link=None,
                session=None,
                csrf=None,
                devices=[],
                error=None,
            )
        log.warning(
            "Восстановление сессии Яндекса не выполнено: %s",
            type(exc).__name__,
        )


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Safe health check: does not access Yandex or send commands to the speaker."""
    return {"status": "ok"}


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
def say(
    payload: dict = Body(...),
    _: bool = Depends(require_password),
):
    phrase = str(payload.get("phrase") or "").strip()
    if not phrase:
        raise HTTPException(400, "Введи текст для произнесения")
    if len(phrase) > 500:
        raise HTTPException(400, "Максимальная длина фразы — 500 символов")

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


@app.post("/api/bridge/say")
def bridge_say(
    payload: dict = Body(...),
    x_bridge_secret: str = Header(default=""),
):
    """Private endpoint for the main DeepSeek Alisa service."""
    if not BRIDGE_SECRET:
        raise HTTPException(503, "На Render не задана переменная TTS_BRIDGE_SECRET")
    if not secrets.compare_digest(x_bridge_secret, BRIDGE_SECRET):
        raise HTTPException(401, "Неверный ключ моста")

    phrase = str(payload.get("phrase") or "").strip()
    if not phrase:
        raise HTTPException(400, "Пустой текст")
    if len(phrase) > 500:
        raise HTTPException(400, "Максимальная длина фразы — 500 символов")

    with _guard:
        session = _state.get("session")
        csrf = _state.get("csrf")
        devices = _state.get("devices") or []
        if _state["status"] != "ready" or session is None or not csrf:
            raise HTTPException(409, "Мост не подключён к Яндексу")
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
        response_json(response, "Отправка фразы через мост")
        return {"ok": True, "message": "Команда отправлена на личную станцию"}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Ошибка отправки фразы через мост")
        raise HTTPException(502, f"Ошибка отправки: {type(exc).__name__}")


def send_voice_command(text: str, require_active_bridge: bool = False, diag_id: str = ""):
    """Отправляет Quasar text_action; автоматические команды закрываются при выходе."""
    diag_id = str(diag_id or "manual")[:32]
    log.info("DIAG command[%s] waiting bridge _guard active_required=%s", diag_id, require_active_bridge)
    # Держим блокировку до завершения HTTP-запроса: stop дождётся уже начатой
    # команды, затем поставит воспроизведение на паузу. После stop новые команды
    # не пройдут до следующего session/start.
    guard_wait_started = time.monotonic()
    with _guard:
        log.info("DIAG command[%s] acquired _guard wait=%.3fs bridge_enabled=%s", diag_id, time.monotonic() - guard_wait_started, _bridge_commands_enabled)
        if require_active_bridge and not _bridge_commands_enabled:
            log.info("DIAG command[%s] rejected: bridge disabled", diag_id)
            raise HTTPException(409, "Команды моста отменены: навык завершён")
        session = _state.get("session")
        csrf = _state.get("csrf")
        devices = _state.get("devices") or []
        if _state["status"] != "ready" or session is None or not csrf:
            raise HTTPException(409, "Мост не подключён к Яндексу")
        if len(devices) != 1:
            raise HTTPException(409, "Личная станция не определена")
        device_id = str(devices[0]["id"])
        try:
            log.info("DIAG command[%s] Quasar text_action POST BEGIN", diag_id)
            response = session.post(
                f"{QUASAR_API}/user/devices/{device_id}/actions",
                json={"actions": [{
                    "type": "devices.capabilities.quasar.server_action",
                    "state": {"instance": "text_action", "value": text},
                }]},
                headers={"x-csrf-token": csrf},
                timeout=20,
            )
            log.info("DIAG command[%s] Quasar text_action POST END http=%s", diag_id, response.status_code)
            response_json(response, "Отправка текстовой команды")
            log.info("DIAG command[%s] accepted by Quasar; _guard released on return", diag_id)
            return {"ok": True, "message": "Текстовая команда отправлена на личную станцию"}
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Ошибка отправки текстовой команды")
            raise HTTPException(502, f"Ошибка отправки команды: {type(exc).__name__}")


@app.post("/api/command")
def command(
    payload: dict = Body(...),
    _: bool = Depends(require_password),
):
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Введи текст команды")
    if len(text) > 500:
        raise HTTPException(400, "Максимальная длина команды — 500 символов")
    return send_voice_command(text)


@app.post("/api/bridge/command")
def bridge_command(
    payload: dict = Body(...),
    x_bridge_secret: str = Header(default=""),
):
    """Private endpoint: send a text command, not a spoken phrase."""
    if not BRIDGE_SECRET:
        raise HTTPException(503, "На Render не задана переменная TTS_BRIDGE_SECRET")
    if not secrets.compare_digest(x_bridge_secret, BRIDGE_SECRET):
        raise HTTPException(401, "Неверный ключ моста")

    text = str(payload.get("text") or "").strip()
    diag_id = str(payload.get("diag_id") or "")[:32]
    if not text:
        raise HTTPException(400, "Пустой текст команды")
    if len(text) > 500:
        raise HTTPException(400, "Максимальная длина команды — 500 символов")
    log.info("DIAG bridge command[%s] endpoint received", diag_id or "no-id")
    return send_voice_command(text, require_active_bridge=True, diag_id=diag_id)



@app.post("/api/bridge/session/start")
def bridge_session_start(x_bridge_secret: str = Header(default="")):
    """Разрешает автоматические команды при новом запуске навыка."""
    global _bridge_commands_enabled
    if not BRIDGE_SECRET:
        raise HTTPException(503, "На Render не задана переменная TTS_BRIDGE_SECRET")
    if not secrets.compare_digest(x_bridge_secret, BRIDGE_SECRET):
        raise HTTPException(401, "Неверный ключ моста")
    with _guard:
        _bridge_commands_enabled = True
    return {"ok": True, "message": "Автоматические команды разрешены"}


@app.post("/api/bridge/stop")
def bridge_stop(x_bridge_secret: str = Header(default=""), x_diag_trace: str = Header(default="")):
    """Отменяет будущие автоматические команды и ставит воспроизведение на паузу."""
    diag_id = str(x_diag_trace or "no-id")[:32]
    log.info("DIAG stop[%s] endpoint received; waiting bridge _guard", diag_id)
    guard_wait_started = time.monotonic()
    global _bridge_commands_enabled
    if not BRIDGE_SECRET:
        raise HTTPException(503, "На Render не задана переменная TTS_BRIDGE_SECRET")
    if not secrets.compare_digest(x_bridge_secret, BRIDGE_SECRET):
        raise HTTPException(401, "Неверный ключ моста")
    with _guard:
        log.info("DIAG stop[%s] acquired _guard wait=%.3fs; disabling automatic commands", diag_id, time.monotonic() - guard_wait_started)
        _bridge_commands_enabled = False
        session = _state.get("session")
        csrf = _state.get("csrf")
        devices = _state.get("devices") or []
        if _state["status"] != "ready" or session is None or not csrf:
            return {"ok": True, "message": "Команды заблокированы; мост пока не подключён"}
        if len(devices) != 1:
            return {"ok": True, "message": "Команды заблокированы; личная станция не определена"}
        device_id = str(devices[0]["id"])
        try:
            log.info("DIAG stop[%s] Quasar pause POST BEGIN", diag_id)
            response = session.post(
                f"{QUASAR_API}/user/devices/{device_id}/actions",
                json={"actions": [{
                    "type": "devices.capabilities.toggle",
                    "state": {"instance": "pause", "value": True},
                }]},
                headers={"x-csrf-token": csrf},
                timeout=20,
            )
            log.info("DIAG stop[%s] Quasar pause POST END http=%s", diag_id, response.status_code)
            response_json(response, "Остановка воспроизведения")
            log.info("DIAG stop[%s] pause accepted; bridge commands disabled", diag_id)
            return {"ok": True, "message": "Команды отменены, воспроизведение остановлено"}
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Ошибка остановки воспроизведения")
            raise HTTPException(502, f"Ошибка остановки: {type(exc).__name__}")



def _ws_read_exact(sock: ssl.SSLSocket, size: int, pending: bytearray) -> bytes:
    """Read exactly size bytes, consuming any bytes left after the HTTP upgrade."""
    if size < 0 or size > 1_048_576:
        raise RuntimeError("Недопустимый размер WebSocket-кадра")
    result = bytearray()
    if pending:
        take = min(size, len(pending))
        result.extend(pending[:take])
        del pending[:take]
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise RuntimeError("WebSocket-соединение закрыто сервером")
        result.extend(chunk)
    return bytes(result)


def _ws_send_control(sock: ssl.SSLSocket, opcode: int, payload: bytes = b"") -> None:
    """Send a masked client control frame (RFC 6455)."""
    if len(payload) > 125:
        raise RuntimeError("Слишком большой WebSocket control frame")
    mask = os.urandom(4)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    sock.sendall(bytes((0x80 | opcode, 0x80 | len(payload))) + mask + masked)


def _ws_receive_text(sock: ssl.SSLSocket, pending: bytearray) -> str | None:
    """Receive one complete text message; answer ping frames and ignore pong frames."""
    fragments = bytearray()
    fragmented_opcode = None
    while True:
        first, second = _ws_read_exact(sock, 2, pending)
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", _ws_read_exact(sock, 2, pending))[0]
        elif length == 127:
            length = struct.unpack("!Q", _ws_read_exact(sock, 8, pending))[0]
        if length > 1_048_576:
            raise RuntimeError("WebSocket-сообщение превышает диагностический лимит")
        mask = _ws_read_exact(sock, 4, pending) if masked else b""
        payload = _ws_read_exact(sock, length, pending) if length else b""
        if masked:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))

        if opcode == 0x8:  # close
            return None
        if opcode == 0x9:  # ping
            _ws_send_control(sock, 0xA, payload)
            continue
        if opcode == 0xA:  # pong
            continue
        if opcode == 0x0:  # continuation
            if fragmented_opcode is None:
                continue
            fragments.extend(payload)
            if fin:
                if fragmented_opcode != 0x1:
                    return None
                return fragments.decode("utf-8", errors="replace")
            continue
        if opcode in (0x1, 0x2):
            if fin:
                return payload.decode("utf-8", errors="replace") if opcode == 0x1 else None
            fragmented_opcode = opcode
            fragments = bytearray(payload)
            continue


def _quasar_websocket_connect(session: requests.Session, updates_url: str):
    """Open Quasar's read-only updates WebSocket using the existing Yandex cookies."""
    parsed = urlparse(updates_url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "wss" or not (hostname == "iot.quasar.yandex.ru" or hostname.endswith(".yandex.ru")):
        raise RuntimeError("Quasar вернул неожиданный адрес WebSocket")
    port = parsed.port or 443
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    prepared = session.prepare_request(requests.Request("GET", updates_url))
    cookie_header = prepared.headers.get("Cookie", "")
    csrf = session.headers.get("x-csrf-token", "")
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    host_header = hostname if port == 443 else f"{hostname}:{port}"

    raw_sock = socket.create_connection((hostname, port), timeout=8)
    tls_sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=hostname)
    try:
        request_headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host_header}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "Origin: https://yandex.ru",
            f"User-Agent: {USER_AGENT}",
        ]
        if cookie_header:
            request_headers.append(f"Cookie: {cookie_header}")
        if csrf:
            request_headers.append(f"x-csrf-token: {csrf}")
        tls_sock.sendall(("\r\n".join(request_headers) + "\r\n\r\n").encode("utf-8"))

        response_buffer = bytearray()
        while b"\r\n\r\n" not in response_buffer:
            chunk = tls_sock.recv(4096)
            if not chunk:
                raise RuntimeError("Quasar закрыл WebSocket handshake")
            response_buffer.extend(chunk)
            if len(response_buffer) > 16384:
                raise RuntimeError("Слишком большой ответ WebSocket handshake")
        header_bytes, remainder = bytes(response_buffer).split(b"\r\n\r\n", 1)
        header_lines = header_bytes.decode("iso-8859-1").split("\r\n")
        if not header_lines or " 101 " not in f" {header_lines[0]} ":
            raise RuntimeError("Quasar не подтвердил WebSocket upgrade")
        response_headers = {}
        for line in header_lines[1:]:
            if ":" in line:
                name, value = line.split(":", 1)
                response_headers[name.strip().lower()] = value.strip()
        expected_accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if response_headers.get("sec-websocket-accept") != expected_accept:
            raise RuntimeError("Некорректное подтверждение WebSocket от Quasar")
        return tls_sock, bytearray(remainder)
    except Exception:
        tls_sock.close()
        raise


@app.get("/api/diagnostic/quasar-updates")
def diagnostic_quasar_updates(
    seconds: int = 300,
    _: bool = Depends(require_password),
):
    """Listen to Quasar cloud update events for up to five minutes; read-only."""
    if seconds < 5 or seconds > 300:
        raise HTTPException(400, "Параметр seconds должен быть от 5 до 300")

    with _guard:
        session = _state.get("session")
        devices = _state.get("devices") or []
        if _state.get("status") != "ready" or session is None:
            raise HTTPException(409, "Сначала восстанови подключение к Яндексу")
        if len(devices) != 1:
            raise HTTPException(409, "Личная станция не определена")
        target_device_id = str(devices[0].get("id") or "")
    if not target_device_id:
        raise HTTPException(409, "У личной станции отсутствует идентификатор")

    try:
        devices_response = session.get(f"{QUASAR_API}/v3/user/devices", timeout=(5, 12))
        devices_data = response_json(devices_response, "Получение адреса облачных обновлений")
        if not isinstance(devices_data, dict) or devices_data.get("status") != "ok":
            raise RuntimeError("Quasar не подтвердил получение списка устройств")
        updates_url = devices_data.get("updates_url")
        if not isinstance(updates_url, str) or not updates_url:
            raise RuntimeError("В ответе Quasar отсутствует updates_url")
        ws, pending = _quasar_websocket_connect(session, updates_url)
    except Exception as exc:
        log.warning("Quasar updates diagnostic connect failed: %s", type(exc).__name__)
        raise HTTPException(502, f"Не удалось подключиться к облачным обновлениям: {type(exc).__name__}")

    # Only short machine-like enum values are exposed; arbitrary text, names, IDs,
    # URLs, commands and credentials are suppressed.
    safe_enum_keys = {"type", "instance", "property_type", "condition", "status", "operation"}
    sensitive_key_parts = ("id", "token", "cookie", "secret", "auth", "name", "text", "title", "url", "uri")

    def project(value, key="", depth=0):
        if depth > 8:
            return "[глубина ограничена]"
        if isinstance(value, dict):
            result = {}
            for raw_key, child in list(value.items())[:80]:
                child_key = str(raw_key)
                normalized = child_key.casefold().replace("-", "_")
                if normalized == "id" or normalized.endswith("_id") or any(part in normalized for part in sensitive_key_parts):
                    result[child_key] = "[скрыто]"
                else:
                    result[child_key] = project(child, child_key, depth + 1)
            return result
        if isinstance(value, list):
            return [project(child, key, depth + 1) for child in value[:30]]
        if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
            return value
        if isinstance(value, str):
            if key.casefold() in safe_enum_keys and re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
                return value
            return {"string_length": len(value)}
        return f"[{type(value).__name__}]"

    operation_counts = {}
    matching_updates = []
    update_messages = 0
    deadline = time.monotonic() + seconds
    timed_out = False
    try:
        while time.monotonic() < deadline:
            remaining = max(0.2, deadline - time.monotonic())
            ws.settimeout(remaining)
            try:
                raw_message = _ws_receive_text(ws, pending)
            except socket.timeout:
                timed_out = True
                break
            if raw_message is None:
                break
            try:
                outer = json.loads(raw_message)
            except Exception:
                continue
            if not isinstance(outer, dict):
                continue
            operation = str(outer.get("operation") or "unknown")
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            if operation != "update_states":
                continue
            update_messages += 1
            message = outer.get("message")
            try:
                inner = json.loads(message) if isinstance(message, str) else message
            except Exception:
                continue
            if not isinstance(inner, dict):
                continue
            updated_devices = inner.get("updated_devices")
            if not isinstance(updated_devices, list):
                continue
            for updated in updated_devices:
                if not isinstance(updated, dict) or str(updated.get("id") or "") != target_device_id:
                    continue
                if len(matching_updates) < 20:
                    matching_updates.append(project(updated))
    except Exception as exc:
        log.warning("Quasar updates diagnostic read failed: %s", type(exc).__name__)
        raise HTTPException(502, f"Ошибка чтения облачных событий: {type(exc).__name__}")
    finally:
        try:
            _ws_send_control(ws, 0x8)
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass

    return {
        "ok": True,
        "websocket_connected": True,
        "listen_seconds": seconds,
        "timed_out_without_more_events": timed_out,
        "operation_counts": operation_counts,
        "update_states_messages": update_messages,
        "matching_station_updates": len(matching_updates),
        "station_updates": matching_updates,
        "note": (
            "Подключение только читает облачные события Quasar и закрывается после окна наблюдения. "
            "По умолчанию окно составляет 300 секунд; его можно сократить параметром seconds (5–300). "
            "Команды станции не отправляются; произвольный текст, имена, ID, URL и секреты скрыты. "
            "Запусти ответ Дипсика в любой момент в течение окна наблюдения."
        ),
    }


@app.get("/api/diagnostic/quasar-device")
def diagnostic_quasar_device(_: bool = Depends(require_password)):
    """
    Read-only diagnostic: fetch the personal device object from Quasar and
    expose only playback-related primitive fields. No speaker actions are sent.
    """
    with _guard:
        session = _state.get("session")
        devices = _state.get("devices") or []
        if _state.get("status") != "ready" or session is None:
            raise HTTPException(409, "Сначала восстанови подключение к Яндексу")
        if len(devices) != 1:
            raise HTTPException(409, "Личная станция не определена")
        device = dict(devices[0])

    item_type = str(device.get("item_type") or "").strip()
    device_id = str(device.get("id") or "").strip()
    if not re.fullmatch(r"[a-zA-Z_]+", item_type) or not device_id:
        raise HTTPException(
            502,
            "В данных личной станции нет item_type или id для диагностического GET",
        )

    # Read-only Quasar device GET; this does not invoke /actions.
    url = f"{QUASAR_API}/user/{item_type}s/{device_id}"
    try:
        response = session.get(url, timeout=(5, 15))
        data = response_json(response, "Чтение состояния устройства")
    except Exception as exc:
        log.warning("Quasar device diagnostic GET failed: %s", type(exc).__name__)
        raise HTTPException(
            502,
            f"Не удалось прочитать устройство Quasar: {type(exc).__name__}",
        )

    # Inspect only the three potentially relevant containers. Return their
    # nested key structure and safe primitive values, never arbitrary text.
    inspected_sections = ("state", "status_info", "quasar_info")
    section_structure = {}
    safe_values = []
    excluded_key_fragments = ("token", "cookie", "secret", "auth", "name", "text", "title")
    # Only these two exact status fields may be returned as text. All other
    # string values remain hidden, including strings nested under these objects.
    safe_text_paths = {"state", "status_info.status"}

    def inspect_value(value, path, depth=0):
        if depth > 12:
            return {"_truncated": True}
        if isinstance(value, dict):
            result = {}
            for key, child in list(value.items())[:100]:
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                normalized = key_text.casefold().replace("-", "_")
                if normalized == "id" or normalized.endswith("_id") or any(fragment in normalized for fragment in excluded_key_fragments):
                    result[key_text] = "[скрыто]"
                    continue
                result[key_text] = inspect_value(child, child_path, depth + 1)
            return result
        if isinstance(value, list):
            return [inspect_value(child, f"{path}[{index}]", depth + 1)
                    for index, child in enumerate(value[:30])]
        if isinstance(value, bool):
            safe_values.append({"path": path, "type": "bool", "value": value})
            return value
        if isinstance(value, (int, float)) or value is None:
            safe_values.append({"path": path, "type": type(value).__name__, "value": value})
            return value
        if isinstance(value, str) and path in safe_text_paths:
            # Limit length and remove control characters before exposing the
            # two explicitly approved status strings.
            cleaned = "".join(ch for ch in value if ch.isprintable())
            return cleaned[:160]
        return "[текст скрыт]"

    for section in inspected_sections:
        if isinstance(data, dict) and section in data:
            section_structure[section] = inspect_value(data[section], section)
        else:
            section_structure[section] = "[отсутствует]"

    return {
        "ok": True,
        "http_status": response.status_code,
        "root_keys": list(data.keys())[:100] if isinstance(data, dict) else [],
        "sections": section_structure,
        "safe_values": safe_values[:200],
        "note": (
            "Показана вложенная структура state, status_info и quasar_info. "
            "Открытым текстом возвращаются только state и status_info.status "
            "(не более 160 символов каждый); остальные текстовые значения, "
            "идентификаторы, cookies и токены скрыты. Команды станции не отправляются."
        ),
    }



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
<label for="phrase">Текст для произнесения:</label>
<textarea id="phrase" rows="4" maxlength="500" placeholder="Напиши фразу, которую должна произнести колонка" style="box-sizing:border-box;width:100%;padding:12px;font-size:16px;margin:8px 0;"></textarea>
<button id="say" hidden>Произнести текст</button>
<label for="commandText">Текстовая команда Алисе (до 500 символов):</label>
<textarea id="commandText" rows="3" maxlength="500" placeholder="Например: Алиса, я жду" style="box-sizing:border-box;width:100%;padding:12px;font-size:16px;margin:8px 0;"></textarea>
<button id="commandButton" hidden>Отправить текстовую команду</button>

<script>
const statusBox = document.getElementById("status");
const linkBox = document.getElementById("link");
const sayButton = document.getElementById("say");
const phraseInput = document.getElementById("phrase");
const commandInput = document.getElementById("commandText");
const commandButton = document.getElementById("commandButton");

async function api(path, method = "GET", body = null) {
    const options = {method, headers: {}};
    if (body !== null) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail || "Ошибка запроса");
    }
    return data;
}

let refreshTimer = null;
let refreshInProgress = false;

function scheduleRefresh() {
    if (refreshTimer !== null) {
        clearTimeout(refreshTimer);
        refreshTimer = null;
    }
    refreshTimer = setTimeout(() => {
        refreshTimer = null;
        refresh();
    }, 2500);
}

async function refresh() {
    if (refreshInProgress) return;
    refreshInProgress = true;
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
        commandButton.hidden = data.status !== "ready";
        if (data.status === "waiting" || data.status === "working") {
            scheduleRefresh();
        } else if (refreshTimer !== null) {
            clearTimeout(refreshTimer);
            refreshTimer = null;
        }
    } catch (error) {
        statusBox.textContent = error.message;
        scheduleRefresh();
    } finally {
        refreshInProgress = false;
    }
}

// Мобильный браузер может приостановить JavaScript, пока открыто
// приложение Яндекса. При возвращении сразу перечитываем статус сервера.
window.addEventListener("pageshow", refresh);
window.addEventListener("focus", refresh);
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") refresh();
});

document.getElementById("start").addEventListener("click", async () => {
    statusBox.textContent = "Запускаю авторизацию...";
    sayButton.hidden = true;
    commandButton.hidden = true;
    linkBox.innerHTML = "";

    try {
        await api("/api/start", "POST");
        refresh();
    } catch (error) {
        statusBox.textContent = error.message;
    }
});

commandButton.addEventListener("click", async () => {
    const text = commandInput.value.trim();
    if (!text) {
        statusBox.textContent = "Сначала введи текстовую команду.";
        commandInput.focus();
        return;
    }
    commandButton.disabled = true;
    statusBox.textContent = "Отправляю текстовую команду на личную станцию...";
    try {
        const data = await api("/api/command", "POST", {text});
        statusBox.textContent = data.message;
    } catch (error) {
        statusBox.textContent = error.message;
    } finally {
        commandButton.disabled = false;
    }
});

sayButton.addEventListener("click", async () => {
    const phrase = phraseInput.value.trim();
    if (!phrase) {
        statusBox.textContent = "Сначала введи текст для произнесения.";
        phraseInput.focus();
        return;
    }
    sayButton.disabled = true;
    statusBox.textContent = "Отправляю текст на личную станцию...";

    try {
        const data = await api("/api/say", "POST", {phrase});
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