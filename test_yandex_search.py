
import os

from yandex_ai_studio_sdk import AIStudio

API_KEY = os.environ.get("YANDEX_API_KEY")
FOLDER_ID = "b1g6bcieampgfmvqn8v6"

if not API_KEY:
    raise RuntimeError("YANDEX_API_KEY не найден")

sdk = AIStudio(
    folder_id=FOLDER_ID,
    auth=API_KEY,
)

search = sdk.search_api.web(
    search_type="ru",
    family_mode="moderate",
)

query = "Как работает искусственный интеллект?"

print("Отправляем запрос в Yandex Search API...")

result = search.run(query, format="xml")

if isinstance(result, bytes):
    result = result.decode("utf-8", errors="replace")

print("ПОИСК УСПЕШНО ВЫПОЛНЕН!")
print(result[:3000])
