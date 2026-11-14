
import os
from yandex_ai_studio_sdk import AIStudio


API_KEY = os.environ.get("YANDEX_API_KEY")
FOLDER_ID = "b1g6bcieampgfmvqn8v6"


def main():
    if not API_KEY:
        print("ОШИБКА: переменная YANDEX_API_KEY не найдена")
        return

    try:
        sdk = AIStudio(
            folder_id=FOLDER_ID,
            auth=API_KEY,
        )

        search = sdk.search_api.web("SEARCH_TYPE_RU").configure(
            family_mode="FAMILY_MODE_MODERATE",
            fix_typo_mode="FIX_TYPO_MODE_ON",
            sort_mode="SORT_MODE_BY_RELEVANCE",
            groups_on_page=5,
            sort_order="SORT_ORDER_DESC",
            group_mode="GROUP_MODE_DEEP",
            docs_in_group=1,
            max_passages=2,
            localization="LOCALIZATION_RU",
        )

        print("Отправляю тестовый запрос в Yandex Search API...")

        result = search.run(
            "Как работает искусственный интеллект?",
            format="xml",
            page=0,
        )

        print("УСПЕХ! Yandex Search API вернул ответ.")
        print("Размер ответа:", len(result), "байт")
        print(result[:500].decode("utf-8", errors="replace"))

    except Exception as error:
        print("ОШИБКА ПРИ ОБРАЩЕНИИ К YANDEX SEARCH API:")
        print(type(error).__name__, str(error))


if __name__ == "__main__":
    main()
