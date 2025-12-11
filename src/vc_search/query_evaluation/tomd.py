import pandas as pd
import os
import glob


def csv_to_markdown_table(csv_file_path, output_file_path):
    """
    Преобразует CSV файл в таблицу Markdown
    """
    try:
        # Читаем CSV файл
        df = pd.read_csv(csv_file_path)

        # Выбираем только нужные колонки
        required_columns = ["relevance_score", "query", "title", "url"]

        # Проверяем, что все нужные колонки существуют
        for col in required_columns:
            if col not in df.columns:
                print(
                    f"Предупреждение: Колонка '{col}' не найдена в файле {csv_file_path}"
                )
                return

        # Создаем новую таблицу с нужными колонками
        markdown_df = df[required_columns].copy()

        # Оборачиваем URL в <>
        markdown_df["url"] = markdown_df["url"].apply(lambda x: f"<{x}>")

        # Создаем Markdown таблицу
        markdown_content = "| Релевантность | Запрос | Заголовок | URL |\n"
        markdown_content += "|---------------|--------|-----------|-----|\n"

        for _, row in markdown_df.iterrows():
            relevance = str(row["relevance_score"])
            query = str(row["query"])
            title = str(row["title"])
            url = str(row["url"])

            # Экранируем символы | в данных
            query = query.replace("|", "\\|")
            title = title.replace("|", "\\|")

            markdown_content += f"| {relevance} | {query} | {title} | {url} |\n"

        # Сохраняем в файл
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Файл {output_file_path} успешно создан")
        print(f"Обработано {len(markdown_df)} строк")

    except Exception as e:
        print(f"Ошибка при обработке файла {csv_file_path}: {str(e)}")


def process_all_csv_files():
    """
    Обрабатывает все CSV файлы в текущей директории
    """
    # Ищем все CSV файлы
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("CSV файлы не найдены в текущей директории")
        return

    for csv_file in csv_files:
        # Создаем имя для выходного файла
        base_name = os.path.splitext(csv_file)[0]
        output_file = f"{base_name}.md"

        print(f"Обрабатывается файл: {csv_file}")
        csv_to_markdown_table(csv_file, output_file)
        print("-" * 50)


def process_specific_csv_files():
    """
    Обрабатывает конкретные файлы из вашего примера
    """
    files_to_process = [
        "search_results_20251115_020918.csv",
        "search_results_20251115_034012.csv",
        "search_results_ml_20251115_062635.csv",
        "search_results_ml_20251115_062902.csv",
    ]

    for csv_file in files_to_process:
        if os.path.exists(csv_file):
            output_file = csv_file.replace(".csv", ".md")
            print(f"Обрабатывается файл: {csv_file}")
            csv_to_markdown_table(csv_file, output_file)
            print("-" * 50)
        else:
            print(f"Файл {csv_file} не найден")


if __name__ == "__main__":
    print("Преобразование CSV файлов в таблицы Markdown")
    print("=" * 50)

    # Обрабатываем все CSV файлы в директории
    process_all_csv_files()

    # Или обрабатываем конкретные файлы
    # process_specific_csv_files()
