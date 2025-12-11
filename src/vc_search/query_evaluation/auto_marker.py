import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class RelevanceAutoMarker:
    def __init__(self, max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)):
        """Инициализация векторизатора TF-IDF"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=[
                "и",
                "в",
                "на",
                "с",
                "по",
                "о",
                "для",
                "не",
                "что",
                "это",
                "а",
                "но",
                "или",
                "же",
                "бы",
                "ли",
                "только",
                "уже",
                "еще",
                "все",
                "так",
                "как",
                "у",
                "из",
                "от",
                "то",
                "за",
                "же",
                "мы",
                "вы",
                "они",
                "он",
                "она",
                "оно",
                "я",
                "ты",
                "вы",
            ],
        )
        self.fitted = False

    def preprocess_text(self, text):
        """Предобработка текста: очистка, приведение к нижнему регистру"""
        if not isinstance(text, str):
            return ""

        # Убираем лишние символы, оставляем буквы, цифры и пробелы
        text = re.sub(r"[^\w\s]", " ", text.lower())
        # Убираем лишние пробелы
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def calculate_cosine_similarity(self, queries, contents):
        """Вычисляет косинусное сходство между запросами и контентом"""
        try:
            # Предобработка текстов
            processed_queries = [self.preprocess_text(q) for q in queries]
            processed_contents = [self.preprocess_text(c) for c in contents]

            # Проверяем, что есть что векторизовать
            if not any(processed_queries) or not any(processed_contents):
                print("⚠️  Один из текстов пустой после предобработки")
                return np.zeros(len(queries))

            # Объединяем все тексты для векторизации
            all_texts = processed_queries + processed_contents

            # Создаем TF-IDF матрицу
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            self.fitted = True

            # Разделяем обратно на запросы и контент
            n_queries = len(queries)
            query_vectors = tfidf_matrix[:n_queries]
            content_vectors = tfidf_matrix[n_queries:]

            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_vectors, content_vectors)

            # Для парных сравнений берем диагональ
            scores = []
            for i in range(min(n_queries, len(contents))):
                if i < similarities.shape[0] and i < similarities.shape[1]:
                    scores.append(similarities[i][i])
                else:
                    scores.append(0.0)

            return np.array(scores)

        except Exception as e:
            print(f"❌ Ошибка при вычислении сходства: {e}")
            return np.zeros(len(queries))

    def find_optimal_threshold(self, scores):
        """Находит оптимальный порог для бинарной классификации"""
        if len(scores) == 0:
            return 0.3

        # Фильтруем нулевые и очень низкие значения
        non_zero_scores = scores[scores > 0.01]

        if len(non_zero_scores) < 10:
            # Если мало данных, используем эмпирический порог
            if len(scores) > 0:
                mean_score = np.mean(scores[scores > 0])
                return max(0.2, min(0.5, mean_score * 1.5))
            return 0.3

        try:
            # Метод 1: Используем статистику распределения
            mean_score = np.mean(non_zero_scores)
            std_score = np.std(non_zero_scores)

            # Начальный порог: среднее + 0.5 стандартных отклонения
            threshold1 = mean_score + 0.5 * std_score

            # Метод 2: Используем квантили
            threshold2 = np.percentile(non_zero_scores, 75)

            # Метод 3: Ищем "долинку" в гистограмме
            hist, bin_edges = np.histogram(non_zero_scores, bins=30)

            # Находим локальные минимумы в гистограмме
            minima_indices = []
            for i in range(1, len(hist) - 1):
                if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
                    minima_indices.append(i)

            if minima_indices:
                # Берем первый значительный минимум
                first_min_idx = minima_indices[0]
                threshold3 = bin_edges[first_min_idx]
            else:
                threshold3 = threshold2

            # Усредняем пороги
            thresholds = [threshold1, threshold2, threshold3]
            valid_thresholds = [t for t in thresholds if 0.1 <= t <= 0.9]

            if valid_thresholds:
                final_threshold = np.mean(valid_thresholds)
            else:
                final_threshold = 0.3

            # Ограничиваем разумными значениями
            final_threshold = max(0.15, min(0.7, final_threshold))

            print(f"   Метод 1 (среднее+0.5σ): {threshold1:.4f}")
            print(f"   Метод 2 (75-й перцентиль): {threshold2:.4f}")
            print(f"   Метод 3 (локальный минимум): {threshold3:.4f}")

            return final_threshold

        except Exception as e:
            print(f"   ⚠️  Ошибка при определении порога: {e}")
            # Fallback на простой метод
            return (
                np.percentile(non_zero_scores, 70) if len(non_zero_scores) > 0 else 0.3
            )

    def mark_csv_file(self, input_csv, output_csv=None, threshold=None):
        """
        Автоматически размечает CSV файл с результатами поиска.
        Заполняет столбец relevance_score значениями 0 или 1.
        """
        if output_csv is None:
            input_path = Path(input_csv)
            output_csv = f"{input_path.stem}_auto_marked{input_path.suffix}"

        print(f"📖 Чтение файла: {input_csv}")

        try:
            # Читаем CSV файл
            df = pd.read_csv(input_csv, encoding="utf-8")
        except Exception as e:
            print(f"❌ Ошибка чтения CSV: {e}")
            # Пробуем другие кодировки
            try:
                df = pd.read_csv(input_csv, encoding="cp1251")
                print("✅ Файл прочитан с кодировкой cp1251")
            except:
                print("❌ Не удалось прочитать файл")
                return None, None

        print(f"📊 Загружено {len(df)} строк")

        # Проверяем наличие необходимых колонок
        if "query" not in df.columns:
            print("❌ В файле нет колонки 'query'")
            return None, None

        if "article_content" not in df.columns:
            # Проверяем другие возможные названия
            content_columns = [col for col in df.columns if "content" in col.lower()]
            if content_columns:
                df["article_content"] = df[content_columns[0]]
                print(
                    f"✅ Используем колонку '{content_columns[0]}' как article_content"
                )
            else:
                print("❌ В файле нет колонки с контентом")
                return None, None

        # Проверяем наличие столбца для оценки
        if "relevance_score" not in df.columns:
            print("⚠️  Столбец 'relevance_score' не найден, создаем...")
            df["relevance_score"] = ""

        # Заполняем пустые значения
        df["query"] = df["query"].fillna("")
        df["article_content"] = df["article_content"].fillna("")

        # Вычисляем косинусное сходство
        print("🔍 Вычисление косинусного сходства между запросами и контентом...")
        queries = df["query"].tolist()
        contents = df["article_content"].tolist()

        scores = self.calculate_cosine_similarity(queries, contents)

        if len(scores) == 0:
            print("❌ Не удалось вычислить сходство")
            return None, None

        # Анализируем распределение scores
        print(f"\n📈 Статистика сходства:")
        print(f"   Минимум: {scores.min():.4f}")
        print(f"   Максимум: {scores.max():.4f}")
        print(f"   Среднее: {scores.mean():.4f}")
        print(f"   Медиана: {np.median(scores):.4f}")
        print(f"   Стандартное отклонение: {scores.std():.4f}")

        # Определяем порог
        if threshold is None:
            print("\n🎯 Автоматический поиск оптимального порога...")
            threshold = self.find_optimal_threshold(scores)
            print(f"   Выбранный порог: {threshold:.4f}")
        else:
            print(f"🎯 Используется заданный порог: {threshold:.4f}")

        # Применяем порог для получения бинарных оценок
        binary_scores = (scores >= threshold).astype(int)

        # Заполняем столбец relevance_score
        df["relevance_score"] = binary_scores

        # Статистика разметки
        relevant_count = binary_scores.sum()
        relevant_percent = (relevant_count / len(df)) * 100

        print(f"\n✅ Разметка завершена:")
        print(
            f"   Релевантных (1): {relevant_count}/{len(df)} ({relevant_percent:.1f}%)"
        )
        print(
            f"   Нерелевантных (0): {len(df) - relevant_count}/{len(df)} ({100 - relevant_percent:.1f}%)"
        )

        # Проверяем распределение
        unique_values, counts = np.unique(binary_scores, return_counts=True)
        for val, count in zip(unique_values, counts):
            print(f"   Значение {val}: {count} строк ({count / len(df) * 100:.1f}%)")

        # Сохраняем результат
        try:
            df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"\n💾 Результат сохранен в: {output_csv}")

            # Показываем первые несколько строк для проверки
            print(f"\n👀 Предпросмотр первых 5 строк:")
            print(df[["relevance_score", "query"]].head().to_string())

            return df, output_csv

        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")
            return None, None

    def analyze_results(self, df):
        """Анализирует результаты разметки"""
        if df is None or len(df) == 0:
            print("❌ Нет данных для анализа")
            return None

        print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")

        # Проверяем столбец relevance_score
        if "relevance_score" not in df.columns:
            print("❌ Столбец 'relevance_score' отсутствует")
            return None

        # Проверяем значения
        unique_values = df["relevance_score"].unique()
        print(f"✅ Уникальные значения в 'relevance_score': {sorted(unique_values)}")

        # Статистика по запросам
        if "query" in df.columns:
            print("\n📝 Статистика по запросам:")

            query_stats = (
                df.groupby("query")["relevance_score"]
                .agg(
                    [
                        ("total", "count"),
                        ("relevant", "sum"),
                        ("relevance_rate", lambda x: x.mean() * 100),
                    ]
                )
                .round(2)
            )

            query_stats = query_stats.sort_values("relevance_rate", ascending=False)

            print(f"\nТоп-5 самых релевантных запросов:")
            print(query_stats.head(5).to_string())

            print(f"\nТоп-5 наименее релевантных запросов:")
            print(query_stats.tail(5).to_string())

            return query_stats
        else:
            print("⚠️  Колонка 'query' отсутствует для детального анализа")
            return None


def main():
    """Главная функция для запуска из командной строки"""
    import sys
    import os

    print("=" * 60)
    print("🤖 АВТОМАТИЧЕСКАЯ РАЗМЕТКА РЕЛЕВАНТНОСТИ")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nИспользование:")
        print("  python auto_marker.py <input_csv> [output_csv] [threshold]")
        print("\nПараметры:")
        print("  input_csv   - входной CSV файл с результатами поиска")
        print("  output_csv  - выходной CSV файл (опционально)")
        print("  threshold   - порог релевантности 0.0-1.0 (опционально)")
        print("\nПримеры:")
        print("  python auto_marker.py search_results.csv")
        print("  python auto_marker.py results.csv marked_results.csv")
        print("  python auto_marker.py results.csv marked.csv 0.35")
        return

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # Проверяем существование файла
    if not os.path.exists(input_csv):
        print(f"\n❌ Файл {input_csv} не найден!")
        return

    # Создаем маркер и запускаем разметку
    marker = RelevanceAutoMarker()

    print(f"\n🎯 Начинаем автоматическую разметку...")
    print(f"📁 Входной файл: {input_csv}")
    if output_csv:
        print(f"📁 Выходной файл: {output_csv}")
    if threshold is not None:
        print(f"🎯 Порог: {threshold}")

    df, output_file = marker.mark_csv_file(input_csv, output_csv, threshold)

    if df is not None and output_file is not None:
        # Анализируем результаты
        marker.analyze_results(df)

        print(f"\n✅ Разметка успешно завершена!")
        print(f"📊 Файл сохранен: {output_file}")

        # Показываем статистику по файлу
        file_size = os.path.getsize(output_file) / 1024
        print(f"📦 Размер файла: {file_size:.1f} KB")

    else:
        print("\n❌ Разметка не удалась!")


if __name__ == "__main__":
    main()
