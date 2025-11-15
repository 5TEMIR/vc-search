import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path


class RelevanceAutoMarker:
    def __init__(self, max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=["и", "в", "на", "с", "по", "о", "для", "не", "что", "это"],
        )

    def preprocess_text(self, text):
        """Предобработка текста"""
        if not isinstance(text, str):
            return ""

        # Убираем специальные символы, оставляем кириллицу и базовую пунктуацию
        text = re.sub(r"[^а-яёa-z0-9\s\.\,\!\?]", " ", text.lower())
        # Убираем лишние пробелы
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def calculate_relevance_scores(self, queries, contents):
        """Вычисляет релевантность между запросами и контентом"""
        try:
            # Предобработка текстов
            processed_queries = [self.preprocess_text(q) for q in queries]
            processed_contents = [self.preprocess_text(c) for c in contents]

            # Объединяем все тексты для векторизации
            all_texts = processed_queries + processed_contents

            # Создаем TF-IDF матрицу
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Разделяем обратно на запросы и контент
            query_vectors = tfidf_matrix[: len(queries)]
            content_vectors = tfidf_matrix[len(queries) :]

            # Вычисляем косинусное сходство для каждой пары запрос-контент
            similarities = cosine_similarity(query_vectors, content_vectors)

            # Берем диагональ (каждый запрос с соответствующим контентом)
            scores = [similarities[i][i] for i in range(len(queries))]
            return scores

        except Exception as e:
            print(f"Ошибка при вычислении релевантности: {e}")
            return [0.0] * len(queries)

    def mark_csv_file(self, input_csv, output_csv=None, threshold=None):
        """Автоматически размечает CSV файл с результатами поиска"""
        if output_csv is None:
            input_path = Path(input_csv)
            output_csv = f"{input_path.stem}_marked{input_path.suffix}"

        # Читаем CSV файл
        print(f"📖 Чтение файла: {input_csv}")
        df = pd.read_csv(input_csv)

        # Проверяем наличие необходимых колонок
        required_columns = ["query", "article_content"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        print(f"📊 Загружено {len(df)} строк")

        # Заполняем пустые значения
        df["query"] = df["query"].fillna("")
        df["article_content"] = df["article_content"].fillna("")

        # Вычисляем релевантность
        print("🔍 Вычисление релевантности...")
        queries = df["query"].tolist()
        contents = df["article_content"].tolist()

        scores = self.calculate_relevance_scores(queries, contents)

        # Анализируем распределение scores для определения порога
        scores_array = np.array(scores)

        print(f"📈 Статистика релевантности:")
        print(f"   Минимум: {scores_array.min():.4f}")
        print(f"   Максимум: {scores_array.max():.4f}")
        print(f"   Среднее: {scores_array.mean():.4f}")
        print(f"   Медиана: {np.median(scores_array):.4f}")
        print(f"   Стандартное отклонение: {scores_array.std():.4f}")

        # Автоматическое определение порога если не задан
        if threshold is None:
            # Используем адаптивный порог на основе квантилей
            threshold = np.percentile(scores_array, 70)  # 70-й перцентиль
            print(f"🎯 Автоматически выбран порог: {threshold:.4f}")
        else:
            print(f"🎯 Используется заданный порог: {threshold:.4f}")

        # Размечаем данные - только 1 или 0
        df["relevance_score"] = scores
        df["relevance"] = (scores_array >= threshold).astype(int)

        # Статистика разметки
        relevant_count = df["relevance"].sum()
        relevant_percent = (relevant_count / len(df)) * 100

        print(f"✅ Разметка завершена:")
        print(
            f"   Релевантных (1): {relevant_count}/{len(df)} ({relevant_percent:.1f}%)"
        )
        print(
            f"   Нерелевантных (0): {len(df) - relevant_count}/{len(df)} ({(100 - relevant_percent):.1f}%)"
        )

        # Проверяем, что в столбце relevance только 0 и 1
        unique_values = df["relevance"].unique()
        print(f"   Уникальные значения в столбце relevance: {sorted(unique_values)}")

        # Сохраняем результат
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"💾 Результат сохранен в: {output_csv}")

        return df, output_csv

    def analyze_results(self, df):
        """Анализирует результаты разметки"""
        print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")

        # Проверяем, что в столбце relevance только 0 и 1
        unique_values = df["relevance"].unique()
        if set(unique_values) <= {0, 1}:
            print("✅ В столбце 'relevance' только значения 0 и 1")
        else:
            print(f"⚠️  В столбце 'relevance' неожиданные значения: {unique_values}")

        # Статистика по запросам
        query_stats = (
            df.groupby("query")
            .agg(
                {
                    "relevance": ["count", "sum", "mean"],
                    "relevance_score": ["mean", "std"],
                }
            )
            .round(3)
        )

        query_stats.columns = [
            "total",
            "relevant",
            "relevance_rate",
            "score_mean",
            "score_std",
        ]
        query_stats = query_stats.sort_values("relevance_rate", ascending=False)

        print("📝 Статистика по запросам (топ-10 по релевантности):")
        print(query_stats.head(10).to_string())

        return query_stats


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Автоматическая разметка релевантности поисковых результатов"
    )
    parser.add_argument("input_csv", help="Входной CSV файл с результатами поиска")
    parser.add_argument(
        "--output", "-o", help="Выходной CSV файл (по умолчанию: input_marked.csv)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, help="Порог релевантности (0-1)"
    )
    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Показать детальный анализ результатов",
    )

    args = parser.parse_args()

    if not Path(args.input_csv).exists():
        print(f"❌ Файл {args.input_csv} не найден!")
        return

    # Создаем маркер и размечаем данные
    marker = RelevanceAutoMarker()

    try:
        df, output_file = marker.mark_csv_file(
            input_csv=args.input_csv, output_csv=args.output, threshold=args.threshold
        )

        if args.analyze:
            marker.analyze_results(df)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return


if __name__ == "__main__":
    main()
