import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import re
from pathlib import Path
import joblib


class RelevanceModel:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=[
                "в",
                "для",
                "и",
                "или",
                "на",
                "с",
                "по",
                "о",
                "об",
                "как",
                "где",
                "что",
                "почему",
                "зачем",
                "когда",
                "если",
                "то",
                "это",
                "так",
                "же",
                "но",
                "а",
                "да",
                "нет",
            ],
        )
        self.model = None
        self.is_trained = False

    def preprocess_text(self, text):
        """Предобработка текста: очистка, приведение к нижнему регистру"""
        if not isinstance(text, str):
            return ""

        # Убираем лишние символы, оставляем буквы, цифры и пробелы
        text = re.sub(r"[^\w\s]", " ", text.lower())
        # Убираем лишние пробелы
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def prepare_features(self, df):
        """Подготовка признаков для обучения"""
        # Предобработка текстов
        queries = [self.preprocess_text(q) for q in df["query"]]
        contents = [self.preprocess_text(c) for c in df["article_content"]]

        # Создаем комбинированные признаки
        combined_texts = [f"{q} {c}" for q, c in zip(queries, contents)]

        # Векторизуем текст
        if self.is_trained:
            X = self.vectorizer.transform(combined_texts)
        else:
            X = self.vectorizer.fit_transform(combined_texts)

        return X

    def train(self, csv_file, test_size=0.2, random_state=42, train_only=False):
        """
        Обучение модели на размеченных данных

        Args:
            csv_file: Путь к CSV файлу с данными
            test_size: Доля тестовых данных (игнорируется если train_only=True)
            random_state: Seed для воспроизводимости
            train_only: Если True, использует все данные для тренировки без тестов
        """
        print(f"📖 Загрузка данных из {csv_file}")
        df = pd.read_csv(csv_file)

        # Проверяем наличие необходимых колонок
        required_columns = ["query", "article_content", "relevance_score"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        # Заполняем пропуски
        df["query"] = df["query"].fillna("")
        df["article_content"] = df["article_content"].fillna("")
        df["relevance_score"] = df["relevance_score"].fillna(0).astype(int)

        print(f"📊 Загружено {len(df)} примеров")
        print(
            f"📈 Распределение классов: {df['relevance_score'].value_counts().to_dict()}"
        )

        # Подготовка признаков
        print("🔧 Подготовка признаков...")
        X = self.prepare_features(df)
        y = df["relevance_score"].values

        if train_only:
            # Используем все данные для тренировки
            print("🎯 Используем ВСЕ данные для тренировки (без тестовой выборки)")
            X_train, X_test, y_train, y_test = X, None, y, None
            test_size_used = 0.0
        else:
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            test_size_used = test_size

        print(f"📐 Размер тренировочных данных: {X_train.shape[0]} примеров")
        if not train_only:
            print(f"📐 Размер тестовых данных: {X_test.shape[0]} примеров")

        # Выбор и обучение модели
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight="balanced",
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=random_state, class_weight="balanced", max_iter=1000
            )
        elif self.model_type == "svm":
            self.model = SVC(
                random_state=random_state, class_weight="balanced", probability=True
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

        print(f"🤖 Обучение модели {self.model_type}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        if train_only:
            # Если тренируем только на всех данных, нет тестов для оценки
            print("✅ Модель обучена на всех данных")

            # Можно провести кросс-валидацию или оценить на тренировочных данных
            y_train_pred = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)

            print(f"📊 Метрики на тренировочных данных:")
            print(f"   Точность (accuracy): {train_accuracy:.4f}")
            print(f"   F1-score: {train_f1:.4f}")

            # Сохранение важных признаков (для Random Forest)
            if hasattr(self.model, "feature_importances_"):
                feature_names = self.vectorizer.get_feature_names_out()
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]

                print("\n🔝 Топ-10 важных признаков:")
                for i, idx in enumerate(top_indices):
                    print(f"   {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

            return train_accuracy, train_f1, None, None
        else:
            # Оценка модели на тестовых данных
            print("📊 Оценка модели на тестовых данных...")
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"✅ Точность (accuracy): {accuracy:.4f}")
            print(f"🎯 F1-score: {f1:.4f}")
            print("\n📋 Classification Report:")
            print(classification_report(y_test, y_pred))

            # Сохранение важных признаков (для Random Forest)
            if hasattr(self.model, "feature_importances_"):
                feature_names = self.vectorizer.get_feature_names_out()
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]

                print("\n🔝 Топ-10 важных признаков:")
                for i, idx in enumerate(top_indices):
                    print(f"   {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

            return accuracy, f1, X_test, y_test

    def predict(self, query, content):
        """Предсказание релевантности для одной пары запрос-контент"""
        if not self.is_trained:
            raise ValueError("Модель не обучена! Сначала вызовите метод train()")

        # Предобработка
        processed_query = self.preprocess_text(query)
        processed_content = self.preprocess_text(content)
        combined_text = f"{processed_query} {processed_content}"

        # Векторизация
        X = self.vectorizer.transform([combined_text])

        # Предсказание
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]

        return prediction, probability

    def predict_batch(self, queries, contents):
        """Пакетное предсказание релевантности"""
        if not self.is_trained:
            raise ValueError("Модель не обучена! Сначала вызовите метод train()")

        # Предобработка
        processed_queries = [self.preprocess_text(q) for q in queries]
        processed_contents = [self.preprocess_text(c) for c in contents]
        combined_texts = [
            f"{q} {c}" for q, c in zip(processed_queries, processed_contents)
        ]

        # Векторизация
        X = self.vectorizer.transform(combined_texts)

        # Предсказание
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def save(self, model_path):
        """Сохранение модели"""
        if not self.is_trained:
            raise ValueError("Модель не обучена!")

        model_data = {
            "model_type": self.model_type,
            "vectorizer": self.vectorizer,
            "model": self.model,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, model_path)
        print(f"💾 Модель сохранена в {model_path}")

    def load(self, model_path):
        """Загрузка модели"""
        model_data = joblib.load(model_path)

        self.model_type = model_data["model_type"]
        self.vectorizer = model_data["vectorizer"]
        self.model = model_data["model"]
        self.is_trained = model_data["is_trained"]

        print(f"📥 Модель загружена из {model_path}")
        print(f"🤖 Тип модели: {self.model_type}")
        print(f"📊 Обучена: {self.is_trained}")

    def cross_validate(self, csv_file, n_folds=5, random_state=42):
        """
        Кросс-валидация для оценки модели без разделения на train/test

        Args:
            csv_file: Путь к CSV файлу с данными
            n_folds: Количество фолдов для кросс-валидации
            random_state: Seed для воспроизводимости
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        print(f"📖 Загрузка данных из {csv_file}")
        df = pd.read_csv(csv_file)

        # Проверяем наличие необходимых колонок
        required_columns = ["query", "article_content", "relevance_score"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        # Заполняем пропуски
        df["query"] = df["query"].fillna("")
        df["article_content"] = df["article_content"].fillna("")
        df["relevance_score"] = df["relevance_score"].fillna(0).astype(int)

        print(f"📊 Загружено {len(df)} примеров")

        # Подготовка признаков
        print("🔧 Подготовка признаков для кросс-валидации...")
        X = self.prepare_features(df)
        y = df["relevance_score"].values

        # Выбор модели
        if self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight="balanced",
            )
        elif self.model_type == "logistic_regression":
            model = LogisticRegression(
                random_state=random_state, class_weight="balanced", max_iter=1000
            )
        elif self.model_type == "svm":
            model = SVC(
                random_state=random_state, class_weight="balanced", probability=True
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

        # Кросс-валидация
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        print(f"🔍 Проводим {n_folds}-фолдовую кросс-валидацию...")

        # Оценка accuracy
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(
            f"📊 Accuracy (среднее по {n_folds} фолдам): {accuracy_scores.mean():.4f} (±{accuracy_scores.std():.4f})"
        )

        # Оценка F1-score
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        print(
            f"🎯 F1-score (среднее по {n_folds} фолдам): {f1_scores.mean():.4f} (±{f1_scores.std():.4f})"
        )

        return accuracy_scores, f1_scores


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Обучение модели релевантности")
    parser.add_argument("input_csv", help="CSV файл с размеченными данными")
    parser.add_argument(
        "--model-type",
        "-m",
        choices=["random_forest", "logistic_regression", "svm"],
        default="random_forest",
        help="Тип модели для обучения",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="relevance_model.pkl",
        help="Путь для сохранения модели",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.2,
        help="Доля тестовых данных (не используется с --train-only)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Использовать все данные для тренировки (без тестовой выборки)",
    )
    parser.add_argument(
        "--cross-validate",
        "-c",
        action="store_true",
        help="Провести кросс-валидацию вместо обычного обучения",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Количество фолдов для кросс-валидации (только с --cross-validate)",
    )

    args = parser.parse_args()

    if not Path(args.input_csv).exists():
        print(f"❌ Файл {args.input_csv} не найден!")
        return

    # Создание модели
    model = RelevanceModel(model_type=args.model_type)

    try:
        if args.cross_validate:
            # Кросс-валидация
            print("🧪 ПРОВОДИМ КРОСС-ВАЛИДАЦИЮ")
            print("=" * 50)

            accuracy_scores, f1_scores = model.cross_validate(
                csv_file=args.input_csv, n_folds=args.folds, random_state=42
            )

            print("=" * 50)
            print(f"🎯 Результаты кросс-валидации ({args.folds} фолдов):")
            print(
                f"   Accuracy: {accuracy_scores.mean():.4f} (±{accuracy_scores.std():.4f})"
            )
            print(f"   F1-score: {f1_scores.mean():.4f} (±{f1_scores.std():.4f})")

            # После кросс-валидации все равно обучаем на всех данных
            print("\n🤖 Теперь обучаем модель на ВСЕХ данных...")
            train_accuracy, train_f1, _, _ = model.train(
                csv_file=args.input_csv, train_only=True
            )

            # Сохранение модели
            model.save(args.output)

            print(f"\n🎉 Модель успешно обучена и сохранена!")
            print(f"📁 Файл модели: {args.output}")

        elif args.train_only:
            # Обучение только на всех данных без тестов
            print("🎯 ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ (БЕЗ ТЕСТОВОЙ ВЫБОРКИ)")
            print("=" * 50)

            train_accuracy, train_f1, _, _ = model.train(
                csv_file=args.input_csv, test_size=args.test_size, train_only=True
            )

            # Сохранение модели
            model.save(args.output)

            print(f"\n🎉 Модель успешно обучена и сохранена!")
            print(f"📁 Файл модели: {args.output}")
            print(
                f"📊 Метрики на тренировочных данных: Accuracy={train_accuracy:.4f}, F1={train_f1:.4f}"
            )

        else:
            # Стандартное обучение с разделением на train/test
            print("🎯 ОБЫЧНОЕ ОБУЧЕНИЕ С ТЕСТОВОЙ ВЫБОРКОЙ")
            print("=" * 50)

            accuracy, f1, _, _ = model.train(
                csv_file=args.input_csv, test_size=args.test_size, train_only=False
            )

            # Сохранение модели
            model.save(args.output)

            print(f"\n🎉 Модель успешно обучена и сохранена!")
            print(f"📁 Файл модели: {args.output}")
            print(f"📊 Итоговые метрики: Accuracy={accuracy:.4f}, F1={f1:.4f}")

        # Тестовое предсказание
        print("\n🧪 Тестовое предсказание:")
        test_query = "ИИ в трейдинге"
        test_content = (
            "Искусственный интеллект активно применяется в алгоритмическом трейдинге..."
        )
        prediction, probability = model.predict(test_query, test_content)
        print(f"   Запрос: '{test_query}'")
        print(f"   Релевантность: {prediction} (вероятность: {probability:.4f})")

        # Информация о модели
        print("\n📋 ИНФОРМАЦИЯ О МОДЕЛИ:")
        print(f"   Тип модели: {args.model_type}")
        print(f"   Размер словаря: {len(model.vectorizer.get_feature_names_out())}")
        print(f"   Модель обучена: {model.is_trained}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
