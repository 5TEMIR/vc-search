import json
from pathlib import Path
from typing import List, Optional
from ..models.article import Article, ArticleJSONEncoder


class JSONStorage:
    def __init__(self, output_file: str = "data/articles.json"):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Инициализируем файл если его нет
        if not self.output_file.exists():
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def save_articles_batch(self, articles: List[Article]) -> int:
        """Сохраняет пачку статей в JSON файл"""
        if not articles:
            return 0

        try:
            # Читаем существующие данные
            existing_articles = []
            if self.output_file.exists():
                with open(self.output_file, "r", encoding="utf-8") as f:
                    existing_articles = json.load(f)

            # Добавляем новые статьи
            existing_urls = {article["url"] for article in existing_articles}
            new_articles = [
                article for article in articles if article.url not in existing_urls
            ]

            if not new_articles:
                return 0

            # Конвертируем в dict и добавляем
            new_articles_dict = [article.to_dict() for article in new_articles]
            existing_articles.extend(new_articles_dict)

            # Записываем обратно
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(
                    existing_articles,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    cls=ArticleJSONEncoder,
                )

            print(f"✅ Сохранено {len(new_articles)} статей в {self.output_file}")
            return len(new_articles)

        except Exception as e:
            print(f"❌ Ошибка сохранения статей: {e}")
            return 0

    def get_article_count(self) -> int:
        """Возвращает количество статей в файле"""
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
                return len(articles)
        except:
            return 0
