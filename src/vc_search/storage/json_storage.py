import json
from datetime import datetime
from pathlib import Path
from typing import List

from ..models.article import Article


class JSONStorage:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.articles_dir = self.output_dir / "articles"
        self.articles_dir.mkdir(exist_ok=True)

    def save_articles_batch(
        self, articles: List[Article], batch_num: int, section: str
    ):
        """Сохраняет пачку статей в отдельные JSON файлы"""
        saved_count = 0

        for i, article in enumerate(articles):
            try:
                # Создаем имя файла на основе ID из URL
                article_id = article.url.split("/")[-1].split("-")[0]
                filename = f"{section}_{article_id}_{batch_num}_{i}.json"
                filepath = self.articles_dir / filename

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(article.to_dict(), f, ensure_ascii=False, indent=2)

                saved_count += 1
            except Exception as e:
                print(f"Ошибка сохранения статьи {article.url}: {e}")

        return saved_count

    def save_section_progress(
        self, section: str, urls: List[str], articles: List[Article]
    ):
        """Сохраняет прогресс по разделу"""
        progress_file = self.output_dir / f"{section}_progress.json"
        progress_data = {
            "section": section,
            "total_urls": len(urls),
            "parsed_articles": len(articles),
            "last_update": datetime.now().isoformat(),
            "urls": urls[:100],  # Сохраняем только первые 100 URL для отладки
        }

        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
