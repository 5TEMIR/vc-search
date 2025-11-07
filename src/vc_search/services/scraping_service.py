import time
from typing import List, Dict, Tuple
from ..scraper.vc_scraper import VCScraper
from ..storage.json_storage import JSONStorage
from ..models.article import Article
from ..config import ScrapingConfig


class ScrapingService:
    def __init__(self, config: ScrapingConfig):
        self.sections = config.sections
        self.articles_per_section = config.articles_per_section
        self.delay = config.delay
        self.batch_size = config.batch_size
        self.scraper = VCScraper(delay=config.delay, headless=config.headless)
        self.storage = JSONStorage()

    def scrape_section(self, section: str) -> Tuple[int, float, float]:
        """Скрапит один раздел"""
        print(f"\n🎯 Начинаем раздел: {section.upper()}")

        # Сбор URL
        start_time = time.time()
        urls = self.scraper.get_article_urls_from_section(
            section, max_articles=self.articles_per_section
        )
        url_collection_time = time.time() - start_time

        print(
            f"📄 Найдено {len(urls)} уникальных статей за {url_collection_time:.1f} сек"
        )

        if not urls:
            print(f"⏭️ Пропускаем раздел {section} - нет статей")
            return 0, url_collection_time, 0

        # Парсинг статей батчами
        print(f"🔍 Парсинг {len(urls)} статей...")
        start_parse_time = time.time()
        total_parsed = 0

        for batch_start in range(0, len(urls), self.batch_size):
            batch_end = batch_start + self.batch_size
            batch_urls = urls[batch_start:batch_end]

            batch_articles = []
            for url in batch_urls:
                article = self.scraper.parse_article(url, section)
                if article:
                    batch_articles.append(article)

            # Сохраняем батч
            saved_count = self.storage.save_articles_batch(batch_articles)
            total_parsed += saved_count

            print(
                f"📦 Батч {batch_start // self.batch_size + 1}: "
                f"спарсено {len(batch_articles)}, сохранено {saved_count}"
            )
            print(f"📊 Прогресс: {min(batch_end, len(urls))}/{len(urls)} статей")

        parse_time = time.time() - start_parse_time

        print(f"✅ Раздел {section.upper()} завершен:")
        print(f"   Статей: {total_parsed}")
        print(f"   Время сбора URL: {url_collection_time:.1f} сек")
        print(f"   Время парсинга: {parse_time:.1f} сек")

        return total_parsed, url_collection_time, parse_time

    def scrape_all_sections(self) -> Dict:
        """Скрапит все разделы"""
        initial_count = self.storage.get_article_count()
        print(f"📊 Начальное количество статей: {initial_count}")

        total_stats = {
            "total_articles": 0,
            "total_url_time": 0,
            "total_parse_time": 0,
            "section_stats": {},
        }

        for section in self.sections:
            try:
                articles_count, url_time, parse_time = self.scrape_section(section)

                total_stats["total_articles"] += articles_count
                total_stats["total_url_time"] += url_time
                total_stats["total_parse_time"] += parse_time
                total_stats["section_stats"][section] = {
                    "articles": articles_count,
                    "url_time": url_time,
                    "parse_time": parse_time,
                }

            except Exception as e:
                print(f"❌ Ошибка в разделе {section}: {e}")
                continue

        final_count = self.storage.get_article_count()
        print(f"📊 Финальное количество статей: {final_count}")
        print(f"📈 Добавлено новых статей: {final_count - initial_count}")

        return total_stats

    def close(self):
        """Закрывает ресурсы"""
        if self.scraper:
            self.scraper.close()
