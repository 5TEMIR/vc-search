import time
from typing import List, Dict, Tuple
from ..scraper.vc_scraper import VCScraper
from ..storage.json_storage import JSONStorage
from ..utils.parallel import process_parallel
from ..config import ScrapingConfig


class ScrapingService:
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.scraper = VCScraper(delay=config.delay, headless=config.headless)
        self.storage = JSONStorage()

    def scrape_section(self, section: str) -> Tuple[int, float, float]:
        """Скрапит один раздел и возвращает (количество_статей, время_сбора_url, время_парсинга)"""
        print(f"\nНачинаем раздел: {section.upper()}")

        # Сбор URL
        start_time = time.time()
        urls = self.scraper.get_article_urls_from_section(
            section, max_articles=self.config.articles_per_section
        )
        url_collection_time = time.time() - start_time

        print(f"Найдено {len(urls)} уникальных статей за {url_collection_time:.1f} сек")

        if not urls:
            print(f"Пропускаем раздел {section} - нет статей")
            return 0, url_collection_time, 0

        # Парсинг статей
        print(f"Парсинг {len(urls)} статей...")
        start_parse_time = time.time()

        articles = self._parse_articles_parallel(urls, section)
        parse_time = time.time() - start_parse_time

        # Сохранение результатов
        if articles:
            self._save_articles_batched(articles, section)

        print(f"Раздел {section.upper()} завершен:")
        print(f"Статей: {len(articles)}")
        print(f"Время сбора URL: {url_collection_time:.1f} сек")
        print(f"Время парсинга: {parse_time:.1f} сек")

        return len(articles), url_collection_time, parse_time

    def _parse_articles_parallel(self, urls: List[str], section: str) -> List:
        """Параллельный парсинг статей"""

        def parse_article_wrapper(url):
            return self.scraper.parse_article(url, section)

        articles, failed_urls = process_parallel(
            items=urls,
            process_func=parse_article_wrapper,
            max_workers=self.config.max_workers,
            timeout=self.config.timeout,
        )

        if failed_urls:
            print(f"Не удалось спарсить {len(failed_urls)} статей")

        return articles

    def _save_articles_batched(self, articles: List, section: str):
        """Сохранение статей пачками"""
        for batch_num, batch_start in enumerate(
            range(0, len(articles), self.config.batch_size), 1
        ):
            batch_end = batch_start + self.config.batch_size
            batch = articles[batch_start:batch_end]

            saved_count = self.storage.save_articles_batch(batch, batch_num, section)
            print(f"Пачка {batch_num}: сохранено {saved_count} статей")

            print(f"Прогресс: {min(batch_end, len(articles))}/{len(articles)} статей")

    def scrape_all_sections(self) -> Dict:
        """Скрапит все разделы и возвращает статистику"""
        total_stats = {
            "total_articles": 0,
            "total_url_time": 0,
            "total_parse_time": 0,
            "section_stats": {},
        }

        for section in self.config.sections:
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
                print(f"Ошибка в разделе {section}: {e}")
                continue

        return total_stats

    def close(self):
        """Закрывает ресурсы"""
        if self.scraper:
            self.scraper.close()
