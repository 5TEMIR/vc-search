import argparse
from vc_search.config import ScrapingConfig, DEFAULT_CONFIG
from vc_search.services.scraping_service import ScrapingService


def main():
    parser = argparse.ArgumentParser(description="VC.ru Scraper")
    parser.add_argument("--sections", nargs="+", help="Разделы для скрапинга")
    parser.add_argument("--articles", type=int, help="Количество статей на раздел")
    parser.add_argument("--delay", type=float, help="Задержка между запросами")
    parser.add_argument(
        "--no-headless", action="store_true", help="Запустить браузер в видимом режиме"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Размер батча для сохранения"
    )

    args = parser.parse_args()

    config = ScrapingConfig(
        sections=args.sections or DEFAULT_CONFIG.sections,
        articles_per_section=args.articles or DEFAULT_CONFIG.articles_per_section,
        delay=args.delay or DEFAULT_CONFIG.delay,
        headless=not args.no_headless,
    )

    print("Конфигурация скрапинга:")
    print(f"  Разделы: {', '.join(config.sections)}")
    print(f"  Статей на раздел: {config.articles_per_section}")
    print(f"  Задержка: {config.delay} сек")
    print(f"  Headless: {config.headless}")

    service = None
    try:
        service = ScrapingService(config)

        stats = service.scrape_all_sections()

        print(f"\n{'=' * 50}")
        print("ИТОГИ СКРАПИНГА:")
        print(f"  Всего собрано статей: {stats['total_articles']}")
        print(f"  Общее время сбора URL: {stats['total_url_time']:.1f} сек")
        print(f"  Общее время парсинга: {stats['total_parse_time']:.1f} сек")
        print(
            f"  Общее время: {stats['total_url_time'] + stats['total_parse_time']:.1f} сек"
        )

        print("\nСтатистика по разделам:")
        for section, section_stats in stats["section_stats"].items():
            print(
                f"  {section}: {section_stats['articles']} статей "
                f"({section_stats['url_time']:.1f}+{section_stats['parse_time']:.1f} сек)"
            )

    except Exception as e:
        print(f"Критическая ошибка: {e}")

    finally:
        if service:
            service.close()


if __name__ == "__main__":
    main()
