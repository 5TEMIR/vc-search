from vc_search.search.elastic_client import VCElasticSearch
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_elasticsearch():
    """Настройка и проверка Elasticsearch"""
    es = VCElasticSearch()

    if not es.health_check():
        logger.error("Elasticsearch не доступен! Запустите:")
        logger.error("docker-compose -f docker-compose.elastic.yml up -d")
        return None

    logger.info("✅ Elasticsearch подключен успешно")

    if es.setup_index():
        logger.info("✅ Индекс настроен")
        return es
    else:
        logger.error("❌ Ошибка настройки индекса")
        return None


def index_articles(es: VCElasticSearch):
    """Индексация статей"""
    logger.info("Начинаем индексацию статей...")

    result = es.index_articles_from_json()

    if result["success"] > 0:
        logger.info(f"✅ Успешно проиндексировано {result['success']} статей")
        if result["errors"] > 0:
            logger.warning(f"⚠️  Ошибок при индексации: {result['errors']}")
    else:
        logger.error("❌ Не удалось проиндексировать статьи")

    return result


def show_stats(es):
    """Показать статистику индекса"""
    stats = es.get_index_stats()

    print("\n" + "=" * 50)
    print("📊 СТАТИСТИКА ПОИСКОВОГО ИНДЕКСА")
    print("=" * 50)
    print(f"📄 Всего документов: {stats.get('doc_count', 0)}")
    print(f"💾 Размер индекса: {stats.get('size_bytes', 0) / 1024 / 1024:.2f} MB")

    if stats.get("sections"):
        print("\n📂 Распределение по разделам:")
        for section, count in sorted(
            stats["sections"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   {section}: {count} статей")


def interactive_search(es):
    """Интерактивный поиск с поддержкой опечаток"""
    print("\n" + "=" * 50)
    print("🔍 ИНТЕРАКТИВНЫЙ ПОИСК С ОПЕЧАТОЧНИКОМ")
    print("=" * 50)
    print("Доступные команды:")
    print("  /stats - показать статистику")
    print("  /sections - показать разделы")
    print("  /fuzzy <запрос> - поиск с исправлением опечаток")
    print("  /smart <запрос> - умный поиск (автовыбор стратегии)")
    print("  /improved <запрос> - улучшенный поиск (адаптивные стратегии)")
    print("  /quit - выход")

    while True:
        try:
            user_input = input("\nВведите поисковый запрос или команду: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/stats":
                show_stats(es)
                continue
            elif user_input.lower() == "/sections":
                stats = es.get_index_stats()
                if stats.get("sections"):
                    print("\n📂 Разделы:")
                    for section in sorted(stats["sections"].keys()):
                        print(f"   {section}")
                continue
            elif user_input.lower().startswith("/fuzzy "):
                query = user_input[7:].strip()
                print(f"🔍 Fuzzy поиск: '{query}'...")
                results = es.search_with_fuzzy(query, limit=10)
            elif user_input.lower().startswith("/smart "):
                query = user_input[7:].strip()
                print(f"🤖 Умный поиск: '{query}'...")
                results = es.smart_search(query, limit=10)
            elif user_input.lower().startswith("/improved "):
                query = user_input[10:].strip()
                print(f"🚀 Улучшенный поиск: '{query}'...")
                results = es.improved_search(query, limit=10)
            else:
                query = user_input
                print(f"🔍 Обычный поиск: '{query}'...")
                results = es.search(query, limit=10)

            print(f"\nНайдено: {results['total']} результатов ({results['took']}ms)")

            if not results["results"]:
                print("❌ Ничего не найдено")
                return

            for i, hit in enumerate(results["results"], 1):
                print(f"\n{i}. [{hit['section']}] {hit['title']}")
                print(f"   👤 {hit['author']} | 📅 {hit.get('published_date', 'N/A')}")
                print(f"   📝 {hit['content_preview']}")
                print(f"   🔗 {hit['url']}")
                print(f"   📊 Score: {hit['score']:.3f} | Слов: {hit['word_count']}")

                if hit.get("highlights"):
                    print("   💡 Совпадения:")
                    for highlight in hit["highlights"][:2]:
                        print(f"      - {highlight}")

        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")


def main():
    print("vc.ru Search Engine - Elasticsearch")

    es = setup_elasticsearch()
    if not es:
        return

    # Проверяем есть ли уже данные
    stats = es.get_index_stats()
    if stats.get("doc_count", 0) == 0:
        print("\n📥 Индекс пустой, начинаем индексацию...")
        index_articles(es)
    else:
        print(f"\n📊 В индексе уже есть {stats['doc_count']} статей")

    # Показываем статистику
    show_stats(es)

    # Запускаем интерактивный поиск
    interactive_search(es)

    print("\n✅ Поисковый движок завершил работу")


if __name__ == "__main__":
    main()
