from vc_search.search.elastic_client import VCElasticSearch
import logging

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
    """Интерактивный поиск"""
    print("\n" + "=" * 50)
    print("🔍 ИНТЕРАКТИВНЫЙ ПОИСК")
    print("=" * 50)
    print("Доступные команды:")
    print("  /stats - показать статистику")
    print("  /sections - показать разделы")
    print("  /model <запрос> - поиск с использованием модели")
    print("  /delete-index - удаление индекса")
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
            elif user_input.lower().startswith("/model "):
                query = user_input[7:].strip()
                print(f"🤖 Поиск с использованием модели: '{query}'...")
                model_path = "data/models/logistic_regression_0.2.pkl"
                results = es.search_with_relevance_model(
                    query,
                    limit=10,
                    model_path=model_path,
                    threshold=0.4,
                )
            elif user_input.lower().startswith("/delete-index"):
                es.delete_index()
                print("Индекс удален")
                break
            else:
                query = user_input
                print(f"🔍 Поиск: '{query}'...")
                results = es.search(query, limit=10)

            print(f"\nНайдено: {results['total']} результатов ({results['took']}ms)")

            if not results["results"]:
                print("❌ Ничего не найдено")
                continue

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

                if hit.get("relevance_probability") is not None:
                    relevance_icon = "✅" if hit["relevance_prediction"] == 1 else "❌"
                    print(
                        f"   🤖 {relevance_icon} Модель: {hit['relevance_probability']:.3f} "
                        f"(комбинированный: {hit.get('combined_score', 0):.3f})"
                    )

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

    stats = es.get_index_stats()
    if stats.get("doc_count", 0) == 0:
        print("\n📥 Индекс пустой, начинаем индексацию...")
        index_articles(es)
    else:
        print(f"\n📊 В индексе уже есть {stats['doc_count']} статей")

    show_stats(es)

    interactive_search(es)

    print("\n✅ Поисковый движок завершил работу")


if __name__ == "__main__":
    main()
