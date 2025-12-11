import csv
import time
from datetime import datetime
from pathlib import Path
import logging
import sys

# Добавляем путь к src для импорта
sys.path.append("src")

from vc_search.search.elastic_client import VCElasticSearch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_elasticsearch():
    """Настройка подключения к Elasticsearch"""
    es = VCElasticSearch()

    if not es.health_check():
        logger.error("❌ Elasticsearch недоступен!")
        logger.error("Запустите: docker-compose -f docker-compose.elastic.yml up -d")
        return None

    logger.info("✅ Elasticsearch подключен")
    return es


def load_relevance_model(es, model_path):
    """Загрузка модели релевантности"""
    if not Path(model_path).exists():
        logger.error(f"❌ Файл модели не найден: {model_path}")
        return False

    success = es.load_relevance_model(model_path)
    if success:
        logger.info(f"✅ Модель загружена: {model_path}")
    else:
        logger.error("❌ Не удалось загрузить модель")

    return success


def search_queries_with_model(es, queries, model_path, limit_per_query=10):
    """Выполняет поиск по списку запросов с использованием модели"""
    results = []

    # Загружаем модель
    if not load_relevance_model(es, model_path):
        return results

    total_queries = len(queries)

    for idx, query in enumerate(queries, 1):
        logger.info(f"🔍 [{idx}/{total_queries}] Поиск: '{query}'")

        try:
            # Выполняем поиск с моделью
            search_results = es.search_with_relevance_model(
                query=query,
                limit=limit_per_query,
                model_path=model_path,
                use_full_content=True,
                threshold=0.3,  # Порог вероятности релевантности
            )

            if search_results.get("model_used", False):
                logger.info(
                    f"   🤖 Модель применена, найдено: {len(search_results['results'])} результатов"
                )
            else:
                logger.warning(
                    f"   ⚠️  Модель не использована, результаты от Elasticsearch"
                )

            # Добавляем результаты
            for hit in search_results.get("results", []):
                result = {
                    "query": query,
                    "title": hit.get("title", ""),
                    "url": hit.get("url", ""),
                    "relevance_score": "",  # Оставляем пустым для ручной разметки
                    "relevance_probability": hit.get("relevance_probability", 0),
                    "relevance_prediction": hit.get("relevance_prediction", 0),
                    "elastic_score": hit.get("score", 0),
                    "combined_score": hit.get("combined_score", 0),
                    "section": hit.get("section", ""),
                    "author": hit.get("author", ""),
                    "published_date": hit.get("published_date", ""),
                    "word_count": hit.get("word_count", 0),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)

            # Краткая статистика для этого запроса
            if search_results.get("results"):
                first_result = search_results["results"][0]
                logger.info(
                    f"   📊 Топ результат: {first_result.get('title', '')[:50]}..."
                )
                logger.info(
                    f"   📈 Вероятность релевантности: {first_result.get('relevance_probability', 0):.3f}"
                )

            # Пауза между запросами, чтобы не перегружать Elasticsearch
            if idx < total_queries:
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"   ❌ Ошибка при поиске '{query}': {e}")
            continue

    return results


def save_results_to_csv(results, output_file):
    """Сохраняет результаты в CSV файл"""
    if not results:
        logger.warning("Нет результатов для сохранения")
        return False

    # Основные колонки (как требуется)
    fieldnames = ["relevance_score", "query", "title", "url"]

    # Дополнительные колонки для анализа
    extra_fieldnames = [
        "relevance_probability",
        "relevance_prediction",
        "elastic_score",
        "combined_score",
        "section",
        "author",
        "published_date",
        "word_count",
        "timestamp",
    ]

    # Создаем директорию, если её нет
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames + extra_fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(result)

        logger.info(f"✅ Результаты сохранены в {output_file}")
        logger.info(f"📊 Всего сохранено {len(results)} результатов")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении в CSV: {e}")
        return False


def create_summary_report(results, queries):
    """Создает сводный отчет по результатам поиска"""
    if not results:
        return

    print("\n" + "=" * 60)
    print("📊 СВОДНЫЙ ОТЧЕТ ПО РЕЗУЛЬТАТАМ ПОИСКА")
    print("=" * 60)

    # Статистика по запросам
    query_stats = {}
    for query in queries:
        query_results = [r for r in results if r["query"] == query]
        query_stats[query] = len(query_results)

    print(f"Всего запросов: {len(queries)}")
    print(f"Всего результатов: {len(results)}")
    print(f"Среднее результатов на запрос: {len(results) / len(queries):.1f}")

    print("\n📈 Статистика по запросам:")
    for query, count in query_stats.items():
        print(f"   {query}: {count} результатов")

    # Статистика по модели
    if results and "relevance_probability" in results[0]:
        avg_prob = sum(r["relevance_probability"] for r in results) / len(results)
        relevant_count = sum(
            1 for r in results if r.get("relevance_prediction", 0) == 1
        )

        print(f"\n🤖 Статистика модели релевантности:")
        print(f"   Средняя вероятность: {avg_prob:.3f}")
        print(f"   Предсказано релевантными: {relevant_count}/{len(results)}")
        print(f"   Доля релевантных: {relevant_count / len(results) * 100:.1f}%")


def main():
    """Основная функция скрипта"""
    print("🔍 ПОИСК С ИСПОЛЬЗОВАНИЕМ МОДЕЛИ РЕЛЕВАНТНОСТИ")
    print("=" * 60)

    # Список тестовых запросов
    test_queries = [
        "ИИ в трейдинге",
        "новый VR шлем Valve",
        "Telegram обновил дизайн iOS",
        "крипторынок восстановление 2025",
        "создание длинных видео Sora",
        "блокировка SIM карт роуминг",
        "нейросети для дизайна интерьера",
        "зарплатные ожидания зумеров",
        "СберМобайл подписка Литрес",
        "будущее бизнеса эмпатия технологии",
    ]

    # Пути к файлам
    model_path = "../../../data/models/logistic_regression_0.2.pkl"
    output_csv = f"./model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Настройка Elasticsearch
    es = setup_elasticsearch()
    if not es:
        return

    # Проверяем статистику индекса
    stats = es.get_index_stats()
    print(f"\n📊 Статистика индекса:")
    print(f"   Документов: {stats.get('doc_count', 0)}")
    print(f"   Разделов: {len(stats.get('sections', {}))}")

    if stats.get("doc_count", 0) == 0:
        logger.warning("⚠️  Индекс пустой! Нужно сначала проиндексировать статьи.")
        return

    # Выполняем поиск
    print(f"\n🔍 Выполняем поиск по {len(test_queries)} запросам...")
    results = search_queries_with_model(
        es=es, queries=test_queries, model_path=model_path, limit_per_query=10
    )

    # Сохраняем результаты
    if results:
        save_results_to_csv(results, output_csv)
        create_summary_report(results, test_queries)

        # Показываем примеры результатов
        print(f"\n👁️  ПРИМЕРЫ РЕЗУЛЬТАТОВ:")
        for query in test_queries[:3]:  # Показываем для первых 3 запросов
            query_results = [r for r in results if r["query"] == query][:2]
            if query_results:
                print(f"\nЗапрос: '{query}'")
                for i, result in enumerate(query_results, 1):
                    print(f"  {i}. {result['title'][:70]}...")
                    print(f"     URL: {result['url'][:70]}...")
                    print(f"     Вероятность: {result['relevance_probability']:.3f}")
    else:
        logger.error("❌ Не удалось получить результаты поиска")

    print(f"\n✅ Скрипт завершен")
    print(f"📁 Результаты сохранены в: {output_csv}")


if __name__ == "__main__":
    main()
