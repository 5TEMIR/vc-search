import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from vc_search.search.elastic_client import VCElasticSearch


class QueryEvaluator:
    def __init__(self, es_client: VCElasticSearch):
        self.es = es_client
        self.queries = [
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

    def clean_text(self, text: str) -> str:
        """Очищает текст от переносов строк и лишних пробелов"""
        if not text:
            return ""
        # Заменяем переносы строк пробелами и убираем лишние пробелы
        return " ".join(text.replace("\n", " ").replace("\r", " ").split())

    def execute_queries(self, results_per_query: int = 10) -> List[Dict[str, Any]]:
        """Выполняет все запросы и собирает результаты"""
        all_results = []

        for query in self.queries:
            print(f"🔍 Выполняю запрос: '{query}'")

            search_results = self.es.improved_search(query, limit=results_per_query)

            for i, result in enumerate(search_results["results"]):
                # Обрабатываем highlights
                highlights_text = ""
                if result.get("highlights"):
                    # Объединяем все highlights в один текст
                    cleaned_highlights = [
                        self.clean_text(h) for h in result["highlights"]
                    ]
                    highlights_text = " | ".join(cleaned_highlights)

                record = {
                    "relevance": "",  # Пустой столбец для разметки релевантности
                    "query": query,
                    "title": self.clean_text(result.get("title", "")),
                    "highlight": highlights_text,
                    "url": result.get("url", ""),
                    "author": result.get("author", ""),
                    "section": result.get("section", ""),
                    "score": result.get("score", 0),
                    "rank_position": i + 1,  # Позиция в выдаче
                }
                all_results.append(record)

            print(
                f"✅ Найдено {len(search_results['results'])} результатов для '{query}'"
            )

        return all_results

    def save_to_csv(self, results: List[Dict[str, Any]], output_file: str = None):
        """Сохраняет результаты в CSV файл"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"search_results_{timestamp}.csv"

        # Создаем директорию если нужно
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "relevance",
            "query",
            "title",
            "highlight",
            "url",
            "author",
            "section",
            "score",
            "rank_position",
        ]

        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(result)

        print(f"💾 Результаты сохранены в: {output_path}")
        print(f"📊 Всего записей: {len(results)}")

        return output_path


def main():
    """Основная функция для выполнения оценки запросов"""
    print("🎯 Оценка качества поискового движка vc.ru")
    print("=" * 50)

    # Подключаемся к Elasticsearch
    es = VCElasticSearch()

    if not es.health_check():
        print("❌ Elasticsearch не доступен!")
        print("Запустите: docker-compose -f docker-compose.elastic.yml up -d")
        return

    print("✅ Подключение к Elasticsearch установлено")

    # Создаем evaluator и выполняем запросы
    evaluator = QueryEvaluator(es)
    results = evaluator.execute_queries(results_per_query=10)

    if not results:
        print("❌ Не удалось получить результаты поиска")
        return

    # Сохраняем в CSV
    output_file = evaluator.save_to_csv(results)

    # Показываем статистику
    print("\n📈 Статистика оценки:")
    print(f"   Всего запросов: {len(evaluator.queries)}")
    print(f"   Всего результатов: {len(results)}")

    # Статистика по запросам
    queries_stats = {}
    for result in results:
        query = result["query"]
        if query not in queries_stats:
            queries_stats[query] = 0
        queries_stats[query] += 1

    print("\n📊 Результаты по запросам:")
    for query, count in queries_stats.items():
        print(f"   '{query}': {count} результатов")

    print(f"\n🎉 Файл {output_file} готов для разметки релевантности!")
    print(
        "   Заполните столбец 'relevance' значениями 1 (релевантно) или 0 (нерелевантно)"
    )


if __name__ == "__main__":
    main()
