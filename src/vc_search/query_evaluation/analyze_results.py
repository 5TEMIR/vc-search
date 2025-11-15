import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_marked_results(csv_file: str):
    """Анализирует размеченные результаты и вычисляет метрики"""

    df = pd.read_csv(csv_file)

    # Проверяем, размечены ли результаты
    if df["relevance"].isna().all():
        print("❌ Результаты еще не размечены!")
        print(
            "Заполните столбец 'relevance' значениями 1 (релевантно) или 0 (нерелевантно)"
        )
        return

    # Проверяем бинарные ли значения (0/1)
    unique_values = df["relevance"].unique()
    if set(unique_values) - {0, 1}:
        print(
            "⚠️  Внимание: значения релевантности должны быть 0 или 1 для бинарных метрик"
        )

    # Вычисляем метрики для каждого запроса
    metrics = {}

    for query in df["query"].unique():
        query_results = df[df["query"] == query].copy()
        query_results = query_results.sort_values("rank_position")

        relevant_count = query_results["relevance"].sum()
        total_count = len(query_results)

        # Вычисляем все метрики
        precision_5 = calculate_precision_at_k(query_results, k=5)
        precision_10 = calculate_precision_at_k(query_results, k=10)
        ndcg_5 = calculate_ndcg_at_k(query_results, k=5)
        ndcg_10 = calculate_ndcg_at_k(query_results, k=10)
        avg_precision = calculate_average_precision(query_results)
        mrr = calculate_mrr(query_results)

        metrics[query] = {
            "total_results": total_count,
            "relevant_results": int(relevant_count),
            "precision@5": precision_5,
            "precision@10": precision_10,
            "ndcg@5": ndcg_5,
            "ndcg@10": ndcg_10,
            "average_precision": avg_precision,
            "mrr": mrr,
        }

    # Выводим подробные результаты
    print("\n📊 ДЕТАЛЬНЫЕ МЕТРИКИ КАЧЕСТВА ПОИСКА")
    print("=" * 100)
    print(
        f"{'Запрос':<35} {'P@5':<6} {'P@10':<6} {'nDCG@5':<7} {'nDCG@10':<8} {'AvgP':<6} {'MRR':<6} {'Rel/Total':<12}"
    )
    print("-" * 100)

    for query, metric in metrics.items():
        print(
            f"{query:<35} {metric['precision@5']:<6.3f} {metric['precision@10']:<6.3f} "
            f"{metric['ndcg@5']:<7.3f} {metric['ndcg@10']:<8.3f} "
            f"{metric['average_precision']:<6.3f} {metric['mrr']:<6.3f} "
            f"{metric['relevant_results']}/{metric['total_results']:<12}"
        )

    # Общие метрики (средние по запросам)
    overall_metrics = calculate_overall_metrics(metrics)

    print("-" * 100)
    print(
        f"{'СРЕДНИЕ ПО ВСЕМ ЗАПРОСАМ':<35} {overall_metrics['mean_precision@5']:<6.3f} "
        f"{overall_metrics['mean_precision@10']:<6.3f} {overall_metrics['mean_ndcg@5']:<7.3f} "
        f"{overall_metrics['mean_ndcg@10']:<8.3f} {overall_metrics['mean_avg_precision']:<6.3f} "
        f"{overall_metrics['mean_mrr']:<6.3f} {'-':<12}"
    )

    # Дополнительная статистика
    print(f"\n📈 ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print(f"   Всего запросов: {len(metrics)}")
    print(f"   Всего результатов: {len(df)}")
    print(
        f"   Среднее количество результатов на запрос: {overall_metrics['mean_results_per_query']:.1f}"
    )
    print(
        f"   Среднее количество релевантных на запрос: {overall_metrics['mean_relevant_per_query']:.1f}"
    )

    return metrics, overall_metrics


def calculate_precision_at_k(results_df, k: int = 10) -> float:
    """Вычисляет Precision@K"""
    top_k = results_df.head(k)
    if len(top_k) == 0:
        return 0.0
    return top_k["relevance"].sum() / len(top_k)


def calculate_ndcg_at_k(results_df, k: int = 10) -> float:
    """Вычисляет nDCG@K (Normalized Discounted Cumulative Gain)"""
    top_k = results_df.head(k).copy()

    if len(top_k) == 0:
        return 0.0

    # Получаем релевантности для топ-K
    relevances = top_k["relevance"].values

    # Вычисляем DCG
    dcg = 0.0
    for i, rel in enumerate(relevances):
        # Для бинарной релевантности используем стандартную формулу DCG
        dcg += rel / np.log2(
            i + 2
        )  # i+2 потому что позиции начинаются с 1, а логарифм с 2

    # Вычисляем идеальный DCG (IDCG)
    # Сортируем релевантности по убыванию для идеального порядка
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevances):
        idcg += rel / np.log2(i + 2)

    # Избегаем деления на ноль
    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_average_precision(results_df) -> float:
    """Вычисляет Average Precision"""
    relevant_positions = []

    for i, (idx, row) in enumerate(results_df.iterrows()):
        if row["relevance"] == 1:
            relevant_positions.append(i + 1)  # +1 потому что позиции начинаются с 1

    if not relevant_positions:
        return 0.0

    # Вычисляем precision на каждой k-й релевантной позиции
    precisions = []
    for k, pos in enumerate(relevant_positions, 1):
        precision_at_k = k / pos
        precisions.append(precision_at_k)

    return sum(precisions) / len(precisions)


def calculate_mrr(results_df) -> float:
    """Вычисляет Mean Reciprocal Rank"""
    for i, (idx, row) in enumerate(results_df.iterrows()):
        if row["relevance"] == 1:
            return 1.0 / (i + 1)  # +1 потому что позиции начинаются с 1

    return 0.0


def calculate_overall_metrics(metrics: Dict) -> Dict:
    """Вычисляет общие метрики по всем запросам"""
    overall = {
        "mean_precision@5": np.mean([m["precision@5"] for m in metrics.values()]),
        "mean_precision@10": np.mean([m["precision@10"] for m in metrics.values()]),
        "mean_ndcg@5": np.mean([m["ndcg@5"] for m in metrics.values()]),
        "mean_ndcg@10": np.mean([m["ndcg@10"] for m in metrics.values()]),
        "mean_avg_precision": np.mean(
            [m["average_precision"] for m in metrics.values()]
        ),
        "mean_mrr": np.mean([m["mrr"] for m in metrics.values()]),
        "mean_results_per_query": np.mean(
            [m["total_results"] for m in metrics.values()]
        ),
        "mean_relevant_per_query": np.mean(
            [m["relevant_results"] for m in metrics.values()]
        ),
    }

    return overall


def save_metrics_to_csv(
    metrics: Dict, overall_metrics: Dict, output_file: str = "search_metrics.csv"
):
    """Сохраняет метрики в CSV файл для дальнейшего анализа"""

    # Подготовка данных для CSV
    rows = []

    # Метрики по каждому запросу
    for query, metric in metrics.items():
        row = {
            "query": query,
            "precision@5": metric["precision@5"],
            "precision@10": metric["precision@10"],
            "ndcg@5": metric["ndcg@5"],
            "ndcg@10": metric["ndcg@10"],
            "average_precision": metric["average_precision"],
            "mrr": metric["mrr"],
            "relevant_results": metric["relevant_results"],
            "total_results": metric["total_results"],
        }
        rows.append(row)

    # Общие метрики
    overall_row = {
        "query": "OVERALL_MEAN",
        "precision@5": overall_metrics["mean_precision@5"],
        "precision@10": overall_metrics["mean_precision@10"],
        "ndcg@5": overall_metrics["mean_ndcg@5"],
        "ndcg@10": overall_metrics["mean_ndcg@10"],
        "average_precision": overall_metrics["mean_avg_precision"],
        "mrr": overall_metrics["mean_mrr"],
        "relevant_results": "",
        "total_results": "",
    }
    rows.append(overall_row)

    # Сохраняем в CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 Метрики сохранены в: {output_file}")


def print_metric_explanations():
    """Выводит объяснения метрик"""
    print("\n📖 ОБЪЯСНЕНИЕ МЕТРИК:")
    print("   Precision@K  - Точность среди топ-K результатов (релевантные/все)")
    print("   nDCG@K       - Нормализованный дисконтированный кумулятивный выигрыш")
    print("   Average Prec - Средняя точность по всем релевантным документам")
    print(
        "   MRR          - Среднее обратное ранжирование (1/ранг первого релевантного)"
    )
    print("   Rel/Total    - Количество релевантных документов / всего документов")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Использование: python analyze_results.py <csv_file>")
        print("Пример: python analyze_results.py search_results_20241215_143022.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"❌ Файл {csv_file} не найден!")
        sys.exit(1)

    print(f"📊 Анализ файла: {csv_file}")
    metrics, overall_metrics = analyze_marked_results(csv_file)

    # Сохраняем метрики в отдельный файл
    if metrics:
        output_metrics_file = f"metrics_{Path(csv_file).stem}.csv"
        save_metrics_to_csv(metrics, overall_metrics, output_metrics_file)

        # Выводим объяснения метрик
        print_metric_explanations()
