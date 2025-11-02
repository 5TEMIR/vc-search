from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Tuple


def process_parallel(
    items: List[Any], process_func: Callable, max_workers: int = 4, timeout: int = 2
) -> Tuple[List[Any], List[Any]]:
    """
    Обрабатывает элементы параллельно

    Returns:
        Tuple[successful_items, failed_items]
    """
    successful = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(process_func, item): item for item in items}

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result(timeout=timeout)
                if result:
                    successful.append(result)
                else:
                    failed.append(item)
            except Exception as e:
                failed.append(item)

    return successful, failed
