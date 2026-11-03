import time
from contextlib import contextmanager


@contextmanager
def timer(label="Блок"):
    """Контекстный менеджер для замера времени блока кода"""
    time_start = time.time()
    result = {}
    try:
        yield result
    finally:
        result["elapsed_time"] = time.time() - time_start
        print(f"[timer] {label}: {result['elapsed_time']:.3f} сек.")
