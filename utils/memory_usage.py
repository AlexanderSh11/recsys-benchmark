import gc
import os
import threading
import time
from contextlib import contextmanager

import psutil

SLEEP_TIME = 0.001

TO_MB = 1024**2


def get_rss_mb(process):
    return process.memory_info().rss / TO_MB


def monitor_memory(stop_event, process, result):
    while not stop_event.is_set():
        try:
            current_mem = get_rss_mb(process)
            result["peak_memory_usage"] = max(result["peak_memory_usage"], current_mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(SLEEP_TIME)


@contextmanager
def memory_usage(label="Блок"):
    """Контекстный менеджер для замера пикового объема используемой памяти блока кода"""
    result = {"peak_memory_usage": 0.0, "start_rss": 0.0, "delta_memory_usage": 0.0}
    gc.collect()

    process = psutil.Process(os.getpid())
    result["start_rss"] = get_rss_mb(process)
    result["peak_memory_usage"] = result["start_rss"]
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_memory, args=(stop_event, process, result), daemon=True
    )
    monitor_thread.start()

    try:
        yield result
    finally:
        stop_event.set()
        monitor_thread.join()
        result['delta_memory_usage'] = result["peak_memory_usage"] - result["start_rss"]

        print(f"[memory_usage] {label}: {result['delta_memory_usage']:.3f} MB")
