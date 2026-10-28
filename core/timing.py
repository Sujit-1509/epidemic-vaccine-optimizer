import time


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_milliseconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000
