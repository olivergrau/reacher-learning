import time

def measure_time(func, *args, **kwargs):
    """Call func with args and kwargs, and return a tuple (result, elapsed_time)."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed
