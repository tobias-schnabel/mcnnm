# timer.py

import time
from functools import wraps

def timer(func):
    """
    A decorator that times the execution of a function.

    Args:
        func: The function to be timed.

    Returns:
        A wrapped function that prints the execution time and returns the original function's result.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper