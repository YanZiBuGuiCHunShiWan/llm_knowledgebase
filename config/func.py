import time
import functools
from loguru import logger

def async_timer_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"异步函数 {func.__name__} 执行时间：{execution_time:.4f} 秒")
        return result
    return wrapper


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"函数 {func.__name__} 的执行时间为: {execution_time:.4f} 秒")
        return result
    return wrapper