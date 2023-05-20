import time


def time_of_function_compile(function):
    def wrapped(*args, **kwargs):
        start_time = time.time()
        res = function(*args,**kwargs)
        time1 = time.time() - start_time
        start_time = time.time()
        res = function(*args, **kwargs)
        time2 = time.time() - start_time
        print(function.__name__,"С компиляцией: ", time1, 'Без компиляции 1: ', time2)
        return res

    return wrapped

def time_of_function(function):
    def wrapped(*args, **kwargs):
        start_time = time.time()
        res = function(*args,**kwargs)
        time1 = time.time() - start_time
        print(function.__name__,"Выполнилась за", time1)
        return res

    return wrapped

