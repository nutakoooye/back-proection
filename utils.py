import time


def time_of_function(function):
    def wrapped(*args):
        start_time = time.time()
        res = function(*args)
        time1 = time.time() - start_time
        start_time = time.time()
        res = function(*args)
        time2 = time.time() - start_time
        start_time = time.time()
        res = function(*args)
        time3 = time.time() - start_time
        print(function.__name__,"С компиляцией: ", time1, 'Без компиляции 1: ', time2, 'Без компиляции 2: ', time3)
        return res

    return wrapped
