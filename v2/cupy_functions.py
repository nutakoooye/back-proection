import cupy as cp
from v2.utils import time_of_function
import numpy as np

# функция обертка над библиотечной чтобы разделять
# большой массив если он не помещается в видеопамять
@time_of_function
def cuifft(a):
    # переменная для доступа к управлению памятью видеокарты
    mempool = cp.get_default_memory_pool()

    # Размер блока, который помещается в память видеокарты
    # с помощью mempool.used_bytes() и mempool.total_bytes() можно получить
    # количество доступной памяти чтобы делить массив на блоки без еонстант
    block_size = 1000

    # Создание пустого массива для результата
    result = np.empty_like(a, dtype=np.complex64)

    # Вычисление FFT по блокам
    for i in range(0, a.shape[1], block_size):
        start = i
        end = min(i + block_size, a.shape[1])

        # Копирование блока данных на видеокарту - cp.asarray(a[:, start:end])
        # Вычисление FFT блока - cp.fft.ifft
        # Копирование результатов обратно в память CPU - cp.asnumpy
        result[:, start:end] = cp.asnumpy(cp.fft.ifft(cp.asarray(a[:, start:end]), axis=0))

        # освобождаем память видеокарты после каждого патча данных
        mempool.free_all_blocks()
        cp.cuda.Device().synchronize()
    return result
