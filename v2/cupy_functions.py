import cupy as cp
from v2.utils import time_of_function
import numpy as np


@time_of_function
def cuifft(a):
    mempool = cp.get_default_memory_pool()

    block_size = 1000  # Размер блока, который помещается в память видеокарты

    # Создание пустого массива для результата
    result = np.empty_like(a, dtype=np.complex64)

    # Вычисление FFT по блокам
    for i in range(0, a.shape[1], block_size):
        start = i
        end = min(i + block_size, a.shape[1])

        # Копирование блока данных на видеокарту
        # Вычисление FFT блока
        # Копирование результатов обратно в память CPU
        result[:, start:end] = cp.asnumpy(cp.fft.ifft(cp.asarray(a[:, start:end]), axis=0))

        mempool.free_all_blocks()
        cp.cuda.Device().synchronize()
    return result
