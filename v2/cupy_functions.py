import cupy as cp
from v1.utils import time_of_function
import numpy as np


@time_of_function
def cuifft(a):
    block_size = 1000  # Размер блока, который помещается в память видеокарты

    # Создание пустого массива для результата
    result = np.empty_like(a, dtype=np.complex128)

    # Вычисление FFT по блокам
    for i in range(0, a.shape[1], block_size):
        start = i
        end = min(i + block_size, a.shape[1])

        # Копирование блока данных на видеокарту
        x_gpu = cp.asarray(a[:, start:end])

        # Вычисление FFT блока
        x_fft_gpu = cp.fft.ifft(x_gpu, axis=0)

        # Копирование результатов обратно в память CPU
        result[:, start:end] = cp.asnumpy(x_fft_gpu)

    return result
