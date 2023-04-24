from numba import cuda
import numba as nb
import numpy as np
import math
from utils import time_of_function


@cuda.jit
def kernel_2d_array_1(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
           Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
           Uout01ss, Fs, WinSampl, Zxy1, lamda):
    nx, ny = cuda.grid(2)
    # координаты текущей точки наблюдения
    if nx < Zxy1.shape[0] and ny < Zxy1.shape[1]:
        xt = X1sint + nx * dxsint
        yt = Y1sint + ny * dysint
        zt = 0
        # определяем интервал индексов отсчетов для суммирования отсчетов исходя из
        # нахождения на траверсе для передающего канала
        q0 = int((yt - Yrls[0]) / (Vyrls * Tr))  # индекс номера периода повторения для траверса
        q1 = q0 - int(Tsint / 2 / Tr)  # начальный индекс суммирования
        q2 = q0 + int(Tsint / 2 / Tr)  # конечный индекс суммирования
        d0 = ((Xrls[0] - xt) ** 2 + (
                Zrls[0] - zt) ** 2) ** 0.5  # дальность на траверсе

        ar = Vrls ** 2 / d0  # радиальное ускорение для компенсации МД и МЧ
        # непосредственно суммирование - КН с компенсацией МД и МЧ
        sum1 = 0
        for q in range(q1, q2):  # суммирование импульсов
            # дальность между точкой синтезирования и РЛС в q-ом
            # периоде повторения
            # disp(q)
            d = ((xt - (Xrls[0] + Vxrls * q * Tr)) ** 2 +
                        (yt - (Yrls[0] + Vyrls * q * Tr)) ** 2 +
                        (zt - (Zrls[0] + Vzrls * q * Tr)) ** 2) ** 0.5
            # дробный номер отсчета, где находится сигнал, по быстрому времени
            ndr = (2 * d / c - t2 + T0) * Kss * Fs
            # целая и дробная часть индекса
            n = int(ndr) + 1
            drob = ndr % 1
            # линейная интерполяция отсчетов
            ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[
                n + 1, q] * drob

            # суммирование с учетом поворота фазы и взвешивания
            sum1 = sum1 + ut * WinSampl[q - q1 + 1] * 2.71828183 ** (
                -4j * np.pi / lamda * ar / 2 * ((q - q0) * Tr) ** 2)
        Zxy1[nx, ny] = sum1


@cuda.jit
def kernel_2d_array_2(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, L, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, Zxy2, lamda, Uout02ss):
    e = 2.71828183
    nx, ny = cuda.grid(2)
            # координаты текущей точки наблюдения
    if nx < Zxy1.shape[0] and ny < Zxy1.shape[1]:
        xt = X1sint + nx * dxsint
        yt = Y1sint + ny * dysint
        zt = 0
        # определяем интервал индексов отсчетов для суммирования отсчетов исходя из
        # нахождения на траверсе для передающего канала
        q0 = int((yt - Yrls[0]) / (
                Vyrls * Tr))  # индекс номера периода повторения для траверса
        q1 = q0 - int(Tsint / 2 / Tr)  # начальный индекс суммирования
        q2 = q0 + int(Tsint / 2 / Tr)  # конечный индекс суммирования
        d0 = ((Xrls[0] - xt) ** 2 + (
                Zrls[0] - zt) ** 2) ** 0.5 # дальность на траверсе

        ar = Vrls ** 2 / d0  # радиальное ускорение для компенсации МД и МЧ
        # нескомпенсированные скорости для приемных каналов
        Vr1 = L / 2 / d0 * Vrls
        Vr2 = -L / 2 / d0 * Vrls
        # непосредственно суммирование - КН с компенсацией МД и МЧ
        sum1 = 0
        sum2 = 0
        for q in range(q1, q2):
            d = ((xt - (Xrls[0] + Vxrls * q * Tr)) ** 2 +
                 (yt - (Yrls[0] + Vyrls * q * Tr)) ** 2 +
                 (zt - (Zrls[0] + Vzrls * q * Tr)) ** 2) ** 0.5
            ndr = (2 * d / c - t2 + T0) * Kss * Fs  # дробный номер отсчета по быстрому времени
            n = int(ndr) + 1
            drob = ndr % 1
            ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
            # суммируем с учетом сдвига РЛИ по скорости
            sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (
                -1j * 4 * np.pi / lamda * ar / 2 * (
                            (q - q0) * Tr) ** 2) * e ** (
                1j * 2 * np.pi * Vr1 / lamda * (q - q0) * Tr)
            ut = Uout02ss[n, q] * (1 - drob) + Uout02ss[n + 1, q] * drob
            sum2 = sum2 + ut * WinSampl[q - q1 + 1] * e ** (
                -1j * 4 * np.pi / lamda * ar / 2 * (
                            (q - q0) * Tr) ** 2) * e ** (
                1j * 2 * np.pi * Vr2 / lamda * (q - q0) * Tr)
        Zxy1[nx, ny] = sum1
        Zxy2[nx, ny] = sum2


@time_of_function
def big_cycle1(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, lamda):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array_1[blocks_per_grid, threads_per_block](X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
                                                        Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
                                                        Uout01ss, Fs, WinSampl, Zxy1, lamda)
    return Zxy1

@time_of_function
def big_cycle2(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, L, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, Zxy2, lamda, Uout02ss):
    threads_per_block = (8, 8)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array_2[blocks_per_grid, threads_per_block](X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, L, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, Zxy2, lamda, Uout02ss)
    return Zxy1, Zxy2