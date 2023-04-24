from numba import cuda
import numpy as np
import math


@cuda.jit
def kernel_2d_array(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
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
        d0 = np.sqrt((Xrls[0] - xt) ** 2 + (
                Zrls[0] - zt) ** 2)  # дальность на траверсе

        ar = Vrls ** 2 / d0  # радиальное ускорение для компенсации МД и МЧ
        # непосредственно суммирование - КН с компенсацией МД и МЧ
        sum1 = 0
        for q in range(q1, q2):  # суммирование импульсов
            # дальность между точкой синтезирования и РЛС в q-ом
            # периоде повторения
            # disp(q)
            d = np.sqrt((xt - (Xrls[0] + Vxrls * q * Tr)) ** 2 +
                        (yt - (Yrls[0] + Vyrls * q * Tr)) ** 2 +
                        (zt - (Zrls[0] + Vzrls * q * Tr)) ** 2)
            # дробный номер отсчета, где находится сигнал, по быстрому времени
            ndr = (2 * d / c - t2 + T0) * Kss * Fs
            # целая и дробная часть индекса
            n = int(ndr) + 1
            drob = ndr % 1
            # линейная интерполяция отсчетов
            ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[
                n + 1, q] * drob

            # суммирование с учетом поворота фазы и взвешивания
            sum1 = sum1 + ut * WinSampl[q - q1 + 1] * np.exp(
                -4j * np.pi / lamda * ar / 2 * ((q - q0) * Tr) ** 2)
        Zxy1[nx, ny] = sum1


def big_cycle1(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, lamda):
    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array[blocks_per_grid, threads_per_block](X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
                                                        Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
                                                        Uout01ss, Fs, WinSampl, Zxy1, lamda)
    return Zxy1