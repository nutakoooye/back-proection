import numba as nb
import numpy as np


# @time_of_function
@nb.jit(nopython=True)
def add_zeros(Q, N, Kss, Y01):
    Y01ss = np.zeros((Kss * N, Q), np.complex128)
    for q in range(Q):
        for n in range(N):
            Y01ss[n, q] = Y01[n, q]
    return Y01ss


# @time_of_function
@nb.jit(nopython=True)
def add_zeros_2_channels(Q, N, Kss, Y01, Y02):
    Y01ss = np.zeros((Kss * N, Q), np.complex128)
    Y02ss = np.zeros((Kss * N, Q), np.complex128)
    for q in range(Q):
        for n in range(N):
            Y01ss[n, q] = Y01[n, q]
            Y02ss[n, q] = Y02[n, q]
    return Y01ss, Y02ss


# @time_of_function
@nb.jit(nopython=True)
def big_cycle1(Nxsint, Nysint, X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, lamda):
    for nx in range(Nxsint):
        for ny in range(Nysint):
            # координаты текущей точки наблюдения
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
        print(nx, '/', Nxsint)
    return Zxy1


# @time_of_function
@nb.jit(nopython=True)
def big_cycle2(Nxsint, Nysint, X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, L, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, Zxy2, lamda, Uout02ss):
    for nx in range(Nxsint):
        for ny in range(Nysint):
            # координаты текущей точки наблюдения
            xt = X1sint + nx * dxsint
            yt = Y1sint + ny * dysint
            zt = 0
            # определяем интервал индексов отсчетов для суммирования отсчетов исходя из
            # нахождения на траверсе для передающего канала
            q0 = int((yt - Yrls[0]) / (
                    Vyrls * Tr))  # индекс номера периода повторения для траверса
            q1 = q0 - int(Tsint / 2 / Tr)  # начальный индекс суммирования
            q2 = q0 + int(Tsint / 2 / Tr)  # конечный индекс суммирования
            d0 = np.sqrt((Xrls[0] - xt) ** 2 + (
                    Zrls[0] - zt) ** 2)  # дальность на траверсе

            ar = Vrls ** 2 / d0  # радиальное ускорение для компенсации МД и МЧ
            # нескомпенсированные скорости для приемных каналов
            Vr1 = L / 2 / d0 * Vrls
            Vr2 = -L / 2 / d0 * Vrls
            # непосредственно суммирование - КН с компенсацией МД и МЧ
            sum1 = 0
            sum2 = 0
            for q in range(q1, q2):
                d = np.sqrt((xt - (Xrls[0] + Vxrls * q * Tr)) ** 2 +
                            (yt - (Yrls[0] + Vyrls * q * Tr)) ** 2 +
                            (zt - (Zrls[0] + Vzrls * q * Tr)) ** 2)
                ndr = (2 * d / c - t2 + T0) * Kss * Fs  # дробный номер отсчета по быстрому времени
                n = int(ndr) + 1
                drob = ndr % 1
                ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
                # суммируем с учетом сдвига РЛИ по скорости
                sum1 = sum1 + ut * WinSampl[q - q1 + 1] * np.exp(
                    -1j * 4 * np.pi / lamda * ar / 2 * (
                                (q - q0) * Tr) ** 2) * np.exp(
                    1j * 2 * np.pi * Vr1 / lamda * (q - q0) * Tr)
                ut = Uout02ss[n, q] * (1 - drob) + Uout02ss[n + 1, q] * drob
                sum2 = sum2 + ut * WinSampl[q - q1 + 1] * np.exp(
                    -1j * 4 * np.pi / lamda * ar / 2 * (
                                (q - q0) * Tr) ** 2) * np.exp(
                    1j * 2 * np.pi * Vr2 / lamda * (q - q0) * Tr)
            Zxy1[nx, ny] = sum1
            Zxy2[nx, ny] = sum2
        print(nx, '/', Nxsint)
    return Zxy1, Zxy2