import math
import numpy as np
from numba import cuda
from v1.utils import time_of_function, time_of_function_compile


@cuda.jit
def kernel_2d_array_1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
                      Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, Vrsa, tauRli, Inabl, qq, tt,
                      Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0,
                      sumtt, sumtt2, sumtt3, sumtt4, sumtt5, sumtt6, q1, q2, Tst):
    nx, ny = cuda.grid(2)
    # координаты текущей точки наблюдения
    xt = (-(Nxsint - 1) / 2 + nx) * dxsint
    yt = (-(Nysint - 1) / 2 + ny) * dysint
    zt = 0
    # подготовка исходных данных для вызова OrbitRange
    fizt = fizt0 + yt / Rz
    betazt = betazt0 + xt / Rz / math.cos(fizt0)
    Hzt = zt
    fiztSh = 0
    betaztSh = 0
    # номер первого периода на интервале синтезирования
    qst = int(tauRli / Tr) + 1
    # аппроксимируем суммарную дальность фазовый центр антенны на передачу -
    # земная точка - фазовый центр приемного канала на интервале
    # синтезирования полиномом третьей степени по пяти точкам
    r_Rch_zt = np.zeros((1, Inabl))
    rzt = np.zeros((3, 1))
    RR = np.zeros(5)
    for k in range(5):
        # получение координат фазового центра передающей антенны в НГцСК
        rrsa = np.array(
            [[XYZ_rsa_ts[qq[k], 0]], [XYZ_rsa_ts[qq[k], 1]], [XYZ_rsa_ts[qq[k], 2]]]) * dxConsort
        # получение координат фазового центра приемного канала в НГцСК
        rRch = np.array(
            [[XYZ_rsa_ts[qq[k], 6]], [XYZ_rsa_ts[qq[k], 7]], [XYZ_rsa_ts[qq[k], 8]]]) * dxConsort

            # вычисление координат земной точки в НГцСК
        rzt[0, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[k]) * np.cos(
            2 * np.pi / Tz * (Tst + tt[k]) + betazt + betaztSh * tt[k])
        rzt[1, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[k]) * np.sin(
            2 * np.pi / Tz * (Tst + tt[k]) + betazt + betaztSh * tt[k])
        rzt[2, 0] = (Rz + Hzt) * np.sin(fizt + fiztSh * tt[k])
        RR[k] = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch - rzt)
    # аппроксимация дальности полиномом третьей степени
    B0 = np.sum(RR)
    B1 = np.sum(RR * tt)
    B2 = np.sum(RR * (tt ** 2))
    B3 = np.sum(RR * (tt ** 3))
    A = np.array([[5, sumtt, sumtt2, sumtt3],
                  [sumtt, sumtt2, sumtt3, sumtt4],
                  [sumtt2, sumtt3, sumtt4, sumtt5],
                  [sumtt3, sumtt4, sumtt5, sumtt6]])
    B = np.array([[B0], [B1], [B2], [B3]])
    pr = np.linalg.inv(A).dot(B)
    pr1 = np.flip(pr.T)
    # вычисляем все дальности на интервале наблюдения
    for i in range(Inabl):
        t1 = i * Tr
        b = np.array([[t1 ** 3], [t1 ** 2], [t1], [1.0]])
        r_Rch_zt[0, i] = pr1[0, 0] * b[0, 0] + pr1[0, 1] * b[1, 0] + \
                                pr1[0, 2] * b[2, 0] + pr1[0, 3] * b[3, 0]

    # дальность на траверсе
    d0 = r_Rch_zt[0, 0]
    # ar=Vrsa^2/d0  # радиальное ускорение для компенсации МД и МЧ
    # нескомпенсированные скорости для приемных каналов
    Vr1 = (Lrch / 2) / d0 * Vrsa
    # непосредственно суммирование - КН с компенсацией МД и МЧ
    sum1 = 0
    # дальность в момент начала синтезирования для первого канала
    for q in range(q1 - 1, q2):
        d = r_Rch_zt[0, q - q1 + 1]
        # дробный номер отсчета по быстрому времени
        ndr = (d / speedOfL - t_r_w + T0) * Kss * Fs - 1
        n = int(ndr)
        drob = ndr % 1
        ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
        fiq = 2 * math.pi / lamda * (r_Rch_zt[0, q - q1 + 1] - q1)
        # суммируем с учетом сдвига РЛИ по скорости
        sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq) * e ** (
                1j * 2 * np.pi * Vr1 / lamda * (q - q1 + 1) * Tr)

    Zxy1[nx, ny] = sum1


@cuda.jit
def kernel_2d_array_2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
                      Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, Vrsa, tauRli, Inabl, qq, tt,
                      Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0,
                      sumtt, sumtt2, sumtt3, sumtt4, sumtt5, sumtt6, q1, q2, Tst):
    nx, ny = cuda.grid(2)
    # координаты текущей точки наблюдения
    xt = (-(Nxsint - 1) / 2 + nx) * dxsint
    yt = (-(Nysint - 1) / 2 + ny) * dysint
    zt = 0
    # подготовка исходных данных для вызова OrbitRange
    fizt = fizt0 + yt / Rz
    betazt = betazt0 + xt / Rz / math.cos(fizt0)
    Hzt = zt
    fiztSh = 0
    betaztSh = 0
    # номер первого периода на интервале синтезирования
    qst = int(tauRli / Tr) + 1
    # аппроксимируем суммарную дальность фазовый центр антенны на передачу -
    # земная точка - фазовый центр приемного канала на интервале
    # синтезирования полиномом третьей степени по пяти точкам
    r_Rch_zt = np.zeros((2, Inabl))
    rzt = np.zeros((3, 1))
    RR = np.zeros(5)
    for irch in range(1, 3):
        for k in range(5):
            # получение координат фазового центра передающей антенны в НГцСК
            rrsa = np.array(
                [[XYZ_rsa_ts[qq[k], 0]], [XYZ_rsa_ts[qq[k], 1]], [XYZ_rsa_ts[qq[k], 2]]]) * dxConsort
            # получение координат фазового центра приемного канала в НГцСК
            if irch == 1:
                rRch = np.array(
                    [[XYZ_rsa_ts[qq[k], 6]], [XYZ_rsa_ts[qq[k], 7]], [XYZ_rsa_ts[qq[k], 8]]]) * dxConsort
            if irch == 2:
                rRch = np.array(
                    [[XYZ_rsa_ts[qq[k], 9]], [XYZ_rsa_ts[qq[k], 10]], [XYZ_rsa_ts[qq[k], 11]]]) * dxConsort
                # вычисление координат земной точки в НГцСК
            rzt[0, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[k]) * np.cos(
                2 * np.pi / Tz * (Tst + tt[k]) + betazt + betaztSh * tt[k])
            rzt[1, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[k]) * np.sin(
                2 * np.pi / Tz * (Tst + tt[k]) + betazt + betaztSh * tt[k])
            rzt[2, 0] = (Rz + Hzt) * np.sin(fizt + fiztSh * tt[k])
            RR[k] = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch - rzt)
        # аппроксимация дальности полиномом третьей степени
        B0 = np.sum(RR)
        B1 = np.sum(RR * tt)
        B2 = np.sum(RR * (tt ** 2))
        B3 = np.sum(RR * (tt ** 3))
        A = np.array([[5, sumtt, sumtt2, sumtt3],
                      [sumtt, sumtt2, sumtt3, sumtt4],
                      [sumtt2, sumtt3, sumtt4, sumtt5],
                      [sumtt3, sumtt4, sumtt5, sumtt6]])
        B = np.array([[B0], [B1], [B2], [B3]])
        pr = np.linalg.inv(A).dot(B)
        pr1 = np.flip(pr.T)
        # вычисляем все дальности на интервале наблюдения
        for i in range(Inabl):
            t1 = i * Tr
            b = np.array([[t1 ** 3], [t1 ** 2], [t1], [1.0]])
            r_Rch_zt[irch - 1, i] = pr1[0, 0] * b[0, 0] + pr1[0, 1] * b[1, 0] + \
                                    pr1[0, 2] * b[2, 0] + pr1[0, 3] * b[3, 0]

    # дальность на траверсе
    d0 = r_Rch_zt[0, 0]
    # ar=Vrsa^2/d0  # радиальное ускорение для компенсации МД и МЧ
    # нескомпенсированные скорости для приемных каналов
    Vr1 = (Lrch / 2) / d0 * Vrsa
    Vr2 = (-Lrch / 2) / d0 * Vrsa
    # непосредственно суммирование - КН с компенсацией МД и МЧ
    sum1 = 0
    sum2 = 0
    # дальность в момент начала синтезирования для первого канала
    d1 = r_Rch_zt[0, 0]
    for q in range(q1 - 1, q2):
        d = r_Rch_zt[0, q - q1 + 1]
        # дробный номер отсчета по быстрому времени
        ndr = (d / speedOfL - t_r_w + T0) * Kss * Fs - 1
        n = int(ndr)
        drob = ndr % 1
        ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
        fiq = 2 * math.pi / lamda * (r_Rch_zt[0, q - q1 + 1] - q1)
        # суммируем с учетом сдвига РЛИ по скорости
        sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq) * e ** (
                1j * 2 * np.pi * Vr1 / lamda * (q - q1 + 1) * Tr)

        ut = Uout02ss[n, q] * (1 - drob) + Uout02ss[n + 1, q] * drob
        fiq = 2 * np.pi / lamda * (r_Rch_zt[1, q - q1 + 1] - q1)
        sum2 = sum2 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq) * e ** (
                1j * 2 * np.pi * Vr2 / lamda * (q - q1 + 1) * Tr)

    Zxy1[nx, ny] = sum1
    Zxy2[nx, ny] = sum2


@time_of_function
def gpu_detail_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
              Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, Vrsa, tauRli, Inabl, qq, tt,
              Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0,
              sumtt, sumtt2, sumtt3, sumtt4, sumtt5, sumtt6, q1, q2, Tst):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array_1[blocks_per_grid, threads_per_block](
        Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
        Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, Vrsa, tauRli, Inabl, qq, tt,
        Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0,
        sumtt, sumtt2, sumtt3, sumtt4, sumtt5, sumtt6, q1, q2, Tst
    )
    return Zxy1


@time_of_function
def gpu_detail_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
                      Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, Vrsa, tauRli, Inabl, qq, tt,
                      Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0,
                      sumtt, sumtt2, sumtt3, sumtt4, sumtt5, sumtt6, q1, q2, Tst):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array_2[blocks_per_grid, threads_per_block](
        Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
        Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, Vrsa, tauRli, Inabl, qq, tt,
        Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0,
        sumtt, sumtt2, sumtt3, sumtt4, sumtt5, sumtt6, q1, q2, Tst
    )
    return Zxy1