import math
import numpy as np
from numba import cuda

from v1.utils import time_of_function


@cuda.jit
def kernel_2d_array_1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
               Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
               Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0):
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
    # получаем аппроксимацию дальности и азимута точки на всем времени
    # наблюдения
    Tst = Tst0
    Inabl = Q
    Tnabl = Q * Tr  # время общее наблюдения
    Q_cons_0 = 1

    # аппрокимация закона изменения азимута
    # Calc_R_az_route
    qq = np.empty(5, dtype=np.int32)
    tt = np.empty(5)
    qq[0] = Q_cons_0 - 1
    tt[0] = 0
    qq[1] = Q_cons_0 - 1 + int(Inabl / 4) - 1
    tt[1] = Tnabl / 4
    qq[2] = Q_cons_0 - 1 + int(Inabl / 2) - 1
    tt[2] = Tnabl / 2
    qq[3] = Q_cons_0 - 1 + int(3 * Inabl / 4) - 1
    tt[3] = 3 * Tnabl / 4
    qq[4] = Q_cons_0 - 1 + Inabl - 1
    tt[4] = Tnabl
    RRaz = np.empty(5)
    for i in range(5):
        # вектор координат РСА в НГцСК
        rrsa = np.array([[XYZ_rsa_ts[qq[i], 0]], [XYZ_rsa_ts[qq[i], 1]], [XYZ_rsa_ts[qq[i], 2]]]) * dxConsort
        # вектор скоростей РСА в НГцСК
        rShrsa = np.array([XYZ_rsa_ts[qq[i], 3], XYZ_rsa_ts[qq[i], 4], XYZ_rsa_ts[qq[i], 5]]) * dVxConsort
        # вектор координат земной точки в НГцСК
        # rzt=xyzZt(Tst+tt(k),fizt+fiztSh*tt(k),betazt+betaztSh*tt(k),Hzt)
        rzt = np.zeros((3, 1))
        rzt[0, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[i]) * np.cos(
            2 * np.pi / Tz * (Tst + tt[i]) + betazt + betaztSh * tt[i])
        rzt[1, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[i]) * np.sin(
            2 * np.pi / Tz * (Tst + tt[i]) + betazt + betaztSh * tt[i])
        rzt[2, 0] = (Rz + Hzt) * np.sin(fizt + fiztSh * tt[i])
        # матрица преобразования из НГцСК в ССК
        M_ngsk_ssk = np.zeros((3, 3))
        M_ngsk_ssk[0, 0] = (rShrsa[1] * rrsa[2, 0] - rShrsa[2] * rrsa[1, 0]) / Vrsa / (Rz + Hrsa)
        M_ngsk_ssk[0, 1] = -(rShrsa[0] * rrsa[2, 0] - rShrsa[2] * rrsa[0, 0]) / Vrsa / (Rz + Hrsa)
        M_ngsk_ssk[0, 2] = (rShrsa[0] * rrsa[1, 0] - rShrsa[1] * rrsa[0, 0]) / Vrsa / (Rz + Hrsa)
        M_ngsk_ssk[1, 0] = rShrsa[0] / Vrsa
        M_ngsk_ssk[1, 1] = rShrsa[1] / Vrsa
        M_ngsk_ssk[1, 2] = rShrsa[2] / Vrsa
        M_ngsk_ssk[2, 0] = rrsa[0, 0] / (Rz + Hrsa)
        M_ngsk_ssk[2, 1] = rrsa[1, 0] / (Rz + Hrsa)
        M_ngsk_ssk[2, 2] = rrsa[2, 0] / (Rz + Hrsa)
        # вектор координат точки земной поверхности в ССК
        rzt_ssk = M_ngsk_ssk.dot(rzt - rrsa)
        # матрица пересчета из антенной системы в скоростную
        al = XYZ_rsa_ts[qq[i], 12] * dugConsort
        be = XYZ_rsa_ts[qq[i], 13] * dugConsort
        Msskask = np.array([
            [np.cos(al) * np.cos(be), np.sin(al) * np.cos(be), np.sin(be)],
            [-np.sin(al), np.cos(al), 0],
            [-np.cos(al) * np.sin(be), -np.sin(al) * np.sin(be), np.cos(be)]
        ])
        # вектор координат точки земной поверхности в АСК
        rzt_ask = Msskask.dot(rzt_ssk)
        # азимут земной точки относительно нормали к антенной системе
        RRaz[i] = math.atan(rzt_ask[1, 0] / rzt_ask[2, 0])

    # аппроксимация азимута полиномом второй степени
    sumtt = np.sum(tt)
    sumtt2 = np.sum(tt ** 2)
    sumtt3 = np.sum(tt ** 3)
    sumtt4 = np.sum(tt ** 4)
    B0 = np.sum(RRaz)
    B1 = np.sum(RRaz * tt)
    B2 = np.sum(RRaz * (tt ** 2))
    A = np.array([[5, sumtt, sumtt2],
                  [sumtt, sumtt2, sumtt3],
                  [sumtt2, sumtt3, sumtt4]])
    B = np.array([
        [B0],
        [B1],
        [B2]
    ])
    paz = np.linalg.inv(A).dot(B)
    paz = np.flip(paz.T)
    # определяем время прохождения траверса и интервал индексов отсчетов
    # для суммирования отсчетов
    aa = paz[0, 0]
    bb = paz[0, 1]
    cc = paz[0, 2]
    dd = bb ** 2 - 4 * aa * cc
    # момент времени прохождения траверса относительно Tst0
    ttr = (-bb - np.sqrt(dd)) / 2 / aa
    # индекс номера периода повторения для траверса
    q0 = int(ttr / Tr)
    q1 = q0 - int(Tsint / 2 / Tr)  # начальный индекс суммирования
    q2 = q0 + int(Tsint / 2 / Tr)  # конечный индекс суммирования

    # уточение закона изменения дальности на участке синтезирования
    # для этого выбираем начало участка на половину интервала
    # синтезирования справа от траверса и уточняем аппроксимацию
    # дальности с использованием полинома третьей степени
    Tst = Tst0 + (q1 - 1) * Tr
    Inabl = q2 - q1 + 1
    Tnabl = (q2 - q1 + 1) * Tr  # время общее наблюдения
    Q_cons_0 = q1
    # аппроксимируем суммарную дальность фазовый центр антенны на передачу -
    # земная точка - фазовый центр приемной подрешетки на интервале наблюдения
    # полиномом третьей степени по пяти точкам
    r_Rch_zt = np.zeros((1, Inabl))
    qq = np.zeros(5, dtype=np.int32)
    tt = np.empty(5)
    qq[0] = Q_cons_0 - 1
    tt[0] = 0
    qq[1] = Q_cons_0 - 1 + round(Inabl / 4) - 1
    tt[1] = Tnabl / 4
    qq[2] = Q_cons_0 - 1 + round(Inabl / 2) - 1
    tt[2] = Tnabl / 2
    qq[3] = Q_cons_0 - 1 + round(3 * Inabl / 4) - 1
    tt[3] = 3 * Tnabl / 4
    qq[4] = Q_cons_0 - 1 + Inabl - 1
    tt[4] = Tnabl
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
    sumtt = np.sum(tt)
    sumtt2 = np.sum(tt ** 2)
    sumtt3 = np.sum(tt ** 3)
    sumtt4 = np.sum(tt ** 4)
    sumtt5 = np.sum(tt ** 5)
    sumtt6 = np.sum(tt ** 6)
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
    d0 = r_Rch_zt[0, q0 - q1]
    # ar=Vrsa^2/d0  # радиальное ускорение для компенсации МД и МЧ
    # нескомпенсированные скорости для приемных каналов
    Vr1 = (Lrch / 2) / d0 * Vrsa
    Vr2 = (-Lrch / 2) / d0 * Vrsa
    # непосредственно суммирование - КН с компенсацией МД и МЧ
    sum1 = 0
    sum2 = 0
    for q in range(q1 - 1, q2):
        d = r_Rch_zt[0, q - q1 + 1]
        # дробный номер отсчета по быстрому времени
        ndr = (d / speedOfL - t_r_w + T0) * Kss * Fs - 1
        n = int(ndr)
        drob = ndr % 1
        ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
        fiq = 2 * math.pi / lamda * (r_Rch_zt[0, q - q1 + 1] - r_Rch_zt[0, q0 - q1])
        # суммируем с учетом сдвига РЛИ по скорости
        sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq) * e ** (
                1j * 2 * np.pi * Vr1 / lamda * (q - q0 + 1) * Tr)

    Zxy1[nx, ny] = sum1


@cuda.jit
def kernel_2d_array_2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
               Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
               Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0):
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
    # получаем аппроксимацию дальности и азимута точки на всем времени
    # наблюдения
    Tst = Tst0
    Inabl = Q
    Tnabl = Q * Tr  # время общее наблюдения
    Q_cons_0 = 1

    # аппрокимация закона изменения азимута
    # Calc_R_az_route
    qq = np.empty(5, dtype=np.int32)
    tt = np.empty(5)
    qq[0] = Q_cons_0 - 1
    tt[0] = 0
    qq[1] = Q_cons_0 - 1 + int(Inabl / 4) - 1
    tt[1] = Tnabl / 4
    qq[2] = Q_cons_0 - 1 + int(Inabl / 2) - 1
    tt[2] = Tnabl / 2
    qq[3] = Q_cons_0 - 1 + int(3 * Inabl / 4) - 1
    tt[3] = 3 * Tnabl / 4
    qq[4] = Q_cons_0 - 1 + Inabl - 1
    tt[4] = Tnabl
    RRaz = np.empty(5)
    for i in range(5):
        # вектор координат РСА в НГцСК
        rrsa = np.array([[XYZ_rsa_ts[qq[i], 0]], [XYZ_rsa_ts[qq[i], 1]], [XYZ_rsa_ts[qq[i], 2]]]) * dxConsort
        # вектор скоростей РСА в НГцСК
        rShrsa = np.array([XYZ_rsa_ts[qq[i], 3], XYZ_rsa_ts[qq[i], 4], XYZ_rsa_ts[qq[i], 5]]) * dVxConsort
        # вектор координат земной точки в НГцСК
        # rzt=xyzZt(Tst+tt(k),fizt+fiztSh*tt(k),betazt+betaztSh*tt(k),Hzt)
        rzt = np.zeros((3, 1))
        rzt[0, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[i]) * np.cos(
            2 * np.pi / Tz * (Tst + tt[i]) + betazt + betaztSh * tt[i])
        rzt[1, 0] = (Rz + Hzt) * np.cos(fizt + fiztSh * tt[i]) * np.sin(
            2 * np.pi / Tz * (Tst + tt[i]) + betazt + betaztSh * tt[i])
        rzt[2, 0] = (Rz + Hzt) * np.sin(fizt + fiztSh * tt[i])
        # матрица преобразования из НГцСК в ССК
        M_ngsk_ssk = np.zeros((3, 3))
        M_ngsk_ssk[0, 0] = (rShrsa[1] * rrsa[2, 0] - rShrsa[2] * rrsa[1, 0]) / Vrsa / (Rz + Hrsa)
        M_ngsk_ssk[0, 1] = -(rShrsa[0] * rrsa[2, 0] - rShrsa[2] * rrsa[0, 0]) / Vrsa / (Rz + Hrsa)
        M_ngsk_ssk[0, 2] = (rShrsa[0] * rrsa[1, 0] - rShrsa[1] * rrsa[0, 0]) / Vrsa / (Rz + Hrsa)
        M_ngsk_ssk[1, 0] = rShrsa[0] / Vrsa
        M_ngsk_ssk[1, 1] = rShrsa[1] / Vrsa
        M_ngsk_ssk[1, 2] = rShrsa[2] / Vrsa
        M_ngsk_ssk[2, 0] = rrsa[0, 0] / (Rz + Hrsa)
        M_ngsk_ssk[2, 1] = rrsa[1, 0] / (Rz + Hrsa)
        M_ngsk_ssk[2, 2] = rrsa[2, 0] / (Rz + Hrsa)
        # вектор координат точки земной поверхности в ССК
        rzt_ssk = M_ngsk_ssk.dot(rzt - rrsa)
        # матрица пересчета из антенной системы в скоростную
        al = XYZ_rsa_ts[qq[i], 12] * dugConsort
        be = XYZ_rsa_ts[qq[i], 13] * dugConsort
        Msskask = np.array([
            [np.cos(al) * np.cos(be), np.sin(al) * np.cos(be), np.sin(be)],
            [-np.sin(al), np.cos(al), 0],
            [-np.cos(al) * np.sin(be), -np.sin(al) * np.sin(be), np.cos(be)]
        ])
        # вектор координат точки земной поверхности в АСК
        rzt_ask = Msskask.dot(rzt_ssk)
        # азимут земной точки относительно нормали к антенной системе
        RRaz[i] = math.atan(rzt_ask[1, 0] / rzt_ask[2, 0])

    # аппроксимация азимута полиномом второй степени
    sumtt = np.sum(tt)
    sumtt2 = np.sum(tt ** 2)
    sumtt3 = np.sum(tt ** 3)
    sumtt4 = np.sum(tt ** 4)
    B0 = np.sum(RRaz)
    B1 = np.sum(RRaz * tt)
    B2 = np.sum(RRaz * (tt ** 2))
    A = np.array([[5, sumtt, sumtt2],
                  [sumtt, sumtt2, sumtt3],
                  [sumtt2, sumtt3, sumtt4]])
    B = np.array([
        [B0],
        [B1],
        [B2]
    ])
    paz = np.linalg.inv(A).dot(B)
    paz = np.flip(paz.T)
    # определяем время прохождения траверса и интервал индексов отсчетов
    # для суммирования отсчетов
    aa = paz[0, 0]
    bb = paz[0, 1]
    cc = paz[0, 2]
    dd = bb ** 2 - 4 * aa * cc
    # момент времени прохождения траверса относительно Tst0
    ttr = (-bb - np.sqrt(dd)) / 2 / aa
    # индекс номера периода повторения для траверса
    q0 = int(ttr / Tr)
    q1 = q0 - int(Tsint / 2 / Tr)  # начальный индекс суммирования
    q2 = q0 + int(Tsint / 2 / Tr)  # конечный индекс суммирования

    # уточение закона изменения дальности на участке синтезирования
    # для этого выбираем начало участка на половину интервала
    # синтезирования справа от траверса и уточняем аппроксимацию
    # дальности с использованием полинома третьей степени
    Tst = Tst0 + (q1 - 1) * Tr
    Inabl = q2 - q1 + 1
    Tnabl = (q2 - q1 + 1) * Tr  # время общее наблюдения
    Q_cons_0 = q1
    # аппроксимируем суммарную дальность фазовый центр антенны на передачу -
    # земная точка - фазовый центр приемной подрешетки на интервале наблюдения
    # полиномом третьей степени по пяти точкам
    r_Rch_zt = np.zeros((2, Inabl))
    RR = 0
    RRaz = 0
    qq = np.zeros(5, dtype=np.int32)
    tt = np.empty(5)
    qq[0] = Q_cons_0 - 1
    tt[0] = 0
    qq[1] = Q_cons_0 - 1 + round(Inabl / 4) - 1
    tt[1] = Tnabl / 4
    qq[2] = Q_cons_0 - 1 + round(Inabl / 2) - 1
    tt[2] = Tnabl / 2
    qq[3] = Q_cons_0 - 1 + round(3 * Inabl / 4) - 1
    tt[3] = 3 * Tnabl / 4
    qq[4] = Q_cons_0 - 1 + Inabl - 1
    tt[4] = Tnabl
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
        sumtt = np.sum(tt)
        sumtt2 = np.sum(tt ** 2)
        sumtt3 = np.sum(tt ** 3)
        sumtt4 = np.sum(tt ** 4)
        sumtt5 = np.sum(tt ** 5)
        sumtt6 = np.sum(tt ** 6)
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
    d0 = r_Rch_zt[0, q0 - q1]
    # ar=Vrsa^2/d0  # радиальное ускорение для компенсации МД и МЧ
    # нескомпенсированные скорости для приемных каналов
    Vr1 = (Lrch / 2) / d0 * Vrsa
    Vr2 = (-Lrch / 2) / d0 * Vrsa
    # непосредственно суммирование - КН с компенсацией МД и МЧ
    sum1 = 0
    sum2 = 0
    for q in range(q1 - 1, q2):
        d = r_Rch_zt[0, q - q1 + 1]
        # дробный номер отсчета по быстрому времени
        ndr = (d / speedOfL - t_r_w + T0) * Kss * Fs - 1
        n = int(ndr)
        drob = ndr % 1
        ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
        fiq = 2 * math.pi / lamda * (r_Rch_zt[0, q - q1 + 1] - r_Rch_zt[0, q0 - q1])
        # суммируем с учетом сдвига РЛИ по скорости
        sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq) * e ** (
                1j * 2 * np.pi * Vr1 / lamda * (q - q0 + 1) * Tr)

        ut = Uout02ss[n, q] * (1 - drob) + Uout02ss[n + 1, q] * drob
        fiq = 2 * np.pi / lamda * (r_Rch_zt[1, q - q1 + 1] - r_Rch_zt[1, q0 - q1])
        sum2 = sum2 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq) * e ** (
                1j * 2 * np.pi * Vr2 / lamda * (q - q0 + 1) * Tr)

    Zxy1[nx, ny] = sum1
    Zxy2[nx, ny] = sum2

@time_of_function
def gpu_route_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
               Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
               Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array_1[blocks_per_grid, threads_per_block](
        Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
        Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
        Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0
    )
    return Zxy1


@time_of_function
def gpu_route_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
               Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
               Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(Zxy1.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(Zxy1.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    kernel_2d_array_1[blocks_per_grid, threads_per_block](
        Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
        Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
        Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0
    )
    return Zxy1