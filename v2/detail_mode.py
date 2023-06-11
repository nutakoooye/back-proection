import numpy as np
import numba as nb
import math


@nb.jit(nopython=True)
def detail_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0, Rz,
                      betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, tauRli, speedOfL,
                      t_r_w, Kss, Fs, lamda, WinSampl, e, T0, q1, q2, Tst0):
    for nx in range(Nxsint):
        for ny in range(Nysint):
            # координаты текущей точки наблюдения
            xt = (-(Nxsint - 1) / 2 + nx) * dxsint
            yt = (-(Nysint - 1) / 2 + ny) * dysint
            zt = 0
            # подготовка исходных данных для вызова OrbitRange
            fizt = fizt0 + yt / Rz
            betazt = betazt0 + xt / Rz / math.cos(fizt0)
            Hzt = zt
            # номер первого периода на интервале синтезирования
            qst = int(tauRli / Tr) + 1
            # ОБНОВЛЕННАЯ ЧАСТЬ - прямой расчет дальностей для каждого периода
            # повторения в большом цикле без аппроксимации дальностей
            # получаем дальности в начале синтезирования
            t1 = Tst0 + (qst - 1) * Tr
            rzt = np.zeros((3, 1))
            # вычисление координат земной точки в НГцСК на момент t1
            rzt[0, 0] = (Rz + Hzt) * math.cos(fizt) * math.cos(2 * np.pi / Tz * t1 + betazt)
            rzt[1, 0] = (Rz + Hzt) * math.cos(fizt) * math.sin(2 * np.pi / Tz * t1 + betazt)
            rzt[2, 0] = (Rz + Hzt) * math.sin(fizt)
            # дальность между фазовым центром АР на передачу и точкой
            rrsa = np.array([[XYZ_rsa_ts[qst, 0]], [XYZ_rsa_ts[qst, 1]], [XYZ_rsa_ts[qst, 2]]]) * dxConsort
            # дальность между фазовым центрами приемных каналов и точкой
            rRch1 = np.array([[XYZ_rsa_ts[qst, 6]], [XYZ_rsa_ts[qst, 7]], [XYZ_rsa_ts[qst, 8]]]) * dxConsort
            # дальность в момент начала синтезирования для первого канала
            d01 = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch1 - rzt)
            # нескомпенсированные скорости для приемных каналов
            # Vr1=(Lrch/2)/d0*Vrsa; Vr2=(-Lrch/2)/d0*Vrsa; % ????
            # непосредственно суммирование - КН с компенсацией МД и МЧ
            sum1 = 0
            for q in range(q1 - 1, q2):
                # ОБНОВЛЕННАЯ ЧАСТЬ - прямой расчет суммарной дальности
                t1 = Tst0 + (q - 1) * Tr
                # вычисление координат земной точки в НГцСК на момент t1
                rzt = np.zeros((3, 1))
                rzt[0, 0] = (Rz + Hzt) * math.cos(fizt) * math.cos(2 * np.pi / Tz * t1 + betazt)
                rzt[1, 0] = (Rz + Hzt) * math.cos(fizt) * math.sin(2 * np.pi / Tz * t1 + betazt)
                rzt[2, 0] = (Rz + Hzt) * math.sin(fizt)
                # дальность между фазовым центром АР на передачу и точкой
                rrsa = np.array([[XYZ_rsa_ts[q, 0]], [XYZ_rsa_ts[q, 1]], [XYZ_rsa_ts[q, 2]]]) * dxConsort
                # дальность между фазовым центром 1-го приемного канала и точкой
                rRch1 = np.array([[XYZ_rsa_ts[q, 6]], [XYZ_rsa_ts[q, 7]], [XYZ_rsa_ts[q, 8]]]) * dxConsort
                # текущая дальность до первого канала
                dq1 = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch1 - rzt)
                # дробный номер отсчета по быстрому времени
                ndr = (dq1 / speedOfL - t_r_w + T0) * Kss * Fs - 1
                n = int(ndr)
                drob = ndr % 1
                ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
                fiq = 2 * math.pi / lamda * (dq1 - d01)
                # суммируем с учетом сдвига РЛИ по скорости
                sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq)

            Zxy1[nx, ny] = sum1

        print(nx, ' / ', Nxsint)
    return Zxy1


@nb.jit(nopython=True)
def detail_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint,
                      fizt0, Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, tauRli, speedOfL,
                      t_r_w, Kss, Fs, lamda, WinSampl, e, T0, q1, q2, Tst0):
    for nx in range(Nxsint):
        for ny in range(Nysint):
            # координаты текущей точки наблюдения
            xt = (-(Nxsint - 1) / 2 + nx) * dxsint
            yt = (-(Nysint - 1) / 2 + ny) * dysint
            zt = 0
            # подготовка исходных данных для вызова OrbitRange
            fizt = fizt0 + yt / Rz
            betazt = betazt0 + xt / Rz / math.cos(fizt0)
            Hzt = zt
            # номер первого периода на интервале синтезирования
            qst = int(tauRli / Tr) + 1
            # ОБНОВЛЕННАЯ ЧАСТЬ - прямой расчет дальностей для каждого периода
            # повторения в большом цикле без аппроксимации дальностей
            # получаем дальности в начале синтезирования
            t1 = Tst0 + (qst - 1) * Tr
            rzt = np.zeros((3, 1))
            # вычисление координат земной точки в НГцСК на момент t1
            rzt[0, 0] = (Rz + Hzt) * math.cos(fizt) * math.cos(2 * np.pi / Tz * t1 + betazt)
            rzt[1, 0] = (Rz + Hzt) * math.cos(fizt) * math.sin(2 * np.pi / Tz * t1 + betazt)
            rzt[2, 0] = (Rz + Hzt) * math.sin(fizt)
            # дальность между фазовым центром АР на передачу и точкой
            rrsa = np.array([[XYZ_rsa_ts[qst, 0]], [XYZ_rsa_ts[qst, 1]], [XYZ_rsa_ts[qst, 2]]]) * dxConsort
            # дальность между фазовым центрами приемных каналов и точкой
            rRch1 = np.array([[XYZ_rsa_ts[qst, 6]], [XYZ_rsa_ts[qst, 7]], [XYZ_rsa_ts[qst, 8]]]) * dxConsort
            rRch2 = np.array([[XYZ_rsa_ts[qst, 9]], [XYZ_rsa_ts[qst, 10]], [XYZ_rsa_ts[qst, 11]]]) * dxConsort
            # дальность в момент начала синтезирования для первого канала
            d01 = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch1 - rzt)
            # дальность в момент начала синтезирования для второго канала
            d02 = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch2 - rzt)
            # нескомпенсированные скорости для приемных каналов
            # Vr1=(Lrch/2)/d0*Vrsa; Vr2=(-Lrch/2)/d0*Vrsa; % ????
            # непосредственно суммирование - КН с компенсацией МД и МЧ
            sum1 = 0
            sum2 = 0
            for q in range(q1 - 1, q2):
                # ОБНОВЛЕННАЯ ЧАСТЬ - прямой расчет суммарной дальности
                t1 = Tst0 + (q - 1) * Tr
                # вычисление координат земной точки в НГцСК на момент t1
                rzt = np.zeros((3, 1))
                rzt[0, 0] = (Rz + Hzt) * math.cos(fizt) * math.cos(2 * np.pi / Tz * t1 + betazt)
                rzt[1, 0] = (Rz + Hzt) * math.cos(fizt) * math.sin(2 * np.pi / Tz * t1 + betazt)
                rzt[2, 0] = (Rz + Hzt) * math.sin(fizt)
                # дальность между фазовым центром АР на передачу и точкой
                rrsa = np.array([[XYZ_rsa_ts[q, 0]], [XYZ_rsa_ts[q, 1]], [XYZ_rsa_ts[q, 2]]]) * dxConsort
                # дальность между фазовым центром 1-го приемного канала и точкой
                rRch1 = np.array([[XYZ_rsa_ts[q, 6]], [XYZ_rsa_ts[q, 7]], [XYZ_rsa_ts[q, 8]]]) * dxConsort
                rRch2 = np.array([[XYZ_rsa_ts[q, 9]], [XYZ_rsa_ts[q, 10]], [XYZ_rsa_ts[q, 11]]]) * dxConsort
                # текущая дальность до первого канала
                dq1 = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch1 - rzt)
                dq2 = np.linalg.norm(rrsa - rzt) + np.linalg.norm(rRch2 - rzt)
                # дробный номер отсчета по быстрому времени
                ndr = (dq1 / speedOfL - t_r_w + T0) * Kss * Fs - 1
                n = int(ndr)
                drob = ndr % 1
                ut = Uout01ss[n, q] * (1 - drob) + Uout01ss[n + 1, q] * drob
                fiq = 2 * math.pi / lamda * (dq1 - d01)
                # суммируем с учетом сдвига РЛИ по скорости
                sum1 = sum1 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq)
                ndr = (dq2 / speedOfL - t_r_w + T0) * Kss * Fs - 1
                n = int(ndr)
                drob = ndr % 1
                ut = Uout02ss[n, q] * (1 - drob) + Uout02ss[n + 1, q] * drob
                fiq = 2 * math.pi / lamda * (dq2 - d02)
                # суммируем с учетом сдвига РЛИ по скорости
                sum2 = sum2 + ut * WinSampl[q - q1 + 1] * e ** (-1j * fiq)

            Zxy1[nx, ny] = sum1
            Zxy2[nx, ny] = sum2
        print(nx, ' / ', Nxsint)
    return Zxy1, Zxy2
