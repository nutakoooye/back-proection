import numpy as np
import math
import numba as nb


# @time_of_function
@nb.jit(nopython=True)
def range_calculation(K, Q, Tr, x, y, z, Vx,
                      Vy, Vz, Xrls, Yrls, Zrls,
                      Vxrls, Vyrls, Vzrls, df,
                      T0, SigBt, N, c, t2, Fs,
                      gamma, lamda, FiBT):
    RT1 = np.zeros((K, Q + 1), dtype=np.float64)
    for q in range(Q + 1):
        t = q * Tr  # текущее время
        for k in range(K):
            RT1[k, q] = math.sqrt(
                (x[k] + Vx[k] * t - (Xrls[0] + Vxrls * t)) ** 2 +
                (y[k] + Vy[k] * t - (Yrls[0] + Vyrls * t)) ** 2 +
                (z[k] + Vz[k] * t - (Zrls[0] + Vzrls * t)) ** 2)
    Akcomp = np.sqrt(df * T0 * SigBt)
    U01comp = np.zeros((N, Q), dtype=np.complex128)
    for q in range(Q):
        for k in range(K):
            n0 = int((2 * RT1[k, q] / c - t2 + T0) * Fs)
            n2 = n0 + gamma
            if n2 > N:
                n2 = N
            n1 = n0 - gamma
            if n1 < 1:
                n1 = 0
            for n in range(n1, n2):  # добавляем отсчеты сигнала
                t = t2 + (n + 1) / Fs - (2 * RT1[k, q] / c + T0)
                a = Akcomp[k] * math.sinc(df * t) * np.exp(
                    4j * math.pi * RT1[k, q] / lamda + FiBT[k])
                U01comp[n, q] += a  # *exp(j*pi*Fs/Ks*t);
    return U01comp


# @time_of_function
@nb.jit(nopython=True)
def range_calculation_2_chanels(K, Q, Tr, x, y, z, Vx,
                                Vy, Vz, Xrls, Yrls, Zrls,
                                Vxrls, Vyrls, Vzrls, df,
                                T0, Ks, Ps, N, c, t2, Fs,
                                gamma, lamda, FiBT):
    RT1s = np.zeros((K, Q + 1))
    RT2s = np.zeros((K, Q + 1))
    for q in range(Q + 1):
        t = q * Tr  # текущее время
        for k in range(K):
            # дальность между передатчиком и БТ
            r00 = math.sqrt((x[k] + Vx[k] * t - (Xrls[0] + Vxrls * t)) ** 2 +
                            (y[k] + Vy[k] * t - (Yrls[0] + Vyrls * t)) ** 2 +
                            (z[k] + Vz[k] * t - (Zrls[0] + Vzrls * t)) ** 2)
            # первая суммарная дальность
            RT1s[k, q] = math.sqrt(
                (x[k] + Vx[k] * t - (Xrls[1] + Vxrls * t)) ** 2 +
                (y[k] + Vy[k] * t - (Yrls[1] + Vyrls * t)) ** 2 +
                (z[k] + Vz[k] * t - (Zrls[1] + Vzrls * t)) ** 2) + r00
            # вторая суммарная дальность
            RT2s[k, q] = math.sqrt(
                (x[k] + Vx[k] * t - (Xrls[2] + Vxrls * t)) ** 2 +
                (y[k] + Vy[k] * t - (Yrls[2] + Vyrls * t)) ** 2 +
                (z[k] + Vz[k] * t - (Zrls[2] + Vzrls * t)) ** 2) + r00

    # генерация сигнала
    Akcomp = np.sqrt(df * T0 * Ps)
    U01comp = np.zeros((N, Q), np.complex128)
    U02comp = np.zeros((N, Q), np.complex128)
    for q in range(Q):
        for k in range(K):
            # первый приемный канал
            n0 = int((RT1s[k, q] / c - t2 + T0) * Fs)
            n2 = n0 + gamma
            if n2 > N:
                n2 = N  # максимальное значение m
            n1 = n0 - gamma
            if n1 < 1:
                n1 = 0  # минимальное значение m
            for n in range(n1, n2):  # добавляем отсчеты сигнала
                t = t2 + (n + 1) / Fs - (RT1s[k, q] / c + T0)
                U01comp[n, q] += Akcomp[k] * math.sinc(df * t) * np.exp(
                    1j * (2 * math.pi * RT1s[k, q] / lamda + FiBT[
                        k])) * np.exp(1j * math.pi * Fs / Ks * t)

            # второй приемный канала
            n0 = int((RT2s[k, q] / c - t2 + T0) * Fs)
            n2 = n0 + gamma
            if n2 > N:
                n2 = N  # максимальное значение m
            n1 = n0 - gamma
            if n1 < 1:
                n1 = 0  # минимальное значение m
            for n in range(n1, n2):  # добавляем отсчеты сигнала
                t = t2 + (n + 1) / Fs - (RT2s[k, q] / c + T0)
                U02comp[n, q] = U02comp[n, q] + Akcomp[k] * math.sinc(
                    df * t) * np.exp(
                    1j * (2 * math.pi * RT2s[k, q] / lamda + FiBT[
                        k])) * np.exp(1j * math.pi * Fs / Ks * t)
    return U01comp, U02comp


# @time_of_function
@nb.jit(nopython=True)
def restore_signale_spectrum(Gh0, Y001, FlagInter, Q, Ks,
                             Y002=np.ndarray((1, 1), np.complex128)):
    N = len(Gh0)
    Gh0max = np.max(np.abs(Gh0[:N // 4]))
    nmax = np.argmax(np.abs(Gh0[:N // 4]))
    for q in range(Q):
        for n in range(N):
            if n >= int(0.75 * nmax) and n <= N / Ks - int(0.75 * nmax):
                Y001[n, q] = Y001[n, q] / Gh0[n]
                if FlagInter == 1:
                    Y002[n, q] = Y002[n, q] / Gh0[n]
            else:
                Y001[n, q] = 0
                if FlagInter == 1:
                    Y002[n, q] = 0
    return Y001, Y002
