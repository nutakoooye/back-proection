import numpy as np
from scipy.special import iv


def ChirpSig(t, tx, b):
    win = np.logical_and(t >= 0, t <= tx)
    comp = win * np.exp(1j * complex(b * (t ** 2)))
    return comp


def Win(tau, delta, n):
    W = 0

    # функции - типовые окна: delta - уровень пъедестала; tau=(t/T0-1/2)
    # 1 - прямоугольное окно
    Wrect = lambda tau, delta: 1

    # 2 - косинус на пъедестале, при delta=0.54 - Хэмминга (-42.7 дБ)
    Wcos = lambda tau, delta: delta + (1 - delta) * np.cos(np.pi * tau)

    # 3 - косинус квадрат на пъедестале
    Wcos2 = lambda tau, delta: delta + (1 - delta) * (np.cos(np.pi * tau) ** 2)

    # 4 - Хэмминга (-42.7 дБ)
    Whemming = lambda tau, delta: 0.54 + 0.46 * np.cos(np.pi * tau)

    # 5 - Хэннинга в третье степени на пъедестале (-39.3 дБ)
    Whenning3 = lambda tau, delta: delta + (1 - delta) * (np.cos(np.pi * tau) ** 3)

    # 6 - Хэннинга в четвертой степени на пъедестале (-46.7 дБ)
    Whenning4 = lambda tau, delta: delta + (1 - delta) * (np.cos(np.pi * tau) ** 4)

    # 7,8,9 - Кайзера-Бесселя для alfa=2.7; 3.1; 3.5 (-62.5; -72ю1; -81.8 дБ)
    Wkb27 = lambda tau, delta: iv(0, 2.7 * np.pi * np.sqrt(1 - (2 * tau) ** 2)) / iv(0, 2.7 * np.pi)
    Wkb31 = lambda tau, delta: iv(0, 3.1 * np.pi * np.sqrt(1 - (2 * tau) ** 2)) / iv(0, 3.1 * np.pi)
    Wkb35 = lambda tau, delta: iv(0, 3.5 * np.pi * np.sqrt(1 - (2 * tau) ** 2)) / iv(0, 3.5 * np.pi)

    # 10 - Блекмана-Херриса (-92 дБ)
    Wbx = lambda tau, delta: 0.35875 + 0.48829 * np.cos(2 * np.pi * tau) + 0.14128 * np.cos(
        4 * np.pi * tau) + 0.01168 * np.cos(6 * np.pi * tau)

    if n == 1:
        W = Wrect(tau, delta)
    elif n == 2:
        W = Wcos(tau, delta)
    elif n == 3:
        W = Wcos2(tau, delta)
    elif n == 4:
        W = Whemming(tau, delta)
    elif n == 5:
        W = Whenning3(tau, delta)
    elif n == 6:
        W = Whenning4(tau, delta)
    elif n == 7:
        W = Wkb27(tau, delta)
    elif n == 8:
        W = Wkb31(tau, delta)
    elif n == 9:
        W = Wkb35(tau, delta)
    elif n == 10:
        W = Wbx(tau, delta)

    return W
