import math

import numpy as np
import matplotlib.pyplot as plt

from v2.functions import Win, ChirpSig
from v2.utils import time_of_function
from v2.route_mode import route_big_cycle1, route_big_cycle2
from v2.detail_mode import detail_big_cycle1, detail_big_cycle2
from v2.route_mode_gpu import gpu_route_big_cycle1, gpu_route_big_cycle2
from v2.detail_mode_gpu import gpu_detail_big_cycle1, gpu_detail_big_cycle2

# МАРШРУТНЫЙ РЕЖИМ: построение РЛИ в координатах (широта/долгота) при
# увеличенной частоты дискретизации.
# Вход: 
# файл  c с именем app.fileIsx с параметрами моделирования 
# бинарные файлы  с именами app.fileTS1, fileTS2 с траекторными сигналами 
# одного (Nrch=1) или двух (Nrch=2) интерферометрических преимных каналов
# файл с именем app.FileConsort с координатами РСА по периодам повторения 
# При ВПО проводится увеличение частоты дискретизации траекторного сигнала 
# в Kss раз за счет дополнения спектра нулями cправа,вычисляется ИХ и КЧХ 
# согласованного фильтра ВПО при увеличенной частоте дискретизации с учетом 
# типа TypeWinDn оконной функции по дальности, перемножение спектра сигнала 
# и КЧХ фильтра и обратное ДПФ.
# При МПО проводится синтез для увеличенной частоты дискретизации, для чего 
# с заданной дискретностью просматриваются все точки заданной зоны 
# синтезирования по широте и долготе при нулевой высоте точки на Земле. 
# Для каждой точки вычисляется момент времени прохождения траверса, когда 
# азимут точки равен нулю, интервал синтезирования симметричен  
# относительно траверса, центральный период повторения равен q0 
# для каждого периода повторения в зоне синтезирования от q1=q0-Tsint/Tr/2 
# до q2=q0+Tsint/Tr/2 проводится вычисление дальности до точки, индекс n 
# положения сжатого сигнала по быстрому времени и суммирование этих 
# сигналов на всем интервале синтезирования с компенсацией изменения фазы. 
# Расчеты проводятся для одного или двух приемных каналов с построением
# разности фаз сформированных РЛИ или разностного РЛИ

####################### ПАРАМЕТРЫ ОБРАБОТКИ ###############################
# оконные функции:
# 1 - прямоугольное окно
# 2 - косинус на пъедестале, при delta=0.54 - Хэмминга (-42.7 дБ)
# 3 - косинус квадрат на пъедестале
# 4 - Хэмминга (-42.7 дБ)
# 5 - Хэннинга в третье степени на пъедестале (-39.3 дБ)
# 6 - Хэннинга в четвертой степени на пъедестале (-46.7 дБ)
# 7,8,9 - Кайзера-Бесселя для  alfa=2.7 3.1 3.5 (-62.5 -72.1 -81.8 дБ)
# 10 - Блекмана-Херриса (-92 дБ)

# вариант задания исходных данных через форму
def calc_rli(client_values, QPrint):
    ####################  Считывание данных из UI #########################
    Kss = client_values['Kss']  # коэффициент передискретизации
    dxsint = client_values['dxsint']  # дискретность зоны синтезирования по Ox
    dysint = client_values['dysint']  # дискретность зоны синтезирования по Oy
    StepBright = client_values['StepBright']  # показатель степени при отображении
    Nxsint = client_values['Nxsint']  # число точек по оси Ox (долготе)
    Nysint = client_values['Nysint']  # число точек по оси Oy (широте)
    Tsint = client_values['Tsint']  # время синтезирования
    tauRli = client_values['tauRli']
    RegimRsa = client_values['RegimRsa']  # режим радиолокационной съемки 1 - детальный, 2 - маршрутный
    TypeWinDp = client_values['TypeWinDp']  # тип оконной функции при сжатии по поперечной дальности
    TypeWinDn = client_values['TypeWinDn']  # тип оконной функции при сжатии по наклонной дальности
    GPUCalculationFlag = client_values['isGPU']
    ConsortPath = client_values['ConsortPath']
    ModelDatePath = client_values['ModelDatePath']
    Yts1Path = client_values['Yts1Path']
    Yts2Path = client_values['Yts2Path']
    FlagViewSignal = client_values['FlagViewSignal']  # флаг отображения сигналов в ходе расчетов
    FlagWriteRli = client_values['FlagWriteRli']
    NumRli = 1  # номер РЛИ в последовательности ***

    ####################  Считывание исходных данных #########################
    # параметры Земли
    # считывание параметров моделирования из консорт-файла параметров РСА
    ModelDate=[]
    with open(ModelDatePath, 'r') as f:
        while True:
            # считываем строку
            line = f.readline()
            if not line:
                break
            ModelDate.append(float(line))
            # прерываем цикл, если строка пустая

    Hrsa = ModelDate[0]  # высота орбиты РСА
    lamda = ModelDate[1]  # длина волны
    alfa05Ar = ModelDate[2]  # ширина главного лепестка ДН по азимуту
    teta05Ar = ModelDate[3]  # ширина главного лепестка ДН по углу места
    T0 = ModelDate[4]  # длительность импульса
    Tr = ModelDate[5]  # период повторения
    df = ModelDate[6]  # ширина спектра сигнала
    Fs = ModelDate[7]  # частота дискретизации
    fizt0 = ModelDate[8]  # широта центра участка синтезирования
    betazt0 = ModelDate[9]  # долгота центра участка синтезирования
    bzc = ModelDate[10]  # параметр согласованного фильтра bzc=pi*df/T0
    t_r_w = ModelDate[11]  # время задержки записи по отношению к началу периода
    N = int(ModelDate[12])  # число отсчетов по быстрому времени
    Q = int(ModelDate[13])  # число периодов повторения
    Nrch = int(ModelDate[14])
    Nrch = 1  # число приемных каналов
    Lrch = ModelDate[15]  # расстояние между фазовыми центрами приемных каналов
    Vrsa = ModelDate[16]  # скорость РСА
    # Tsint = ModelDate[17]  # время синтезирования ?
    Tst0 = ModelDate[18]  # момент начала получения траекторного сигнала
    dxConsort = ModelDate[19]  # дискретность данных в консорт-файле по координатам
    dVxConsort = ModelDate[20]  # дискретность данных в консорт-файле по скорости
    dugConsort = ModelDate[21]  # дискретность данных в консорт-файле по углам

    # считывание координат РСА из консорт-файла размерностью Q*14
    XYZ_rsa_ts = np.genfromtxt(ConsortPath, dtype=float, delimiter='  ')
    # считывание траекторного сигнала из файла
    with open(Yts1Path, 'rb') as file:
        Yts = np.fromfile(file, dtype=np.int16).reshape((Q, 2 * N)).T

    # формируем комплексный массив и преобразуем его к целым числам
    Yts1r = Yts[:N, :Q] + 1j * Yts[N:2 * N, :Q]
    Yts1r = Yts1r.astype(np.complex64)
    if Nrch == 2:
        with open(Yts2Path, 'rb') as file:
            Yts = np.fromfile(file, dtype=np.int16).reshape((Q, 2 * N)).T
        # формируем комплексный массив и преобразуем его к целым числам
        Yts2r = Yts[:N, :Q] + 1j * Yts[N:2 * N, :Q]

    del Yts

    # КОНСТАНТЫ
    Rz = 6371000  # радиус Земли
    wz = 7.2921158553e-05  # угловая скорость Земли
    Tz = 2 * np.pi / wz  # период вращения Земли - солнечные сутки
    speedOfL = 3 * 10 ** 8  # скорость света
    e = 2.71828

    qst = int(tauRli / Tr) + 1  # номер периода повторение с которого начинаем расчет для детального режима

    ######################### ОСНОВНЫЕ ВЫЧИСЛЕНИЯ ############################
    # вычисление внутрипериодных спектров сигналов приемных каналов

    Y01 = np.fft.fft(Yts1r, axis=0)
    if Nrch == 2:
        Y02 = np.fft.fft(Yts2r, axis=0)
        del Yts2r

    del Yts1r
    ###################   ВПО с передискретизацией  ##########################

    # вычисление КЧХ фильтра с увеличенным числом отсчетов с учетом оконной функции
    h0ss = np.zeros(N * Kss, np.complex64)
    for n in range(int(T0 * Fs * Kss + 1)):
        h0ss[n] = np.conj(ChirpSig(T0 - (n) / (Fs * Kss), T0, bzc)) * Win((T0 - (n) / (Fs * Kss)) / T0 - 0.5, 0,
                                                                          TypeWinDn)

    Gh0ss = np.fft.fft(h0ss)
    Gh0ss = Gh0ss / math.sqrt(N * Kss)

    del h0ss

    # передискретизация по наклонной дальности - дополнение спектра нулями
    # новый вариант передискретизации - просто добавляем нули
    @time_of_function
    def oversampling(N, Q, Kss, Y0X, Gh0ss): # в отдельной функции для удобства
        Y01ss = np.empty((Kss * N, Q), np.complex64)
        Y01ss[:N, :Q] = Y0X[:N, :Q]

        # сжатие по дальности
        Goutss = Y01ss * Gh0ss[:, np.newaxis]  # Векторизованная операция numpy для ускорения

        return Goutss
        # 4 секунды копирование и перемножение матриц 16 - обратное БПФ

    np.fft.ifft = time_of_function(np.fft.ifft)

    Goutss = oversampling(N, Q, Kss, Y01, Gh0ss)
    # вычисляем обратное БПФ после передискретизации на видеокарте если включен флаг
    if GPUCalculationFlag:
        from v2.cupy_functions import cuifft
        Uout01ss = cuifft(Goutss) * np.sqrt(N * Kss)
    else:
        Uout01ss = np.fft.ifft(Goutss, axis=0).astype(np.complex64) * np.sqrt(N * Kss)
    if Nrch == 2:
        # второй канал
        # новый вариант передискретизации - просто добавляем нули
        Goutss = oversampling(N, Q, Kss, Y02, Gh0ss)
        if GPUCalculationFlag:
            Uout02ss = cuifft(Goutss) * np.sqrt(N * Kss)
        else:
            Uout02ss = np.fft.ifft(Goutss, axis=0).astype(np.complex64) * np.sqrt(N * Kss)
    del Gh0ss
    # отображение сигналов до после ВПО при установленном флаге отображения
    if FlagViewSignal == 1:  # 13 секунд !!!
        # График 1
        fig1 = plt.figure()
        n = np.arange(1, N * Kss + 1)
        plt.plot(n, np.abs(Uout01ss[n - 1, 0]), n, np.abs(Uout01ss[n - 1, Q // 2]), n, np.abs(Uout01ss[n - 1, Q - 1]),
                 linewidth=0.5)
        plt.grid(True)
        plt.title('Сигналы после ВПО')

        # График 2
        fig2 = plt.figure()

        Vrli0 = (np.abs(Y01.T) ** StepBright)
        Vrli0 = Vrli0 / np.max(np.max(Vrli0))
        Vrli0 = np.flipud(Vrli0)

        plt.subplot(2, 1, 1)
        plt.imshow(Vrli0)
        plt.xlabel('Наклонная дальность')
        plt.ylabel('Период повторения')
        plt.title('Сигналы на входе ВПО')
        Vrli0 = (np.abs(Uout01ss.T) ** StepBright)
        Vrli0 = Vrli0 / np.max(np.max(Vrli0))
        Vrli0 = np.flipud(Vrli0)

        plt.subplot(2, 1, 2)
        plt.imshow(Vrli0)
        plt.xlabel('Наклонная дальность')
        plt.ylabel('Период повторения')
        plt.title('Сигналы на выходе ВПО')

    ######## сжатие в координатах (широта/долгота) Backprojection  ###############
    # отсчеты накопленного комплексного РЛИ-1
    Zxy1 = np.zeros((Nxsint, Nysint), np.complex64)
    if Nrch == 2:
        Zxy2 = np.zeros((Nxsint, Nysint), np.complex64)

    # подготовка значений оконной функции
    Nr = int(Tsint / Tr)  # число периодов повторения на интервале синтезирования
    WinSampl = np.ones(Nr + 1)
    for q in range(Nr + 1):
        WinSampl[q] = Win((q + 1 - Nr / 2) / Nr, 0, TypeWinDp)  # отсчеты оконной функции

    if RegimRsa == 2:  # маршрутный режим
        # основной расчетный цикл ДЛЯ МАКСИМАЛЬНОГО РАСПАРАЛЛЕЛИВАНИЯ
        if GPUCalculationFlag: # если включен флаг считаем на ГП
            if Nrch == 1:
                Zxy1 = gpu_route_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
                                            Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
                                            Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e,
                                            T0)
            if Nrch == 2:
                Zxy1, Zxy2 = gpu_route_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
                                                  Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
                                                  Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda,
                                                  WinSampl, e,
                                                  T0)
        else:
            if Nrch == 1:
                Zxy1 = route_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0,
                                        Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
                                        Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl, e, T0)
            if Nrch == 2:
                Zxy1, Zxy2 = route_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint, fizt0,
                                              Rz, betazt0, Tst0, Q, Tr, XYZ_rsa_ts, dxConsort, dVxConsort, Tz, Vrsa,
                                              Hrsa, dugConsort, Tsint, Lrch, speedOfL, t_r_w, Kss, Fs, lamda, WinSampl,
                                              e,
                                              T0)
    if RegimRsa == 1:  # детальный режим
        # подготовка исходных данных для расчета
        Tst = Tst0 + (qst - 1) * Tr  # момент начала
        Inabl = Nr  # число точек
        Tnabl = Nr * Tr  # время общее наблюдения
        q1 = qst  # номер первого периода повторения
        q2 = qst + Nr - 1  # номер последнего периода повторения
        if q2 > Q:
            q2 = Q
            print('Интервал синтезирования выходит за пределы траекторного сигнала')
            QPrint('Интервал синтезирования выходит за пределы траекторного сигнала')

        # основной расчетный цикл ДЛЯ МАКСИМАЛЬНОГО РАСПАРАЛЛЕЛИВАНИЯ
        if GPUCalculationFlag:
            if Nrch == 1:
                Zxy1 = gpu_detail_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0, Rz,
                                             betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz,tauRli, speedOfL,
                                             t_r_w, Kss, Fs, lamda, WinSampl, e, T0, q1, q2, Tst0)
            if Nrch == 2:
                Zxy1, Zxy2 = gpu_detail_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint,
                                                   fizt0, Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, tauRli, speedOfL,
                                                   t_r_w, Kss, Fs, lamda, WinSampl, e, T0, q1, q2, Tst0)
        else:
            if Nrch == 1:
                Zxy1 = detail_big_cycle1(Zxy1, Nxsint, Nysint, Uout01ss, dxsint, dysint, fizt0, Rz,
                                         betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz,tauRli, speedOfL,
                                         t_r_w, Kss, Fs, lamda, WinSampl, e, T0, q1, q2, Tst0)
            if Nrch == 2:
                Zxy1, Zxy2 = detail_big_cycle2(Zxy1, Zxy2, Nxsint, Nysint, Uout01ss, Uout02ss, dxsint, dysint,
                                               fizt0, Rz, betazt0, Tr, XYZ_rsa_ts, dxConsort, Tz, tauRli, speedOfL,
                                               t_r_w, Kss, Fs, lamda, WinSampl, e, T0, q1, q2, Tst0)

    if FlagViewSignal == 1:
        # Трехмерное РЛИ-1
        fig = plt.figure(figsize=(12.8, 9.6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(Zxy1.shape[1]), np.arange(Zxy1.shape[0]))
        ax.plot_surface(X, Y, np.abs(Zxy1) ** StepBright, cmap='pink')
        ax.view_init(10, -60)
        ax.set_xlabel('Широта')
        ax.set_ylabel('Долгота')
        ax.set_title('Трехмерное РЛИ-1. NumRli=' + str(NumRli) + ', степень=' + str(StepBright))

        # Яркостное РЛИ-1
        Vrli0 = np.abs(Zxy1) ** StepBright
        Vrli0 = Vrli0 / np.max(Vrli0)
        Vrli0 = np.flipud(Vrli0)

        fig = plt.figure(figsize=(12.8, 9.6))
        plt.imshow(Vrli0, cmap='gist_gray')
        plt.xlabel('Широта')
        plt.ylabel('Долгота')
        plt.title('Яркостное РЛИ-1. NumRli=' + str(NumRli) + ', степень=' + str(StepBright))

        if Nrch == 2:
            # Трехмерное РЛИ-2
            fig = plt.figure(figsize=(12.8, 9.6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, np.abs(Zxy2) ** StepBright, cmap='pink')
            ax.view_init(10, -60)
            ax.set_xlabel('Широта')
            ax.set_ylabel('Долгота')
            ax.set_title('Трехмерное РЛИ-2. NumRli=' + str(NumRli) + ', степень=' + str(StepBright))

            # Яркостное РЛИ-2
            Vrli0 = np.abs(Zxy2) ** StepBright
            Vrli0 = Vrli0 / np.max(Vrli0)
            Vrli0 = np.flipud(Vrli0)

            fig = plt.figure(figsize=(12.8, 9.6))
            plt.imshow(Vrli0, cmap='gist_gray')
            plt.xlabel('Широта')
            plt.ylabel('Долгота')
            plt.title('Яркостное РЛИ-2. NumRli=' + str(NumRli) + ', степень=' + str(StepBright))

        plt.show()

    # Вывод максимумов и разности фаз
    Zxy1max = np.max(np.abs(Zxy1))
    nqmax = np.argmax(np.abs(Zxy1))
    nmax1 = nqmax % Nxsint
    qmax1 = nqmax // Nxsint + 1
    print(
        f'Zxy1({nmax1},{qmax1})={Zxy1max} x={(-Nxsint + 1) / 2 + nmax1 - 1 * dxsint} y={(-Nysint + 1) / 2 + qmax1 - 1 * dxsint}')
    QPrint(
         f'Zxy1({nmax1},{qmax1})={Zxy1max} x={(-Nxsint + 1) / 2 + nmax1 - 1 * dxsint} y={(-Nysint + 1) / 2 + qmax1 - 1 * dxsint}')
    if Nrch == 2:
        Zxy2max = np.max(np.abs(Zxy2))
        nqmax = np.argmax(np.abs(Zxy2))
        nmax2 = nqmax % Nxsint
        qmax2 = nqmax // Nxsint + 1
        print(
            f'Zxy2({nmax2},{qmax2})={Zxy2max} x={(-Nxsint + 1) / 2 + nmax2 - 1 * dxsint} y={(-Nysint + 1) / 2 + qmax2 - 1 * dxsint}')
        QPrint(
             f'Zxy2({nmax2},{qmax2})={Zxy2max} x={(-Nxsint + 1) / 2 + nmax2 - 1 * dxsint} y={(-Nysint + 1) / 2 + qmax2 - 1 * dxsint}')
        print(f'Разность фаз={np.angle(Zxy1[nmax1, qmax1]) - np.angle(Zxy2[nmax2, qmax2])}')
        QPrint(f'Разность фаз={np.angle(Zxy1[nmax1, qmax1]) - np.angle(Zxy2[nmax2, qmax2])}')

    # Оценка отношения сигнал/шум
    Psh = np.sum(np.abs(Zxy1) ** 2) / float(Nxsint) / float(Nysint)
    print(f'ОСШ максимальное={np.max(np.abs(Zxy1) ** 2) / Psh}')
    QPrint(f'ОСШ максимальное={np.max(np.abs(Zxy1) ** 2) / Psh}')
    t = np.datetime64('now')
    print(f'Завершение РЛИ xy {t}')
    QPrint(f'Завершение РЛИ xy {t}')
    # Запись сформированного РЛИ в файл
    # Сначала записывается массив N*Q реальных значений, потом массив мнимых значений;
    # общая размерность 2*2N*Q байт
    if FlagWriteRli == 1:
        print('Запись РЛИ в файл')
        QPrint('Запись РЛИ в файл')
        # первый канал
        Zxy = np.zeros((2 * Nxsint, Nysint), dtype=np.float64)
        Zxy[:Nxsint, :] = np.real(Zxy1[:Nxsint, :Nysint])
        Zxy[Nxsint:2 * Nxsint, :] = np.imag(Zxy1[:Nxsint, :Nysint])
        # формируем имя файла путем замены 'Yts1' в начале на Rli1 с теми же значениями даты и времени
        filename = 'Yts1-14-May-2023 19:38:28'
        filename = Yts1Path.split('/')[-1]
        filename = 'RL11' + filename[4:]
        fileID = open(filename, 'wb')
        if fileID.closed:
            raise Exception('Текущая папка защищена. Смените текущую папку')
        fileID.write(Zxy.tobytes())
        fileID.close()
        # второй канал
        if Nrch == 2:
            # первый канал
            Zxy[:Nxsint, :] = np.real(Zxy2[:Nxsint, :Nysint])
            Zxy[Nxsint:2 * Nxsint, :] = np.imag(Zxy2[:Nxsint, :Nysint])
            # формируем имя файла путем замены 'Yts1' в начале на Rli1 с теми же значениями даты и времени
            filename = 'Yts1-14-May-2023 19:38:28'
            filename = 'RL12' + filename[4:]
            fileID = open(filename, 'wb')
            if fileID.closed:
                raise Exception('Текущая папка защищена. Смените текущую папку')
            fileID.write(Zxy.tobytes())
            fileID.close()

    print('РЛИ записаны в файл')
    QPrint('РЛИ записаны в файл')
    t = np.datetime64('now')
    print(f'Завершение РЛИ xy {t}')
    QPrint(f'Завершение РЛИ xy {t}')