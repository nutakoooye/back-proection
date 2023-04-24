# набор скриптов для моделирования РСА
# Назначение и порядок использования:
# main - задание пространственной ситуации, параметров РЛС, координат блестящих точек
# CalcSigShortTime - рассчет отраженного сигнала восстановлением из сжатого
# CalcRliXYss-1 - расчет РЛИ в координатах (x,y) с маршрутном режиме с передискретизацией
# ChirpSig - расчет зондирующего сигнала и импульсной характеристики
import math
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from calc_sig_short_time import (range_calculation,
                                 range_calculation_2_chanels,
                                 restore_signale_spectrum)
from calc_rli_xy_ss import (add_zeros,
                            add_zeros_2_channels,
                            big_cycle1,
                            big_cycle2)
from gpu_rli_xy_ss import big_cycle1

time_prev= time.time()
# общие переменные
c = 3 * 10 ** 8
df = 25 * 10 ** 6  # ширина спектра зондирующего сигнала *********
Ks = 1  # коэффициент Найквиста для исходного сигнала
Kss = 1  # коэффициент передискретизации *****
T0 = 10 ** -4  # длительность импульса
Fs = Ks * df  # частота дискретизации
T = 4 * T0  # длительность моделируемого участка
N = int(T * Fs)  # число отчетов в пределах периода повторения
N = 2 ** math.ceil(math.log2(N))  # приводим число отсчетов к основанию 2
b = math.pi * df / T0  # скорость изменения круговой частоты пополам
mu = df / T0  # скорость изменения циклической частоты
h0 = np.zeros(N, "complex64")  # импульсная характеристика фильтра ВПО
lamda = 0.03  # длина волны
f0 = c / lamda  # несущая частота

Tn = 1.0  # общее время наблюдения
Tr = 0.25 * 10 ** -2  # период повторения ******
Q = int(Tn / Tr)  # число импульсов в пачке
Tsint = 0.4  # время синтезирования

Prsa = 5000  # мощность излучения
Grsa = 2.5e04  # коэффициент усиления антенны
N0 = 1e-20  # спектральная плотность мощности внутренних шумов
Pnoise = N0 * df  # мощность внутренних шумов
alfa = 0.5 * math.pi / 180  # ширина ГЛ ДН по азимуту
beta05 = 2.5 * math.pi / 180  # ширина ГЛ ДН по углу места
Nadc = 10  # число разрядов АЦП
dacp = 3 * math.sqrt(Pnoise / 2) / (2 ** 4)  # выделяем на шум 4 разряда АЦП

# параметры моделирования отраженного сигнала
gamma = 5  # главный и до четырех боковых лепестков

# FlagInter=0 - расчет ТС для совмещенного излучающего и приемного канала,
# координаты котор`ого в Xrls(1),Yrls(1),Zrls(1)
# FlagInter=1 - расчет ТС для излучающего канала и двух приемных каналов, 
# координаты передатчика в Xrls(1),Yrls(1),Zrls(1), координаты приемников
# Xrls(:3),Yrls(2:3),Zrls(2:3)
FlagInter = 0
L = 6  # расстояние между приёмными каналами

# скорость РСА
Vxrls = 0 * (10 ** 3)
Vyrls = 7612
Vzrls = 0

Vrls = math.sqrt(Vxrls ** 2 + Vyrls ** 2 + Vzrls ** 2)
Axrls = 0
Ayrls = 0
Azrls = 0
# координаты РСА: первое значение - фазового центра передающей антенны, 
# второй и третье значения - фазовых центров приемных антенны  
X0rls = 0
Y0rls = -Q * Tr * Vrls / 2
Z0rls = 500 * 10 ** 3

Xrls = np.zeros(3)
Xrls[0] = X0rls
Xrls[1] = X0rls
Xrls[2] = X0rls

Yrls = np.zeros(3)
Yrls[0] = Y0rls
Yrls[1] = Y0rls - L / 2
Yrls[2] = Y0rls + L / 2

Zrls = np.zeros(3)
Zrls[0] = Z0rls
Zrls[1] = Z0rls
Zrls[2] = Z0rls

# координаты стрелки с отражателями
# x=[700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3700*10**3-10700*10**3-20700*10**3-30700*10**3-40700*10**3+10700*10**3+20700*10**3+30700*10**3+40]
# y=[0255075100125150175200225250275300275250225200275250225200]
# z=[000000000000000000000]

# координаты далёких неподвижных точек
# Nx=10
# Ny=10
# x=zeros(Nx*Ny,1)
# y=zeros(Nx*Ny,1)
# z=zeros(Nx*Ny,1)
# x0=700*10**3
# y0=0
# z0=0
# dx=50
# dy=50
# for i=1:Nx
#     for k=1:Ny
#         n=(i+(k-1)*Nx)
#         x(n)=x0-Nx*dx/2+i*dx
#         y(n)=y0-Ny*dy/2+k*dy
#     end    
# end

# figure scatter(x,y)

# координаты блестящих точек на земной поверхности
x = np.array([701021, 701024, 701021, 701024])
y = np.array([0, 30, 60, 90])
z = np.array([0, 0, 0, 0])

# ЭПР блестящих точек
SigBt = np.zeros(len(x), np.float64)
SigBt[0] = 1000
SigBt[1] = 0
SigBt[2] = 0
SigBt[3] = 0

FiBT = np.random.uniform(0, 2 * np.pi, (len(x)))
K = len(SigBt)
# скорости блестящих точек
Vx = np.zeros(len(x), "float")
Vy = np.zeros(len(x), "float")
Vz = np.zeros(len(x), "float")

Vx[0] = 0
Vy[0] = 0

Vx[1] = 0
Vy[1] = 0

Vx[2] = 0
Vy[2] = 0

Vx[3] = 0
Vy[3] = 0


# для всех БТ рассчитываваем мощность отраженного сигнала на для t=0
Ps = np.zeros(K)
for k in range(K):
    d = math.sqrt((x[k] - Xrls[0]) ** 2 + (y[k] - Yrls[0]) ** 2 + (z[k] - Zrls[0]) ** 2)
    Ps[k] = Prsa * Grsa ** 2 * lamda ** 2 * SigBt[k] / ((4 * math.pi) ** 3 * d ** 4)
print(Ps)


def ChirpSig(t, tx, b):
    win = np.logical_and(t>=0, t<=tx)
    comp = win*np.exp(1j*complex(b * (t ** 2)))
    return comp


# импульсная характеристика СФ
for n in range(N):
    h0[n] = np.conj(ChirpSig(T0 - (n) / Fs, T0, b))


Gh0 = np.fft.fft(h0)
Gh0 = Gh0 / math.sqrt(N)  # КЧХ СФ

# начальное время задержки в 1-ом моменте
tz0 = 2 * math.sqrt((Xrls[0] - x[0]) ** 2 + (Yrls[0] - y[0]) ** 2 + (Zrls[0] - z[0]) ** 2) / c
t2 = tz0 - 5 * 10 ** -6

# формирование траекторного сигнала
print('Расчет дальностей ', datetime.datetime.now())

# расчет для одного совмещенного канала
if FlagInter == 0:
    # расчет дальностей
    U01comp = range_calculation(K, Q, Tr, x, y, z, Vx,
                      Vy, Vz, Xrls, Yrls, Zrls,
                      Vxrls, Vyrls, Vzrls, df,
                      T0, SigBt, N, c, t2, Fs,
                      gamma, lamda, FiBT)

# расчет для передающего и двух приемных каналов
if FlagInter == 1:
    U01comp, U02comp = range_calculation_2_chanels(K, Q, Tr, x, y, z, Vx,
                      Vy, Vz, Xrls, Yrls, Zrls,
                      Vxrls, Vyrls, Vzrls, df,
                      T0, Ks, Ps, N, c, t2, Fs,
                      gamma, lamda, FiBT)


#  вычисляем спектр сжатого сигнала приемном канале
Y001 = np.fft.fft2(U01comp)  # БПФ сжатого сигнала в 1-ом пк
if FlagInter == 1:
    Y002 = np.fft.fft2(U02comp)  # БПФ сжатого сигнала во 2-ом пк


# восстанавливаем спектр сигнала на входе согласованного фильтра
if FlagInter == 0:
    Y001, _ = restore_signale_spectrum(Gh0, Y001, FlagInter, Q, Ks)
if FlagInter == 1:
    Y001, Y002 = restore_signale_spectrum(Gh0, Y001, FlagInter, Q, Ks, Y002)


# восстановленный траекторный  сигнал
Y01v = np.fft.ifft2(Y001)
if FlagInter == 1:
    Y02v = np.fft.ifft2(Y002)

# добавляем шум
SigNoise = math.sqrt(Pnoise / 2)  # СКО квадратурных компонентов шума
Y01v = Y01v + SigNoise * (np.random.randn(N, Q) + 1j * np.random.randn(N, Q))

Y01 = np.fft.fft2(Y01v)  # спектр восстановленного сигнала с шумом
if FlagInter == 1:  # если интерферометрический режим
    Y02v = Y02v + SigNoise * (np.random.randn(N, Q) + 1j * np.random.randn(N, Q))
    Y02 = np.fft.fft2(Y02v)  # спектр восстановленного сигнала с шумом

# квантование траекторного сигнала в соответствии с разрядностью АЦП
Y01v = Y01v / dacp
if FlagInter == 1:
    Y02v = Y02v / dacp
print('Сигнал рассчитан ', datetime.datetime.now())

# формирование РЛИ в координатах (x,y) c передискретизацией
# CalcRliXYss_1

# Построение РЛИ в координатах (x,y) для увеличенной частоты дискретизации
# Вход: массивы Y001,Y002 размерностью N,Q со СПЕКТРАМИ траекторных
# сигналов; N - число отсчетов по дальности; Q - число периодов повторения
# При ВПО проводится увеличение частоты дискретизации траекторного сигнала за счет
# дополнения спектра нулями; вычисляется ИХ и КЧХ согласованного фильтра ВПО с
# учетом оконной функции по дальности, перемножение спектра и КЧХ,
#  обратное ДПФ.
# Далее проводится синтез для увеличенной частоты дискретизации, для чего
# с заданной дискретностью просматриваюися все точки заданной зоны синтезирования.
# Для каждой точки выбирается симметричный интервал синтезирования относительно траверса
# Для каждого положения РСА в зоне синтезирования проводится
# вычисление дальности до точки, индекс сжатого сигнала и проводится
# суммирование этих сигналов с компенсацией изменения фазы.
# Приемущества: нет необходимости в последующих преобразованиях из
# координат "наклонная дальность-поперечная дальность" в координаты (x,y)

# задаем зону синтезирования
dxsint = 3  # дискретность по Ox
dysint = 3  # дискретность по Oy
# размеры и число точек по Ox
X1sint = np.min(x) - 200
X2sint = np.max(x) + 200
Nxsint = int((X2sint - X1sint) / dxsint)
# размеры и число точек по Oy
Y1sint = -1000
Y2sint = 1000
Nysint = int((Y2sint - Y1sint) / dysint)
# оконные функции:
# 1 - прямоугольное окно
# 2 - косинус на пъедестале, при delta=0.54 - Хэмминга (-42.7 дБ)
# 3 - косинус квадрат на пъедестале
# 4 - Хэмминга (-42.7 дБ)
# 5 - Хэннинга в третье степени на пъедестале (-39.3 дБ)
# 6 - Хэннинга в четвертой степени на пъедестале (-46.7 дБ)
# 7,8,9 - Кайзера-Бесселя для  alfa=2.7; 3.1; 3.5 (-62.5; -72ю1; -81.8 дБ)
# 10 - Блекмана-Херриса (-92 дБ)
TypeWinDn = 1  # тип оконной функции при сжатии по наклонной дальности
TypeWinDp = 1  # тип оконной функции при сжатии по поперечной дальности

###################   ВПО с передискретизацией  ##########################
# передискретизация по наклонной дальности - дополнение спектра нулями

# новый вариант передискретизации - просто добавляем нули
if FlagInter == 0:
    Y01ss = add_zeros(Q, N, Kss, Y01)
if FlagInter == 1:
    Y01ss, Y02ss = add_zeros_2_channels(Q, N, Kss, Y01, Y02)

# вычисление КЧХ фильтра с увеличенным числом отсчетов с учетом оконной функции 
h0ss = np.zeros(N * Kss, complex)
for n in range(int(T0 * Fs * Kss + 1)):
    h0ss[n] = np.conj(ChirpSig(T0 - (n - 1) / (Fs * Kss), T0, b)) * 1

Gh0ss = np.fft.fft(h0ss)
Gh0ss = Gh0ss / math.sqrt(N * Kss)

# сжатие по дальности по 1-му приемному каналу
Goutss = np.zeros((N * Kss, Q), complex)

for q in range(Q):
    Goutss[:, q] = Y01ss[:, q] * Gh0ss


Uout01ss = np.fft.ifft2(Goutss) * np.sqrt(N * Kss)

# сжатие по дальности по 2-му приемному каналу

if FlagInter == 1:
    Goutss = np.zeros((N * Kss, Q), complex)
    for q in range(Q):
        Goutss[:, q] = Y02ss[:, q] * Gh0ss

Uout02ss = np.fft.ifft2(Goutss) * np.sqrt(N * Kss)

######## сжатие в координатах (x,y) способом Backprojection  ###############
# отсчеты накопленного комплексного РЛИ-1
Zxy1 = np.zeros((Nxsint, Nysint), complex)

# подготовка значений оконной функции
Nr = int(Tsint / Tr)  # число периодов повторения на интервале синтезирования
WinSampl = np.ones(Nr + 1)

print("START BIG CYCLE")
# основной расчетный цикл ДЛЯ МАКСИМАЛЬНОГО РАСПАРАЛЛЕЛИВАНИЯ
if FlagInter == 0:
    Zxy1 = big_cycle1(X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, lamda)
if FlagInter == 1:
    Zxy2 = np.zeros((Nxsint, Nysint), complex)
    Zxy1, Zxy2 = big_cycle2(Nxsint, Nysint, X1sint, Y1sint, dxsint, dysint, Xrls, Yrls, Zrls,
               Vxrls, Vyrls, Vzrls, Tsint, Tr, Vrls, L, c, t2, T0, Kss,
               Uout01ss, Fs, WinSampl, Zxy1, Zxy2, lamda, Uout02ss)


fig = plt.figure(figsize=(12.8,9.6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, Nxsint, 1)
y = np.arange(0, Nysint, 1)
X, Y = np.meshgrid(y, x)

surf = ax.plot_surface(X, Y, np.abs(Zxy1)**2, cmap='viridis')
ax.view_init(10, 30)
plt.xlabel('Oy')
plt.ylabel('Ox')
plt.title('РЛИ по первому каналу')
plt.show()

if FlagInter == 1:
    fig = plt.figure(figsize=(12.8,9.6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, np.abs(Zxy1-Zxy2)**2, cmap='viridis')
    ax.view_init(10, 30)
    plt.xlabel('Oy')
    plt.ylabel('Ox')
    plt.title('разностное РЛИ')
    plt.show()

    fig = plt.figure(figsize=(12.8,9.6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, np.abs(Zxy2) ** 2, cmap='viridis')
    ax.view_init(10, 30)
    plt.xlabel('Oy')
    plt.ylabel('Ox')
    plt.title('РЛИ по первому каналу')
    plt.show()