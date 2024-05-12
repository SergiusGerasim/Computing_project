import methods
import numpy as np
import pandas as pd
from time import time
from tqdm import trange
from numba import njit
import plotly.graph_objects as go

'''
Код демонстрирует колебательную динамику мат. модели Орегонатора реакции Белоусова-Жаботинского.
Соответственно на графике ожидается увидеть колебания. Графики рисуются по всем осям.
Коэффициенты k1, k2, k3, k4 отвечают за скорость реакции, переменные y1, y2, y3 - концентрации веществ.
Работа ускорена с помощью библиотеки numba.
На данный момент время работы кода - 30s.
'''

@njit
def oregonator(t, Y, k1, k2, k3, k4):      #система из Воропаевой с.132
    return np.array([k1 * (Y[1] - Y[1] * Y[0] + Y[0] - k2 * Y[0]**2),
                    k3 * (-Y[1] - Y[0]*Y[1] + Y[2]),
                    k4 * (Y[0] - Y[2])])

@njit
def oregonator_2(t, Y, f=1, q= 7.62e-5, epsilon= 9.9e-3, epsilon_1= 1.98e-5):      # отмасштабированная система (= без параметров)
    return np.array([(q * Y[1] - Y[0] * Y[1] + Y[0]*(1 - Y[0])) / epsilon,         # можно потестить, если будет энтузиазм, но по-умолчанию не надо
                      (-q * Y[1] - Y[0] * Y[1] + f * Y[2]) / epsilon_1,
                      Y[0] - Y[2]])

def make_plot_economy(step, a, b, method, y_zero, dimension, current_system):      # график решения, которое нашел method
    y_array = method(step, a, b, y_zero, current_system, dimension, filter=1000)
    x_array = np.arange(a, b, step  * 10**3)
    print(y_array)
    data = [go.Scatter(x=x_array, y=y_array[:, 0], name='axe 1'), go.Scatter(x=x_array, y=y_array[:, 1], name='axe 2'),
            go.Scatter(x=x_array, y=y_array[:, 2], name='axe 3')]
    fig = go.Figure(data)
    fig.update_layout(title=f'{step}, seg.[{x_array[0]}, {x_array[-1]}]')      # на графике в заголовке выодится только шаг и отрезок
    fig.update_yaxes(type="log")                                               # параметры и нач. условия не показаны
    fig.show()

def runge_error(points_number, method, a, b, y_zero, current_system, dimension, error= 1, iters= 0, factor= 2, tol= 1e-2):      # ошибка по Рунге (до заданной точности tol)
    step = (b - a) / points_number                                                                                       
    solution_last = method(step, a, b, y_zero, current_system, dimension)
    result = {'error': [], 'step': []}
    while error > tol:      # tol возможно надо уменьшить
        iters += 1
        points_number *= factor
        step = (b - a) / points_number
        solution_current = method(step, a, b, y_zero, current_system, dimension)
        error = np.max(np.abs(solution_last - solution_current[::factor]))
        result['error'].append(error)
        result['step'].append(step)
        solution_last = np.copy(solution_current)
    result = pd.DataFrame(result)      # возвращает dataframe: ошибка и соответствующий шаг
    return result

def get_time(method, step, a, b, y_zero, current_system, dimension, repeats=1):      # время работы method
    start_time = time()
    for _ in trange(repeats):
        method(step, a, b, y_zero, current_system, dimension)
    print('{:f}'.format((time() - start_time) / repeats))

@njit
def current_system(x,y):      # возвращает систему с заданными коэффициентами (написано не очень грамотно из-за особенностей numba, но работает)
    return oregonator(x, y, k1, k2, k3, k4)

k1 = 77.27      # параметры (сейчас - из Воропаевой, но советую поставить k2 = 8.375 * 1e-5 для наглядности)
k2 = 8.375 * 1e-6
k3 = 1 / k1
k4 = 0.161

y1_0 = 4      #начальные условия (сейчас - из Воропаевой, можно для более приятной картинки поставить (1.1, 4, 1.1))
y2_0 = 1.1
y3_0 = 4
y_zero = np.array([y1_0, y2_0, y3_0])

t1 = 0      # отрезок, достаточный, чтообы увидеть колебания (можно 300, если вы поставили k2 = 8.375 * 1e-5)
t2 = 500

dimension = 3      # кол-во уравнений
step = 1e-5      # шаг (меньше можно не делать вроде как, можно чуть больше по типу 2e-5 для ускорения работы)
points_number = int((t2 - t1) / step)

method = methods.runge_kutta_method_economy      # для смены метода достаточно менять эту строчку

make_plot_economy(step, t1, t2, method, y_zero, dimension, current_system)
runge_error(points_number, method, t1, t2, y_zero, current_system, dimension)
get_time(method, step, t1, t2, y_zero, current_system, dimension)
