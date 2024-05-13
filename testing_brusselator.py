import methods
import numpy as np
import pandas as pd
from time import time
from tqdm import trange
from numba import njit
import plotly.graph_objects as go

@njit
def oregonator(t, Y, k1, k2, k3, k4):
    return np.array([k1 * (Y[1] - Y[1] * Y[0] + Y[0] - k2 * Y[0]**2),
                    k3 * (-Y[1] - Y[0]*Y[1] + Y[2]),
                    k4 * (Y[0] - Y[2])])

@njit
def oregonator_2(t, Y, f=1, q= 7.62e-5, epsilon= 9.9e-3, epsilon_1= 1.98e-5):
    return np.array([(q * Y[1] - Y[0] * Y[1] + Y[0]*(1 - Y[0])) / epsilon,
                      (-q * Y[1] - Y[0] * Y[1] + f * Y[2]) / epsilon_1,
                      Y[0] - Y[2]])

@njit
def brusselator(t, Y, a, b):
    return np.array([a + Y[0]**2 * Y[1] - b * Y[0] - Y[0],
                     b * Y[0] - Y[0]**2 * Y[1]])

def make_plot_economy(step, a, b, method, y_zero, dimension, current_system):
    y_array = method(step, a, b, y_zero, current_system, dimension, filter=1000)
    x_array = np.arange(a, b, step  * 10**3)
    print(y_array)
    data = [go.Scatter(x=x_array, y=y_array[:, 0], name='axe 1'), go.Scatter(x=x_array, y=y_array[:, 1], name='axe 2')]
            # go.Scatter(x=x_array, y=y_array[:, 2], name='axe 3')]
    fig = go.Figure(data)
    fig.update_layout(title=f'{step}, seg.[{x_array[0]}, {x_array[-1]}]')
    fig.update_yaxes(type="log")
    fig.show()

def runge_error(points_number, method, a, b, y_zero, current_system, dimension, error= 1, iters= 0, factor= 2, tol= 1e-2):
    step = (b - a) / points_number
    solution_last = method(step, a, b, y_zero, current_system, dimension)
    result = {'error': [], 'step': []}
    while error > tol:
        iters += 1
        points_number *= factor
        step = (b - a) / points_number
        solution_current = method(step, a, b, y_zero, current_system, dimension)
        error = np.max(np.abs(solution_last - solution_current[::factor]))
        result['error'].append(error)
        result['step'].append(step)
        solution_last = np.copy(solution_current)
    result = pd.DataFrame(result)
    return result

def get_time(method, step, a, b, y_zero, current_system, dimension, repeats=1):
    start_time = time()
    for _ in trange(repeats):
        method(step, a, b, y_zero, current_system, dimension)
    print('{:f}'.format((time() - start_time) / repeats))

@njit
def current_system(x, y):
    return brusselator(x, y, a, b)

k1 = 77.27
k2 = 8.375 * 1e-4
k3 = 1 / k1
k4 = 0.161

a = 1
b = 1.7

y1_0 = 1
y2_0 = 1.0
# y3_0 = 4
y_zero = np.array([y1_0, y2_0])

t1 = 0
t2 = 60

dimension = 2
step = 1e-4
points_number = int((t2 - t1) / step)

method = methods.runge_kutta_method_economy

make_plot_economy(step, t1, t2, method, y_zero, dimension, current_system)
# runge_error(points_number, method, t1, t2, y_zero, current_system, dimension)
# get_time(method, step, t1, t2, y_zero, current_system, dimension)

solutions = method(step, t1, t2, y_zero, current_system, dimension, filter=100)

data = [go.Scatter(x=solutions[:, 0][:-1], y=solutions[:, 1][:-1])]
            # go.Scatter(x=x_array, y=y_array[:, 2], name='axe 3')]
fig = go.Figure(data)
fig.show()
