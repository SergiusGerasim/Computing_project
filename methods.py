import numpy as np

def euler_method(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_array = np.empty((size, dimension))
    y_array[0] = y_zero
    for i in range(1, size):
        y_array[i] = y_array[i - 1] + step * function(x_array[i - 1], y_array[i - 1])
    return y_array

def euler_method_economy(step, a, b, y_zero, function, dimension, filter=10000):
    number = int((b - a) / step)
    y_array = np.empty((number - 1) // filter + 1, dimension)
    y = y_zero
    for i in range(number - 1):
        x = a + step * i
        y = y + step * function(x, y)
        if i % filter == 0:
            y_array[i // filter] = y
    return y_array

def euler_method_recalculation(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_array = np.empty((size, dimension))
    y_array[0] = y_zero
    for i in range(1, size):
        y_array[i] = y_array[i - 1] + step * function(x_array[i - 1], y_array[i - 1])
        y_array[i] = y_array[i - 1] + step * (function(x_array[i - 1], y_array[i - 1]) + function(x_array[i], y_array[i])) / 2
    return y_array

def euler_method_recalculation_economy(step, a, b, y_zero, function, dimension, filter=10000):
    number = int((b - a) / step)
    y_array = np.empty((number - 1) // filter + 1, dimension)
    y = y_zero
    for i in range(number - 1):
        x = a + step * i
        y = y + step * (function(x, y) + function(x + step, y + step * function(x, y))) / 2
        if i % filter == 0:
            y_array[i // filter] = y
    return y_array

def runge_kutta_method(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_array = np.empty((size, dimension))
    y_array[0] = y_zero
    for i in range(1, size):
        fi_1 = step * function(x_array[i - 1], y_array[i - 1])
        fi_2 = step * function(x_array[i - 1] + step / 2, y_array[i - 1] + fi_1 / 2)
        fi_3 = step * function(x_array[i - 1] + step / 2, y_array[i - 1] + fi_2 / 2)
        fi_4 = step * function(x_array[i - 1] + step, y_array[i - 1] + fi_3)
        y_array[i] = y_array[i - 1] + (fi_1 + 2 * fi_2 + 2 * fi_3 + fi_4) / 6
    return y_array

def runge_kutta_method_economy(step, a, b, y_zero, function, dimension, filter=10000):
    number = int((b - a) / step)
    y_array = np.empty((number // filter + 1, dimension))
    y = y_zero
    for i in range(number):
        x = a + step * i
        fi_1 = step * function(x, y)
        fi_2 = step * function(x + step / 2, y + fi_1 / 2)
        fi_3 = step * function(x + step / 2, y + fi_2 / 2)
        fi_4 = step * function(x + step, y + fi_3)
        y = y + (fi_1 + 2 * fi_2 + 2 * fi_3 + fi_4) / 6
        if i % filter == 0:
            y_array[i // filter] = y
    return y_array
    
def adams_bashfort_method(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_start = runge_kutta_method(x_array[:4], y_zero, function, dimension)
    y_array = np.empty((size, dimension))
    y_array[:4] = y_start
    f_array = [function(x_array[0], y_start[0]), function(x_array[1], y_start[1]), 
               function(x_array[2], y_start[2]), function(x_array[3], y_start[3])]
    for i in range(4, size):
        y_array[i] = y_array[i - 1] + (step / 24) * (55 * f_array[3] - 59 * f_array[2] + 37 * f_array[1] - 9 * f_array[0])
        f_array.pop(0)
        f_array.append(function(x_array[i], y_array[i]))
    return y_array

def adams_bashfort_method_economy(step, a, b, y_zero, function, dimension, filter= 10000):
    number = int((b - a) / step)
    y_array = np.empty((number // filter + 1, dimension))
    y_start = runge_kutta_method(np.array([a, a + step, a + step * 2, a + step * 3]), y_zero, function, dimension)
    f_array = [function(a, y_start[0]), function(a + step, y_start[1]), 
               function(a + 2 * step, y_start[2]), function(a + 3 * step, y_start[3])]
    y = y_start[-1]
    k = 0
    for i in range(4, number):
        x = a + step * i
        y = y + (step / 24) * (55 * f_array[(k + 3) % 4] - 59 * f_array[(k + 2) % 4]
                                                      + 37 * f_array[(k + 1) % 4] - 9 * f_array[k])
        f_array[k] = function(x, y)
        k = (k + 1) % 4
        if i % filter == 0:
            y_array[i // filter] = y
    return y_array

def adams_bashfort_molton_method(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_start = runge_kutta_method(x_array[:4], y_zero, function, dimension)
    y_array = np.empty((size, dimension))
    y_array[:4] = y_start
    f_array = [function(x_array[0], y_start[0]), function(x_array[1], y_start[1]), 
               function(x_array[2], y_start[2]), function(x_array[3], y_start[3])]
    for i in range(4, size + 1):
        y_array[i] = y_array[i - 1] + (step / 24) * (55 * f_array[3] - 59 * f_array[2] + 37 * f_array[1] - 9 * f_array[0])
        f_array.pop(0)
        f_array.append(function(x_array[i], y_array[i]))
        y_array[i] = y_array[i - 1] + (step / 24) * (9 * f_array[3] + 19 * f_array[2] - 5 * f_array[1]  + f_array[0])
        f_array[3] = function(x_array[i], y_array[i])
    return y_array

def adams_bashfort_molton_method_economy(step, a, b, y_zero, function, dimension, filter= 10000):
    number = int((b - a) / step)
    y_array = np.empty((number // filter + 1, dimension))
    y_start = runge_kutta_method(np.array([a, a + step, a + step * 2, a + step * 3]), y_zero, function, dimension)
    f_array = [function(a, y_start[0]), function(a + step, y_start[1]), 
               function(a + 2 * step, y_start[2]), function(a + 3 * step, y_start[3])]
    y = y_start[-1]
    k = 0
    for i in range(4, number):
        x = a + step * i
        y = y + (step / 24) * (55 * f_array[(k + 3) % 4] - 59 * f_array[(k + 2) % 4] + 37 * f_array[(k + 1) % 4] - 9 * f_array[k])
        f_array[k] = function(x, y)
        k = (k + 1) % 4
        y = y + (step / 24) * (9 * f_array[(k + 3) % 4] + 19 * f_array[(k + 2) % 4] - 5 * f_array[(k + 1) % 4]  + f_array[k])
        f_array[(k + 3) % 4] = function(x, y)
        if i % filter == 0:
            y_array[i // filter] = y
    return y_array

def gear_method(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_start = runge_kutta_method(x_array[:4], y_zero, function, dimension)
    y_array = np.empty((size, dimension))
    y_array[:4] = y_start
    iteration_number = 4
    for i in range(4, size):
        y_iter = y_array[i - 1]
        for _ in range(iteration_number):
            y_array[i] = (48 * y_array[i - 1] - 36 * y_array[i - 2] + 16 * y_array[i - 3] - 3 * y_array[i - 4] + 12 * step * function(x_array[i], y_iter)) / 25
            y_iter = y_array[i]
    return y_array

def gear_method_economy(step, a, b, y_zero, function, dimension, filter= 10000, iterations= 4):
    number = int((b - a) / step)
    y_array = np.empty((number // filter + 1, dimension))
    y = runge_kutta_method(np.array([a, a + step, a + step * 2, a + step * 3]), y_zero, function, dimension)
    k = 0
    for i in range(4, number):
        y_iter = y[(k + 3) % 4]
        for _ in range(iterations):
            y_iter = (48 * y[(k + 3) % 4] - 36 * y[(k + 2) % 4] + 16 * y[(k + 1) % 4]
                           - 3 * y[k] + 12 * step * function(a + step * i, y_iter)) / 25
        y[k] = y_iter
        if i % filter == 0:
            y_array[i // filter] = y[k]
        k = (k + 1) % 4
    return y_array
