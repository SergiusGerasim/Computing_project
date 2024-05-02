import numpy as np

def euler_method(x_array, y_zero, function, dimension):
    step = x_array[1] - x_array[0]
    size = x_array.shape[0]
    y_array = np.empty((size, dimension))
    y_array[0] = y_zero
    for i in range(1, size):
        y_array[i] = y_array[i - 1] + step * function(x_array[i - 1], y_array[i - 1])
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

