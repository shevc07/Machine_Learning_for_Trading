import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo


def f(X):
    Y = (X - 1.5)**2 + 0.5
    print("X: {}, Y: {}".format(X, Y))
    return Y


def p03_minimizer_in_python():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP', options={'disp': True})
    print("Minima found at: X:{}, Y:{}".format(min_result.x, min_result.fun))


def error_func(line, data):
    err = np.sum((data[:, 1] - (line[0]*data[:, 0] + line[1]))**2)
    return err


def fit_line(data, error_func):
    l = np.float32([0, np.mean(data[:, 1])])
    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="initial line")

    result = spo.minimize(error_func, l, args=(data, ), method='SLSQP', options={'disp': True})
    return result.x


def p09_fit_a_line_to_given_data_points():
    l_orig = np.float32([4, 2])
    print("Origin line: C0={}, C1={}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0]*Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    l_fit = fit_line(data, error_func)
    print("Fitted line: C0={}, C1={}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0]*data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label="fit line")

    plt.legend(loc="upper left")
    plt.show()


def error_poly(C, data):
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err


def fit_poly(data, error_func, degree=3):
    Cguess = np.poly1d(np.ones(degree+1, dtype=np.float32))

    x=np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="initial guess")

    result = spo.minimize(error_func, Cguess, args=(data, ), method='SLSQP', options={'disp': True})

    return np.poly1d(result.x)



def test_run():
    p09_fit_a_line_to_given_data_points()


if __name__ == '__main__':
    test_run()