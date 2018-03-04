import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from utils import get_data
from utils import plot_data
from utils import symbol_to_path


def compute_daily_returns(df):
    dr = df.copy()
    dr[1:] = df / df.shift(1) - 1
    dr.ix[0,:] = 0
    return dr


def p05_how_to_plot_a_histogram():
    dates = pd.date_range('2010-01-01', '2018-03-01')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    plot_data(df)
    dr = compute_daily_returns(df)
    plot_data(dr, title='Daily Returns of {}'.format(symbols), xlabel='Date', ylabel='ratio')

    dr.hist(bins=200)
    plt.show()

def p06_computing_histogram_statistics():
    dates = pd.date_range('2010-01-01', '2018-03-01')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    # plot_data(df)
    dr = compute_daily_returns(df)
    # plot_data(dr, title='Daily Returns of {}'.format(symbols), xlabel='Date', ylabel='ratio')

    dr.hist(bins=20)
    # plt.show()

    mean = dr['SPY'].mean()
    std = dr['SPY'].std()
    kurtosis = dr.kurtosis()
    print("mean: {}, std: {}, kurtosis: {}".format(mean, std, kurtosis))

    plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)

    plt.show()


def p08_plot_tow_histograms_together():
    dates = pd.date_range('2010-01-01', '2018-03-01')
    symbols = ['SPY','XOM']
    df = get_data(symbols, dates)
    # plot_data(df)
    dr = compute_daily_returns(df)
    # plot_data(dr, title='Daily Returns of {}'.format(symbols), xlabel='Date', ylabel='ratio')

    dr['SPY'].hist(bins=50, label='SPY')
    dr['XOM'].hist(bins=50, label='XOM')

    plt.show()


def p13_scatterplots_in_python():
    dates = pd.date_range('2010-01-01', '2018-03-01')
    symbols = ['SPY','XOM', 'GLD']
    df = get_data(symbols, dates)
    # plot_data(df)
    dr = compute_daily_returns(df)
    # plot_data(dr, title='Daily Returns of {}'.format(symbols), xlabel='Date', ylabel='ratio')

    dr.plot(kind='scatter', x='SPY', y='XOM')
    beta, alpha = np.polyfit(dr['SPY'], dr['XOM'], 1)
    print('beta: {}, alpha: {}'.format(beta, alpha))
    plt.plot(dr['SPY'], beta*dr['SPY'] + alpha, linestyle='-', color='r')
    plt.show()

    dr.plot(kind='scatter', x='SPY', y='GLD')
    beta, alpha = np.polyfit(dr['SPY'], dr['GLD'], 1)
    print('beta: {}, alpha: {}'.format(beta, alpha))
    plt.plot(dr['SPY'], beta*dr['SPY'] + alpha, linestyle='-', color='r')
    plt.show()

    # calculate correlation coefficient
    print(dr.corr(method='pearson'))


def test_run():
    p13_scatterplots_in_python()


if __name__ == '__main__':
    test_run()