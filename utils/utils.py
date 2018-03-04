import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def fill_missing_values(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)


def symbol_to_path(symbol, basedir='data'):
    return os.path.join(basedir, "{}.csv".format(symbol))


def normed(df):
    return df/df.ix[0, :]

def compute_daily_returns(df):
    daily_returns = df.copy()
    # daily_returns[1:] = df[1:].values / df[:-1].values - 1
    daily_returns = df / df.shift(1) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def get_data_universal(symbols, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(index=dates)

    symbols.insert(0, 'SPY_FOR_STANDARDIZE')

    for symbol in symbols:
        dftmp = pd.read_csv(symbol_to_path(symbol),
                            index_col="Date",
                            parse_dates=True,
                            usecols=['Date', 'Adj Close'],
                            na_values='nan')
        dftmp = dftmp.rename(columns={'Adj Close': symbol})
        df = df.join(dftmp, how='left')
        if 'SPY_FOR_STANDARDIZE' == symbol:
            df = df.dropna(subset=['SPY_FOR_STANDARDIZE'])

    df.drop(columns=['SPY_FOR_STANDARDIZE'], inplace=True)

    fill_missing_values(df)

    return df


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)

    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        dftmp = pd.read_csv(symbol_to_path(symbol),
                            index_col="Date",
                            parse_dates=True,
                            usecols=['Date', 'Adj Close'],
                            na_values='nan')
        dftmp = dftmp.rename(columns={'Adj Close': symbol})
        df = df.join(dftmp, how='left')
        if 'SPY' == symbol:
            df = df.dropna(subset=['SPY'])

    return df


def plot_data(df, title="stock prices", xlabel='Date', ylabel='Price', save=False):
    ax = df.plot(title=title, fontsize=12, linestyle='-', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    if save:
        plt.savefig(title)
    plt.show()

def compute_cumulative_return(port_vals):
    """
    累计收益
    :param port_vals:
    :return:
    """
    return port_vals[-1] / port_vals[0] - 1


def compute_sharpe_ratio(daily_rets, rf=0, freq='daily'):
    """
         mean(Rp - Rf)
    SR = -------------
            std(Rp)

    sharpe ratio is an annual measure, so:
    SR_annualized = K * SR

    K = square root of #samples of year

    daily K = square root of 252
    weekly K = square root of 52
    monthly K = square root of 12

    risk free rate:
    1. LIBOR: London Interbank Offer Rate 伦敦银行同业拆放利率
    2. 3mo T-bill: 3 month Treasury bill 3月期国债利率
    3. 0%

    FOR EXAMPLE, cacluate SR using daily data:

                   mean ( daily_rets - daily_risk_free_rate )
    sharpe_ratio = ------------------------------------------
                    std ( daily_rets - daily_risk_free_rate )

    SR = sqrt(252) * (daily_rets - daily_rf).mean() / daily_rets.std()

    daily risk free rate traditional shortcut:
    daily_rf = 252nd root of (1 + Rf)

    :return:
    """
    freqs = {'daily': 252, 'weekly': 52, 'monthly': 12, 'yearly': 1}
    f = freqs[freq]
    k = math.sqrt(f)

    daily_rf = math.pow(1+rf, 1/f) - 1

    sr = k * (daily_rets - daily_rf).mean() / daily_rets.std()

    return sr