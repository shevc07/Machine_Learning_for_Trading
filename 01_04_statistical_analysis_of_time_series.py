import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def symbol_to_path(symbol, basedir='data'):
    return os.path.join(basedir, "{}.csv".format(symbol))

def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)

    if 'SPY' not in symbols:
        symbols.index(0, 'SPY')

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


def plot_data(df, title="stock prices"):
    ax = df.plot(title=title, fontsize=12, linestyle='-')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def p04_compute_global_statistics():
    dates = pd.date_range(start='2010-01-01', end='2012-12-31')
    symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
    df = get_data(symbols, dates)
    plot_data(df)
    print('mean:\n{}'.format(df.mean()))
    print('median: \n{}'.format(df.median()))
    print('std: \n{}'.format(df.std()))


def p08_computing_rolling_statistics():
    dates = pd.date_range(start='2010-01-01', end='2010-09-01')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    # plot_data(df)
    ax = df['SPY'].plot(title='SPY rolling mean', label='SPY')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    rm_ax = df['SPY'].rolling(window=20).mean().plot(label='SPY mean', ax=ax)
    ax.legend(loc='upper left')
    plt.show()


def get_rolling_mean(values, window):
    return values.rolling(window=window).mean()


def get_rolling_std(values, window):
    return values.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band


def p09_calculate_bollinger_bands():
    dates = pd.date_range(start='2010-01-01', end='2012-09-01')
    symbols = ['SPY']
    df = get_data(symbols,dates)
    ax = df['SPY'].plot(title='SPY Bollinger Bands', label='SPY')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    rm = get_rolling_mean(df['SPY'],20)
    rstd = get_rolling_std(df['SPY'],20)
    rm.plot(label='SPY rolling mean', ax=ax)
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)
    ax.legend(loc='upper left')
    plt.show()


def compute_daily_returns(df):
    daily_returns = df.copy()
    # daily_returns[1:] = df[1:].values / df[:-1].values - 1
    daily_returns = df / df.shift(1) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def p11_compute_daily_returns():
    dates = pd.date_range(start='2010-01-01', end='2010-09-01')
    symbols = ['SPY', 'XOM']
    df = get_data(symbols, dates)
    df_dr = compute_daily_returns(df)

    plot_data(df_dr, title='daily returns')


def compute_cumulative_returns(df):
    cumulative_returns = df.copy()
    cumulative_returns = df / df.ix[0,0] - 1
    # cumulative_returns.ix[0,:] = 0
    return cumulative_returns


def p12_compute_cumulative_returns():
    dates = pd.date_range(start='2010-01-01', end='2010-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    df_cr = compute_cumulative_returns(df)

    plot_data(df_cr, title='cumulative returns')


def test_run():
    # p04_compute_global_statistics()
    # p08_computing_rolling_statistics()
    # p09_calculate_bollinger_bands()
    # p11_compute_daily_returns()
    p12_compute_cumulative_returns()


if __name__ == '__main__':
    test_run()