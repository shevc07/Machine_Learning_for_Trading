import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def symbol_to_path(symbol, basedir='data'):
    return os.path.join(basedir, "{}.csv".format(symbol))


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


def plot_data(df, title="stock prices", xlabel='Date', ylabel='Price'):
    ax = df.plot(title=title, fontsize=12, linestyle='-', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    plt.show()