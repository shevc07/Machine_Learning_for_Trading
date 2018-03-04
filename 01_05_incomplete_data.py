import pandas as pd
import matplotlib.pyplot as plt
import time
import os


def symbol_to_path(symbol, basedir='data'):
    return os.path.join(basedir, '{}.csv'.format(symbol))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)

    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        dftmp = pd.read_csv(symbol_to_path(symbol),
                            index_col='Date',
                            parse_dates=True,
                            usecols=['Date', 'Adj Close'],
                            na_values='nan')

        dftmp = dftmp.rename(columns={'Adj Close': symbol})

        df = df.join(dftmp, how='left')
        if 'SPY' == symbol:
            df = df.dropna(subset=['SPY'])

    return df


def plot_data(df, title='Stock Prices'):
    ax = df.plot(title=title, fontsize=12, linewidth=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def fill_missing_values(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)


def p06_using_fillna():
    dates = pd.date_range(start='2010-01-01', end='2012-12-31')
    symbols= ['FAKE_AMZN']
    df = get_data(symbols, dates)
    fill_missing_values(df)
    plot_data(df)


def test_run():
    p06_using_fillna()


if __name__ == '__main__':
    test_run()