""" Build a dataframe in pandas """
import pandas as pd
import matplotlib.pyplot as plt
import os


def symbol_to_path(symbol, basedir='data'):
    return os.path.join(basedir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)

    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        dftmp = pd.read_csv(symbol_to_path(symbol),
                            index_col="Date",
                            parse_dates=True,
                            usecols=['Date', 'Adj Close'],
                            na_values='nan'
                            )
        dftmp = dftmp.rename(columns={'Adj Close': symbol})
        df = df.join(dftmp, how='left')
        if 'SPY' == symbol:
            df = df.dropna(subset=['SPY'])

    # normalize
    # df = df/df[0]

    return df


def plot_selected(df, columns, start_date, end_date):
    df.ix[start_date : end_date, columns].plot()
    plt.show()


def plot_data(df, title='Stock Prices'):
    ax = df.plot(title=title, fontsize=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def normalized(df):
    df = df /df.ix[0,:]
    return df


def test_run():
    start_date = '2000-01-01'
    end_date = '2018-03-01'
    dates = pd.date_range(start=start_date,end=end_date)

    symbols = ['GOOG', 'IBM', 'GLD']

    df = get_data(symbols, dates)
    df = normalized(df)
    plot_data(df)

    plot_selected(df, ['SPY', 'GOOG'], '2011-01-01', '2011-12-31')

    print(df)



    # # slice by row
    # print("slice by row")
    # print(df['2010-01-01': '2010-12-31'])
    # print(df.ix['2010-01-01': '2010-12-31'])
    #
    # # slice by column
    # print("slice by column")
    # print(df[['GOOG', 'GLD']])
    #
    # # slice by range
    # print("slice by range")
    # print(df.ix['2010-01-01': '2010-12-31', ['GOOG', 'GLD']])






if __name__ == '__main__':
    test_run()
