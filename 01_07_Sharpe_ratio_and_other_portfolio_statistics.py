import pandas as pd
import numpy as np
import time
import os
import utils
import math


def p02_daily_portfolio_values():
    """
    模拟 buy and hold 策略，计算投资组合日价值
    :return:
    """

    # 初始资金
    start_val = 1000000
    # 回归开始日期
    start_date = '2010-01-01'
    # 回归结束日期
    end_date = '2018-03-01'
    # 投资组合标的
    symbols = ['SPY', 'GOOG', 'XOM', 'GLD']
    # 配置策略
    allocs = [0.4, 0.4, 0.1, 0.1]

    # 获取数据
    df = utils.get_data_universal(symbols=symbols, start_date=start_date, end_date=end_date)
    # 归一化
    normed_df = utils.normed(df)
    # 乘以配置比例
    allocs_df = normed_df * allocs
    # 每一个position的价值
    pos_vals = allocs_df * start_val
    # 按x轴求和得到组合价值
    port_vals = pos_vals.sum(axis=1)
    port_vals_df = port_vals.to_frame(name='port_vals_df')
    print(port_vals_df)
    # 画图
    utils.plot_data(port_vals_df, title='Daily Portfolio Values', xlabel='Date', ylabel='Values')

    # daily returns
    daily_rets = utils.compute_daily_returns(port_vals_df)
    print(daily_rets)

    # 累计收益
    cum_ret = compute_cumulative_return(port_vals)
    print("cum_ret: {}".format(cum_ret))
    # avg daily returns
    avg_daily_rets = daily_rets.mean()
    print("avg_daily_rets: {}".format(avg_daily_rets))
    # std daily returns
    std_daily_ret = daily_rets.std()
    print("std_daily_ret: {}".format(std_daily_ret))
    # sharpe ratio
    sr = compute_sharpe_ratio(daily_rets)
    print("sr: {}".format(sr))


def test_run():
    p02_daily_portfolio_values()


if __name__ == '__main__':
    test_run()