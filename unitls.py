import datetime
import pandas as pd
import numpy as np
from pandas import Series, Timestamp
from typing import Union, Literal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyecharts.charts import Line
from pyecharts import options as opts
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
def getLocalFile():
    root=tk.Tk()
    root.withdraw()
    filePath=filedialog.askopenfilename()
    print('文件路径：',filePath)
    return filePath

def format_date(
    date: Union[datetime.datetime, datetime.date, np.datetime64, int, str]
) -> Series | Timestamp:
    if isinstance(date, datetime.datetime):
        return pd.to_datetime(date.date())
    elif isinstance(date, datetime.date):
        return pd.to_datetime(date)
    elif isinstance(date, np.datetime64):
        return pd.to_datetime(date)
    elif isinstance(date, int):
        date = pd.to_datetime(date, format="%Y%m%d")
        return date
    elif isinstance(date, str):
        date = pd.to_datetime(date)
        return pd.to_datetime(date.date())
    else:
        raise TypeError("date should be str, int or timestamp!")

def add_bench_data(
    nav_data: pd.DataFrame,
    bench_symbol: Literal[
        "SHSE.000300", "SHSE.000905", "SHSE.000852", "SZSE.399303", "SHSE.000985"
    ],
    index_data_path: Path = Path("index_data.csv"),
):
    """
    "SHSE.000300",  # 沪深300
    "SHSE.000905",  # 中证500
    "SHSE.000852",  # 中证1000
    "SZSE.399303",  # 国证2000
    "SHSE.000985",  # 中证全指
    """

    index_data = pd.read_csv(index_data_path)
    index_data = index_data[index_data["symbol"] == bench_symbol].reset_index(drop=True)
    index_data["bob"] = pd.to_datetime(index_data["bob"]).dt.strftime("%Y-%m-%d")
    index_data.set_index("bob", inplace=True)
    index_data.reindex(nav_data["日期"].dt.strftime("%Y-%m-%d"))
    index_data = index_data["close"].reset_index(drop=False)
    index_data["bob"] = pd.to_datetime(index_data["bob"])
    if index_data["bob"].max() < nav_data["日期"].max():
        print(f"benchmark数据最新为：{index_data["bob"].max()}，请更新")
    return pd.merge(nav_data, index_data, left_on="日期", right_on="bob", how="left")[
        :-1
    ]
def clean(arr: np.ndarray, inplace=False, fill_value=0.0) -> np.ndarray:
    """
    将array中的Inf, NaN转为 fill_value
    - fill_value默认值为0
    - inplace会同时改变传入的arr
    """
    assert arr.dtype == int or arr.dtype == float
    if inplace:
        res = arr
    else:
        res = arr.copy()
    res[~np.isfinite(res)] = fill_value
    return res


def maximum_draw_down(rtn: np.ndarray):
    assert rtn.ndim == 1
    min_all = 0
    sum_here = 0
    for x in rtn:
        sum_here += x
        if sum_here < min_all:
            min_all = sum_here
        elif sum_here >= 0:
            sum_here = 0
    return -min_all


def max_drawdown_period(nav: np.ndarray, date: np.ndarray):
    """
    计算最大回撤的开始时间和结束时间
    """
    out_put = {}
    assert len(nav) == len(date)
    # 动态回撤
    drawdown = nav - np.maximum.accumulate(nav)
    idx_maxDrawDown = np.argmin(drawdown)
    # 往前找到最大值
    idx_maxDrawDown_begin = int(np.where(drawdown[0:idx_maxDrawDown] == 0)[0][-1])
    out_put["最大回撤开始时间"] = date[idx_maxDrawDown_begin].astype("datetime64[D]")
    out_put["最大回撤结束时间"] = date[idx_maxDrawDown].astype("datetime64[D]")
    out_put["最大回撤持续天数"] = f"{(date[idx_maxDrawDown] - date[idx_maxDrawDown_begin]).astype("timedelta64[D]").astype(int)} 天"

    # 往后找到修复时间
    weather_recover = np.where(drawdown[idx_maxDrawDown:] == 0)[0]
    if len(weather_recover) > 0:
        idx_maxDrawDown_fix = (
            int(np.where(drawdown[idx_maxDrawDown:] == 0)[0][0]) + idx_maxDrawDown
        )
        out_put["最大回撤修复时间"] = date[idx_maxDrawDown_fix].astype("datetime64[D]")
        out_put["最大回撤修复天数"] = f"{(date[idx_maxDrawDown_fix] - date[idx_maxDrawDown]).astype("timedelta64[D]").astype(int)} 天"
    else:
        out_put["最大回撤修复时间"] = "尚未修复"
        out_put["最大回撤修复天数"] = "尚未修复"
    return drawdown, out_put


def curve_analysis(rtn: np.ndarray):
    assert rtn.ndim == 1
    rtn = clean(rtn)
    nav = np.cumsum(rtn) + 1
    result = {"total_rtn": nav[-1] / nav[0] - 1}
    number_of_years = len(rtn) / 250
    result["年化收益"] = result["total_rtn"] / number_of_years
    result["total_std"] = np.nanstd(rtn)
    result["年化波动率"] = result["total_std"] * np.sqrt(250)
    result["夏普比率"] = result["年化收益"] / result["年化波动率"]
    result["最大回撤"] = maximum_draw_down(rtn)
    return result


def nav_analysis_plot(
    date: np.ndarray[np.datetime64],
    nav: np.ndarray,
    drawdown: np.ndarray,
    bench_mark_nav: np.ndarray = None,
):
    # 绘图, 2*1
    _, ax = plt.subplots(2, 1, figsize=(20, 15))

    # Rotate x-axis labels by 45 degrees
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].tick_params(axis="x", rotation=30)

    # Rest of the code
    ax[0].xaxis.set_major_locator(mdates.MonthLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax[0].plot(date, nav, label="nav")
    if bench_mark_nav is not None:
        ax[0].plot(date, bench_mark_nav, label="benchmark")
        # 超额收益
        ax[0].plot(date, nav / bench_mark_nav, label="excess nav")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].xaxis.set_major_locator(mdates.MonthLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax[1].plot(date, drawdown)
    ax[1].grid(True)

    plt.show()
    
    
def win_ratio_stastics(nav_data: np.ndarray,start_date: np.datetime64 = None):
    """
    目前只支持月度胜率统计
    """
    if start_date is not None:
        nav_data = nav_data[nav_data["日期"]>= start_date]
    nav_data["rtn"] = nav_data["累计净值"].pct_change()
    # nav_data["rtn"] = np.log(nav_data["累计净值"]) - np.log(nav_data["累计净值"].shift(1))
    monthly_rtn = (
        nav_data.groupby(pd.Grouper(key="日期", freq="ME"))
        .apply(
            lambda x: (1 + x["rtn"]).prod() - 1,
            # lambda x: x["rtn"].sum(),
            include_groups=False,
        )
        .to_frame("月度收益")
        .reset_index()
    )
    monthly_rtn["year"] = monthly_rtn["日期"].dt.year
    monthly_rtn["month"] = monthly_rtn["日期"].dt.month
    monthly_rtn = monthly_rtn.pivot_table(index="year", columns="month", values="月度收益", aggfunc="sum")
    monthly_rtn.columns = [f"{x}月" for x in monthly_rtn.columns]
    monthly_rtn.index.name = None
    monthly_rtn["月度胜率"] = monthly_rtn.apply(lambda x: (x >= 0).sum() / (~np.isnan(x)).sum(), axis=1)
    # 保留四位小数
    monthly_rtn = monthly_rtn.map(lambda x: round(x, 4))
    return monthly_rtn
    
def up_lower_bound(max_value,min_value, precision = 0.1, decimal = 2):
    assert max_value > min_value
    up_bound = max_value + precision * (max_value - min_value)
    up_bound = np.ceil(up_bound * 10 ** decimal) / 10 ** decimal
    
    lower_bound = min_value - precision * (max_value - min_value)
    lower_bound = np.floor(lower_bound * 10 ** decimal) / 10 ** decimal
    return up_bound, lower_bound

def nav_analysis_echarts_plot(
    date: np.ndarray[np.datetime64],
    nav: np.ndarray,
    drawdown: np.ndarray,
    bench_mark_nav: np.ndarray = None,
    table: pd.DataFrame = None,
    additional_table: list[pd.DataFrame] = None,
):
    date_str_list = date.astype("datetime64[D]").astype(str).tolist()
    if bench_mark_nav is not None:
        excess_nav = nav / bench_mark_nav
        y_max = max(max(nav),max(bench_mark_nav),max(excess_nav))
        y_min = min(min(nav),min(bench_mark_nav),min(excess_nav))
        up_bound, lower_bound = up_lower_bound(y_max, y_min)
        nav_line = (
        Line()
        .add_xaxis(date_str_list)
        .add_yaxis("nav", nav.tolist(), is_symbol_show=False)
        .add_yaxis("bench_mark_nav", bench_mark_nav.tolist(), is_symbol_show=False)
        .add_yaxis("excess nav", (nav / bench_mark_nav).tolist(), is_symbol_show=False)
        .set_global_opts(
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
            title_opts=opts.TitleOpts("Value over Time"),
            yaxis_opts=opts.AxisOpts(min_= lower_bound, max_=up_bound)
            ))
    else:
        up_bound, lower_bound = up_lower_bound(max(nav), min(nav))
        nav_line = (
            Line()
            .add_xaxis(date_str_list)
            .add_yaxis("nav", nav.tolist(), is_symbol_show=False)
            .set_global_opts(
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
                title_opts=opts.TitleOpts("Value over Time"),
                yaxis_opts=opts.AxisOpts(min_= lower_bound, max_=up_bound),
            ))        
    # 绘制最大回撤区间
    drawdown_line = (
            Line()
            .add_xaxis(date_str_list)
            .add_yaxis("drawdown", drawdown.tolist(), is_symbol_show=False)
            .set_global_opts(
                datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
                title_opts=opts.TitleOpts("Value over Time")
            )
        )  
    table = table.to_html(render_links=True)
    if additional_table is not None:
        additional_table = [table_i.to_html(render_links=True) for table_i in additional_table]
        html = f"""
            <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Value over Time</title>
                </head> 
                <body>
                    {table}
                    {nav_line.render_embed()}
                    {"".join(additional_table)}
                    {drawdown_line.render_embed()}
                </body>
            </html>
        """
    else:
        html = f"""
            <html>
                <head>
                    <title>Value over Time</title>
                </head> 
                <body>
                    {table}
                    {nav_line.render_embed()}
                    {drawdown_line.render_embed()}
                </body>
            </html>
        """
    return html
    

def nav_analysis(
    date: np.ndarray[np.datetime64],
    nav: np.ndarray[np.datetime64],
    bench_mark_nav: np.ndarray = None,
    begin_date: np.datetime64 = None,
    end_date: np.datetime64 = None,
    html_file_name:str = None,
    additional_table: list[pd.DataFrame] = None,
    plot = True
):
    assert len(date) == len(nav), "date和nav长度不一致"
    assert bench_mark_nav is None or len(bench_mark_nav) == len(
        nav
    ), "bench_mark_nav和nav长度不一致"

    if begin_date is not None:
        nav = nav[date >= begin_date]
        if bench_mark_nav is not None:
            bench_mark_nav = bench_mark_nav[date >= begin_date]
        date = date[date >= begin_date]
    if end_date is not None:
        nav = nav[date <= end_date]
        if bench_mark_nav is not None:
            bench_mark_nav = bench_mark_nav[date <= end_date]
        date = date[date <= end_date]
    print(f"净值区间: {np.datetime_as_string(date[0],unit="D")} - {np.datetime_as_string(date[-1],unit="D")}")
    
    
    nav = nav / nav[0]
    rtn = np.log(nav[1:]) - np.log(nav[:-1])

    if bench_mark_nav is not None:
        bench_mark_nav = bench_mark_nav / bench_mark_nav[0]
        bench_mark_rtn = np.log(bench_mark_nav[1:]) - np.log(bench_mark_nav[:-1])
        excess_rtn = rtn - bench_mark_rtn
        excess_nav = nav / bench_mark_nav
        metrics = curve_analysis(excess_rtn)
        drawdown, max_drawdown_info = max_drawdown_period(excess_nav, date)
    else:
        metrics = curve_analysis(rtn)
        drawdown,max_drawdown_info = max_drawdown_period(nav, date)
    for i, v in metrics.items():
        if i == "夏普比率":
            print(f"{i}：{v:.4f}")
        else:
            print(f"{i}：{v*100:.4f}%")
    for i, v in max_drawdown_info.items():
        print(f"{i}：{v}")
    if plot:
        nav_analysis_plot(
            date=date,
            nav=nav,
            drawdown=drawdown,
            bench_mark_nav=bench_mark_nav if bench_mark_nav is not None else None,
        )
    if html_file_name:
        html = nav_analysis_echarts_plot(
            date=date,
            nav=nav,
            drawdown=drawdown,
            bench_mark_nav=bench_mark_nav if bench_mark_nav is not None else None,
            table=pd.concat([pd.DataFrame(metrics, index=[0])[["年化收益", "年化波动率", "夏普比率", "最大回撤"]], pd.DataFrame(max_drawdown_info, index=[0])], axis=1),
            additional_table=additional_table,
        )
        with open(html_file_name, "w", encoding='utf-8') as f:
            f.write(html)
