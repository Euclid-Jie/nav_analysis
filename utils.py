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
def getLocalFiles():
    root = tk.Tk()
    root.withdraw()
    filePaths = filedialog.askopenfilenames()
    for filePath in filePaths:
        print('文件路径：', filePath)
    return filePaths

def match_data(
    nav_data: pd.DataFrame,
    trade_date: np.ndarray[np.datetime64],
) -> pd.DataFrame:
    nav_data = nav_data.set_index("日期")
    nav_data = nav_data.reindex(trade_date, method="ffill")
    return nav_data.reset_index(drop=False)

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

def nav_compare_analysis(
    trade_date: np.ndarray[np.datetime64],
    nav_data_dict: dict[str:np.ndarray[float]],
    bench_mark_nav : np.ndarray[float] = None,
    html_file_name: str = None,
    additional_table: list[pd.DataFrame] = None,
):
    metrics_dict = {}
    drawdown_dict = {}
    max_drawdown_info_dict = {}
    if bench_mark_nav is not None:
        bench_mark_nav = bench_mark_nav / bench_mark_nav[0]
        bench_mark_rtn = np.log(bench_mark_nav[1:]) - np.log(bench_mark_nav[:-1])
        nav_data_dict.update({"bench_mark": bench_mark_nav})
        excess_nav_dict = {}
        
    for key, nav_data in nav_data_dict.items():
        if key == "bench_mark":
            continue
        assert len(trade_date) == len(nav_data), f"{key} date和nav长度不一致"
        assert bench_mark_nav is None or len(bench_mark_nav) == len(
            nav_data
        ), f"{key} bench_mark_nav和nav长度不一致"
        nav = nav_data / nav_data[0]
        rtn = np.log(nav[1:]) - np.log(nav[:-1])

        if bench_mark_nav is not None:
            excess_rtn = rtn - bench_mark_rtn
            excess_nav = nav / bench_mark_nav
            
            nav_data_dict[key] = nav
            excess_nav_dict.update({f"超额_{key}": excess_nav})
            metrics = curve_analysis(excess_rtn)
            drawdown, max_drawdown_info = max_drawdown_period(excess_nav, trade_date)
            metrics_dict[f"超额_{key}"] = metrics
            drawdown_dict[f"超额_{key}"] = drawdown
            max_drawdown_info_dict[f"超额_{key}"] = max_drawdown_info
        else:
            nav_data_dict[key] = nav
            metrics = curve_analysis(rtn)
            drawdown,max_drawdown_info = max_drawdown_period(nav, trade_date)
            metrics_dict[key] = metrics
            drawdown_dict[key] = drawdown
            max_drawdown_info_dict[key] = max_drawdown_info
        print("-*-"*24)
        print(f"{key}净值区间: {np.datetime_as_string(trade_date[0],unit="D")} - {np.datetime_as_string(trade_date[-1], unit='D')}")
        for i, v in metrics.items():
            if i == "夏普比率":
                print(f"{i}：{v:.4f}")
            else:
                print(f"{i}：{v*100:.4f}%")
        for i, v in max_drawdown_info.items():
            print(f"{i}：{v}")
            
            
    if bench_mark_nav is not None:
        nav_data_dict.update(excess_nav_dict)
        bench_mark_drawdown,_ = max_drawdown_period(bench_mark_nav, trade_date)
        drawdown_dict.update({"bench_mark": bench_mark_drawdown})
    if html_file_name:
        html = nav_compare_analysis_echarts_plot(
            date=trade_date,
            nav=nav_data_dict,
            drawdown=drawdown_dict,
            table=pd.concat([pd.DataFrame(metrics_dict).T[["年化收益", "年化波动率", "夏普比率", "最大回撤"]], pd.DataFrame(max_drawdown_info_dict).T], axis=1),
            additional_table=additional_table,
        )
        with open(html_file_name, "w", encoding='utf-8') as f:
            f.write(html)



def nav_compare_analysis_echarts_plot(
    date: np.ndarray[np.datetime64],
    nav: dict[str:np.ndarray],
    drawdown: dict[str:np.ndarray],
    table: pd.DataFrame = None,
    additional_table: list[pd.DataFrame] = None,
):
    date_str_list = date.astype("datetime64[D]").astype(str).tolist()
    max_nav = np.array([max(nav_i) for nav_i in nav.values()])
    min_nav = np.array([min(nav_i) for nav_i in nav.values()])
    up_bound, lower_bound = up_lower_bound(max(max_nav), min(min_nav))
    
    nav_line = Line().add_xaxis(date_str_list)
    for key, value in nav.items():
        nav_line.add_yaxis(key, value.tolist(), is_symbol_show=False)
    nav_line.set_global_opts(
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
        title_opts=opts.TitleOpts("Value over Time"),
        yaxis_opts=opts.AxisOpts(min_=lower_bound, max_=up_bound),
    )
    # 绘制最大回撤区间
    drawdown_line = Line().add_xaxis(date_str_list)
    for key, value in drawdown.items():
        drawdown_line.add_yaxis(key, value.tolist(), is_symbol_show=False)
    drawdown_line.set_global_opts(
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
        title_opts=opts.TitleOpts("max drawdown")
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
