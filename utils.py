import datetime
import re
import pandas as pd
import numpy as np
from pandas import Series, Timestamp
from typing import Union, Literal,NamedTuple,List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyecharts.charts import Line
from pyecharts import options as opts
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
import imgkit
path_wk = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe"
config = imgkit.config(wkhtmltoimage=path_wk) 
options = {
    'javascript-delay': 1000,
        'crop-w': 1200,
        'crop-h': 560,
        'crop-x': 65,
        'crop-y': 5,

}
def keep_chinese_chars(text):
    # 使用正则表达式匹配汉字
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
    
    # 将匹配到的汉字连接成一个字符串
    result = ''.join(chinese_chars)
    
    return result

def getLocalFiles():
    root = tk.Tk()
    root.withdraw()
    filePaths = filedialog.askopenfilenames()
    for filePath in filePaths:
        print('文件路径：', filePath)
    return filePaths


def get_nav_file_paths(nav_data_path) -> List[Path]:
    if nav_data_path == None:
        print(
            "请选择净值数据文件，请确保列名为：日期or净值日期 / 累计净值or累计单位净值"
        )
        nav_file_paths = [Path(path_i) for path_i in getLocalFiles()]
    else:
        assert nav_data_path.exists(), input("未找到文件夹/文件")
        if nav_data_path.is_file():
            nav_file_paths = [nav_data_path]
        else:
            nav_file_paths = [
                Path(path_i)
                for path_i in nav_data_path.glob(
                    "*.xlsx" or "*.xls"
                )
            ]

    assert len(nav_file_paths) > 0, input("未选择文件")
    return nav_file_paths

def format_nav_data(path):
    nav_data = pd.read_excel(path)
    nav_data = nav_data.rename(
        columns={
            "净值日期": "日期",
            "时间": "日期",
            "累计单位净值": "累计净值",
            "实际累计净值": "累计净值",
            "复权净值": "累计净值",
        }
    )[["日期", "累计净值"]]
    assert (nav_data["累计净值"] <= 0.01).sum() == 0, input(
        "Error: 净值数据中存在净值为0的数据"
    )
    assert nav_data["累计净值"].isnull().sum() == 0, input(
        "Error: 净值数据中存在累计净值为空的数据"
    )
    assert nav_data["日期"].isnull().sum() == 0, input(
        "Error: 净值数据中存在日期为空的数据"
    )
    nav_data = nav_data.drop_duplicates(subset="日期")
    if nav_data["日期"].dtype == "int":
        nav_data["日期"] = pd.to_datetime(nav_data["日期"], format="%Y%m%d")
    else:
        nav_data["日期"] = pd.to_datetime(nav_data["日期"])
    nav_data = nav_data.sort_values(by="日期", ascending=True).reset_index(drop=True)
    return nav_data

def match_data(
    nav_data: pd.DataFrame,
    trade_date: np.ndarray[np.datetime64],
) -> pd.DataFrame:
    nav_data = nav_data.set_index("日期")
    nav_data = nav_data.reindex(trade_date, method="ffill")
    return nav_data.reset_index(drop=False)

def _backword_analysis(nav_data, backword_delta: pd.Timedelta):
    date_backword_delta = nav_data["日期"].values[-1] - backword_delta
    assert (
        date_backword_delta >= nav_data["日期"].values[0]
    ), f"数据不足{str(backword_delta)},begin_date: {np.datetime_as_string(nav_data['日期'].values[0], unit='D')} ~ end_date: {np.datetime_as_string(nav_data['日期'].values[-1],unit='D')}"
    backword_period_nav_data = nav_data[nav_data["日期"] >= date_backword_delta]
    nav = backword_period_nav_data["累计净值"].values
    rtn = np.log(nav[1:]) - np.log(nav[:-1])
    res = curve_analysis(rtn, nav)
    res["begin_date"] = np.datetime_as_string(
        backword_period_nav_data["日期"].values[0], unit="D"
    )
    res["end_date"] = np.datetime_as_string(
        backword_period_nav_data["日期"].values[-1], unit="D"
    )
    return res

def backword_analysis(nav_data):
    max_backword_motnhs = (
        (nav_data["日期"].values[-1] - nav_data["日期"].values[0])
        .astype("timedelta64[M]")
        .astype(int)
    )
    res_dict = {}
    for i in [1, 3, 6, 12, 24, 36]:
        if i > max_backword_motnhs:
            break
        res = _backword_analysis(nav_data, pd.DateOffset(months=i))
        res_dict[i] = res
    out_df = pd.DataFrame(res_dict).T
    out_df.index.name = "backword months"
    out_df.index = out_df.index.astype(str) + "M"
    return out_df

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

def max_drawdown_stats(nav: np.ndarray, date: np.ndarray):
    assert len(nav) == len(date)
    # 动态回撤
    drawdown = nav - np.maximum.accumulate(nav)
    drawdown_infos = []
    idx = 0
    while idx < len(drawdown):
        if drawdown[idx] < 0:
            drawdown_info = {}
            drawdown_info["drawdown_start_date"] = date[idx - 1]
            drawdown_info["max_drawdown"] = drawdown[idx]
            drawdown_info["max_drawdown_date"] = date[idx]
            while drawdown[idx] < 0:
                if drawdown[idx] < drawdown_info["max_drawdown"]:
                    drawdown_info["max_drawdown"] = drawdown[idx]
                    drawdown_info["max_drawdown_date"] = date[idx]
                idx += 1
                if idx + 1 >= len(drawdown):
                    break
            drawdown_info["drawdown_end_date"] = date[idx]
            drawdown_infos.append(drawdown_info)
        idx += 1
    if drawdown[-1] < 0:
        drawdown_infos[-1]["drawdown_end_date"] = None
    return drawdown, pd.DataFrame(drawdown_infos)

def max_drawdown_period(nav: np.ndarray, date: np.ndarray):
    """
    计算最大回撤的开始时间和结束时间
    """
    out_put = {}
    assert len(nav) == len(date)
    # 动态回撤
    drawdown = nav - np.maximum.accumulate(nav)
    idx_maxDrawDown = np.argmin(drawdown)
    if idx_maxDrawDown == 0:
        out_put["最大回撤开始时间"] = "尚未回撤"
        out_put["最大回撤结束时间"] = "尚未回撤"
        out_put["最大回撤持续天数"] = "0 天"
        out_put["最大回撤修复时间"] = "尚未回撤"
        out_put["最大回撤修复天数"] = "尚未回撤"
        return drawdown, out_put
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

def curve_analysis(rtn: np.ndarray, nav: np.ndarray):
    assert rtn.ndim == 1, "rtn维度不为1"
    assert nav.ndim == 1, "nav维度不为1"
    assert len(rtn) == len(nav) - 1, "rtn的长度应并nav长度少1"
    rtn = clean(rtn)
    result = {"区间收益率": nav[-1] / nav[0] - 1}
    number_of_years = len(rtn) / 250
    result["年化收益"] = result["区间收益率"] / number_of_years
    result["年化波动率"] = np.nanstd(rtn) * np.sqrt(250)
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
    bench_mark_name: str = None,
    html_file_name: Path = None,
    origin_date: np.ndarray[np.datetime64] = None,
    image_save_path: Path = None
):
    drawdown_dict = {}
    if bench_mark_nav is not None:
        assert bench_mark_name is not None, "bench_mark_name 不能为空"
        bench_mark_nav = bench_mark_nav / bench_mark_nav[0]
        nav_data_dict.update({bench_mark_name: bench_mark_nav})
        excess_nav_dict = {}
        
    for key, nav_data in nav_data_dict.items():
        if key == bench_mark_name:
            continue
        assert len(trade_date) == len(nav_data), f"{key} date和nav长度不一致"
        assert bench_mark_nav is None or len(bench_mark_nav) == len(
            nav_data
        ), f"{key} bench_mark_nav和nav长度不一致"
        nav = nav_data / nav_data[0]

        if bench_mark_nav is not None:
            excess_nav = nav / bench_mark_nav
            nav_data_dict[key] = nav
            excess_nav_dict.update({f"超额_{key}": excess_nav})
            drawdown_dict.update({f"超额_{key}": excess_nav - np.maximum.accumulate(excess_nav)})

        else:
            nav_data_dict[key] = nav
        print(f"净值区间: {np.datetime_as_string(trade_date[0],unit="D")} - {np.datetime_as_string(trade_date[-1], unit='D')}")
            
    if bench_mark_nav is not None:
        nav_data_dict.update(excess_nav_dict)
    if html_file_name:
        html = nav_compare_analysis_echarts_plot(
            date=trade_date,
            nav=nav_data_dict,
            drawdown=drawdown_dict,
            origin_date=origin_date,
        )
        if image_save_path is not None:
            if not image_save_path.exists():
                image_save_path.mkdir(parents=True, exist_ok=True)
            img_path =image_save_path.joinpath(f"{html_file_name.stem}.jpg")
            print(f"正在保存图片至{img_path}, 请稍后...")
            imgkit.from_string(
                html,
                config=config,
                output_path= img_path,
                options = options,
                )
        with open(html_file_name, "w", encoding='utf-8') as f:
            f.write(html)



def nav_compare_analysis_echarts_plot(
    date: np.ndarray[np.datetime64],
    nav: dict[str:np.ndarray],
    drawdown: dict[str:np.ndarray],
    origin_date: np.ndarray[np.datetime64] = None,
):
    if origin_date is not None:
        select_date_idx = np.isin(date, origin_date)
    else:
        select_date_idx = np.ones(len(date), dtype=bool)
    date_str_list = date.astype("datetime64[D]").astype(str)[select_date_idx].tolist()
    max_nav = np.array([max(nav_i) for nav_i in nav.values()])
    min_nav = np.array([min(nav_i) for nav_i in nav.values()])
    up_bound, lower_bound = up_lower_bound(max(max_nav), min(min_nav))
    
    nav_line = Line(
        init_opts={
            "width":"1320px",
            "height":"600px",
            "is_horizontal_center":True,
            }
        ).add_xaxis(date_str_list)
    for key, value in nav.items():
        nav_line.add_yaxis(key, value[select_date_idx].tolist(), is_symbol_show=False)
    nav_line.set_global_opts(
        legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(font_weight="bold", font_size=20)),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
        title_opts=opts.TitleOpts("Value over Time"),
        yaxis_opts=opts.AxisOpts(min_=lower_bound, max_=up_bound),
    ).set_series_opts(
        linestyle_opts=opts.LineStyleOpts(width = 4),
        )
    # 绘制最大回撤区间
    drawdown_line = Line(
        init_opts={
            "width":"1320px",
            "height":"600px",
            "is_horizontal_center":True,
            }
        ).add_xaxis(date_str_list)
    for key, value in drawdown.items():
        drawdown_line.add_yaxis(key, value[select_date_idx].tolist(), is_symbol_show=False)
    drawdown_line.set_global_opts(
        legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(font_weight="bold", font_size=20)),
        datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")],
        title_opts=opts.TitleOpts("max drawdown")
    ).set_series_opts(linestyle_opts=opts.LineStyleOpts(width = 4))
    html = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <title>Value over Time</title>
            </head> 
            <body>
                {nav_line.render_embed()}
                {drawdown_line.render_embed()}
            </body>
        </html>
    """
    return html

class NavAnalysisConfig(NamedTuple):
    begin_date: pd.Timestamp = pd.to_datetime("2000-06-06")
    end_date: pd.Timestamp = pd.to_datetime("2099-06-06")
    special_html_name:bool=False
    open_html:bool=True
    benchmark: Literal[
        "SHSE.000300",
        "SHSE.000905",
        "SHSE.000852",
        "SZSE.399303",
        "SHSE.000985",
    ] = None
    image_save_path: Path = None
    overwrite:bool=False
    nav_data_path: Path=None
    index_data_path:Path=None
