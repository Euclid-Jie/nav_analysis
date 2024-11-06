import datetime
import pandas as pd
import numpy as np
from pandas import Series, Timestamp
from typing import Union, Literal, NamedTuple, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyecharts.charts import Line
from pyecharts import options as opts
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import os
import re
import imgkit


def keep_chinese_chars(text):
    # 使用正则表达式匹配汉字
    chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)

    # 将匹配到的汉字连接成一个字符串
    result = "".join(chinese_chars)

    return result


def getLocalFile(log=True, suffix: List[str] = None) -> List[Path]:
    """
    选中本地文件, 返回单个
    """
    root = tk.Tk()
    root.withdraw()
    filePath = filedialog.askopenfilename()
    filePath = Path(filePath)

    if suffix is not None:
        assert filePath.suffix in suffix, input(f"文件{filePath.name}后缀不符合要求")
    if log:
        print("文件路径：", filePath)
    return filePath


def getLocalFiles(log=True, suffix: List[str] = None) -> List[Path]:
    """
    选中本地文件, 返回列表
    """
    root = tk.Tk()
    root.withdraw()
    filePaths = filedialog.askopenfilenames()
    assert len(filePaths) > 0, input("未选择文件")
    filePaths = [Path(path_i) for path_i in filePaths]

    if suffix is not None:
        for filePath in filePaths:
            assert filePath.suffix in suffix, input(
                f"文件{filePath.name}后缀不符合要求"
            )
    if log:
        for filePath in filePaths:
            print("文件路径：", filePath)
    return filePaths


def load_bench_data(index_data_path: Path = None):
    if index_data_path is not None:
        index_data = pd.read_csv(index_data_path)
    else:
        print("请手动选择指数数据文件")
        index_data = pd.read_csv(getLocalFiles()[0])
    index_data["bob"] = pd.to_datetime(index_data["bob"]).dt.tz_localize(None)
    return index_data


def generate_trading_date(
    begin_date: np.datetime64 = np.datetime64("2015-01-01"),
    end_date: np.datetime64 = np.datetime64("today"),
) -> Tuple[np.ndarray[np.datetime64]]:
    assert begin_date >= np.datetime64(
        "2015-01-04"
    ), "系统预设起始日期仅支持2015年1月4日以后"
    with open(
        Path(__file__).resolve().parent.joinpath("Chinese_special_holiday.txt"), "r"
    ) as f:
        chinese_special_holiday = pd.Series(
            [date.strip() for date in f.readlines()]
        ).values.astype("datetime64[D]")
    working_date = pd.date_range(begin_date, end_date, freq="B").values.astype(
        "datetime64[D]"
    )
    trading_date = np.setdiff1d(working_date, chinese_special_holiday)
    trading_date_df = pd.DataFrame(working_date, columns=["working_date"])
    trading_date_df["is_friday"] = trading_date_df["working_date"].apply(
        lambda x: x.weekday() == 4
    )
    trading_date_df["trading_date"] = (
        trading_date_df["working_date"]
        .apply(lambda x: x if x in trading_date else np.nan)
        .ffill()
    )
    return (
        trading_date,
        np.unique(
            trading_date_df[trading_date_df["is_friday"]]["trading_date"].values[1:]
        ),
    )


def infer_frequency(
    date: np.ndarray[np.datetime64], threshold=0.75
) -> Literal["W", "D"]:
    # 如果大部分日期间隔为 1 天，那么数据可能是日度的
    if (np.diff(date) == np.timedelta64(1, "D")).mean() > threshold:
        return "D"
    elif (np.diff(date) >= np.timedelta64(5, "D")).mean() > threshold:
        return "W"
    else:
        print("无法推断频率, 将自动转为周度")
        return "W"


def format_nav_data(path, ingnore_null=True):
    if path.suffix == ".csv":
        nav_data = pd.read_csv(path)
    else:
        nav_data = pd.read_excel(path)
    nav_data = nav_data.rename(columns={"净值日期": "日期", "时间": "日期"})
    assert "日期" in nav_data.columns, input("Error: 未找到日期列")
    for fake_name in [
        "复权净值",
        "累计净值",
        "累计单位净值",
        "实际累计净值",
        "单位净值",
    ]:
        if fake_name in nav_data.columns:
            nav_data = nav_data[["日期", fake_name]].rename(
                columns={fake_name: "累计净值"}
            )
            break
    assert "累计净值" in nav_data.columns, input("Error: 未找到累计净值列")
    assert (nav_data["累计净值"] <= 0.01).sum() == 0, input(
        "Error: 净值数据中存在净值为0的数据"
    )
    if ingnore_null == False:
        assert nav_data["累计净值"].isnull().sum() == 0, input(
            "Error: 净值数据中存在累计净值为空的数据"
        )
    else:
        if nav_data["累计净值"].isnull().sum() > 0:
            print("已剔除, 净值数据中存在累计净值为空的数据")
            nav_data = nav_data[~nav_data["累计净值"].isnull()]
    assert nav_data["日期"].isnull().sum() == 0, input(
        "Error: 净值数据中存在日期为空的数据"
    )
    if nav_data["日期"].duplicated(keep=False).sum() != 0:
        if (
            input(
                "Info: 净值数据中存在日期重复的数据\n{}\n 键入回车键自动剔除重复".format(
                    nav_data[nav_data["日期"].duplicated()]
                )
            )
            == ""
        ):
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


def _backword_analysis(
    nav_data, backword_delta: pd.Timedelta, freq: Literal["W", "D"] = "W"
):
    date_backword_delta = nav_data["日期"].values[-1] - backword_delta
    assert (
        date_backword_delta >= nav_data["日期"].values[0]
    ), f"数据不足{str(backword_delta)},begin_date: {np.datetime_as_string(nav_data['日期'].values[0], unit='D')} ~ end_date: {np.datetime_as_string(nav_data['日期'].values[-1],unit='D')}"
    backword_period_nav_data = nav_data[nav_data["日期"] >= date_backword_delta]

    nav = backword_period_nav_data["累计净值"].values
    res = curve_analysis(nav=nav, freq=freq)
    res["begin_date"] = np.datetime_as_string(
        backword_period_nav_data["日期"].values[0], unit="D"
    )
    res["end_date"] = np.datetime_as_string(
        backword_period_nav_data["日期"].values[-1], unit="D"
    )
    return res


def backword_analysis(
    nav: np.ndarray, date: np.ndarray[np.datetime64], freq: Literal["W", "D"] = "W"
):
    assert len(nav) == len(date), "nav和date长度不一致"
    nav_data = pd.DataFrame({"日期": date, "累计净值": nav})
    max_backword_motnhs = (
        (nav_data["日期"].values[-1] - nav_data["日期"].values[0])
        .astype("timedelta64[M]")
        .astype(int)
    )
    res_dict = {}
    for i in [1, 3, 6, 12, 24, 36]:
        if i > max_backword_motnhs:
            break
        res = _backword_analysis(nav_data, pd.DateOffset(months=i), freq=freq)
        res_dict[f"{i}M"] = res

    if nav_data["日期"].values.min() <= pd.to_datetime(
        f"{datetime.datetime.now().year}-01-01"
    ):
        YTD_idx = np.where(
            nav_data["日期"] >= pd.to_datetime(f"{datetime.datetime.now().year}-01-01")
        )[0][1]
        ytd_data = nav_data[YTD_idx:]
        ytd_metrics = curve_analysis(ytd_data["累计净值"].values, freq=freq)
        ytd_metrics["begin_date"] = np.datetime_as_string(
            ytd_data["日期"].values[0], unit="D"
        )
        ytd_metrics["end_date"] = np.datetime_as_string(
            ytd_data["日期"].values[-1], unit="D"
        )
        res_dict.update({"YTD": ytd_metrics})

    backword_analysis_df = pd.DataFrame(res_dict).T
    backword_analysis_df.index.name = "backword months"
    for col in ["区间收益率", "年化收益率", "区间波动率", "年化波动率", "最大回撤"]:
        backword_analysis_df[col] = backword_analysis_df[col].map(lambda x: f"{x:.3%}")

    backword_analysis_df["夏普比率"] = backword_analysis_df["夏普比率"].apply(
        lambda x: f"{x:.3f}"
    )
    return backword_analysis_df


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


def drawdown_stats(nav: np.ndarray, date: np.ndarray):
    assert len(nav) == len(date), "nav和date长度不一致, 请检查bench_data是否更新"
    # 动态回撤
    cummax = np.maximum.accumulate(nav)
    drawdown = (nav - cummax) / cummax
    drawdown_infos = []
    idx = 0
    while idx < len(drawdown) - 1:
        if drawdown[idx] < 0:
            drawdown_info = {}
            drawdown_info["drawdown_start_date"] = date[idx - 1]
            drawdown_info["max_drawdown"] = drawdown[idx]
            drawdown_info["max_drawdown_date"] = date[idx]
            while drawdown[idx] < 0:
                if drawdown[idx] < drawdown_info["max_drawdown"]:
                    drawdown_info["max_drawdown"] = drawdown[idx]
                    drawdown_info["max_drawdown_date"] = date[idx]
                if idx == len(drawdown) - 1:
                    break
                idx += 1
            drawdown_info["drawdown_end_date"] = date[idx]
            drawdown_infos.append(drawdown_info)
        else:
            idx += 1
    if drawdown[-1] < 0:
        drawdown_infos[-1]["drawdown_end_date"] = np.datetime64("NaT")

    drawdown_infos = pd.DataFrame(drawdown_infos)
    for col in ["drawdown_start_date", "max_drawdown_date", "drawdown_end_date"]:
        drawdown_infos[col] = drawdown_infos[col]
    drawdown_infos["max_drawdown_days"] = (
        drawdown_infos["max_drawdown_date"] - drawdown_infos["drawdown_start_date"]
    )
    drawdown_infos["drawdown_fix_days"] = (
        drawdown_infos["drawdown_end_date"] - drawdown_infos["max_drawdown_date"]
    )
    return (
        drawdown,
        drawdown_infos[
            [
                "max_drawdown",
                "drawdown_start_date",
                "max_drawdown_date",
                "max_drawdown_days",
                "drawdown_end_date",
                "drawdown_fix_days",
            ]
        ],
    )


def display_df(data: pd.DataFrame):
    df = data.copy()
    for col in df.columns:
        if "date" in col or "日期" in col:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        if "days" in col or "天数" in col:
            df[col] = df[col].dt.days
        if "收益率" in col:
            df[col] = df[col].map(lambda x: f"{x:.3%}")
        if "波动率" in col:
            df[col] = df[col].map(lambda x: f"{x:.3%}")
        if "夏普比率" in col:
            df[col] = df[col].map(lambda x: f"{x:.3f}")
        if "最大回撤" in col:
            df[col] = df[col].map(lambda x: f"{x:.3%}")
    return df


def curve_analysis(nav: np.ndarray, freq: Literal["W", "D"] = "W") -> dict:
    assert nav.ndim == 1, "nav维度不为1"
    assert np.isnan(nav).sum() == 0, "nav中有nan"
    result = {"区间收益率": nav[-1] / nav[0] - 1}
    result["年化收益率"] = (
        result["区间收益率"] / len(nav) * (250 if freq == "D" else 52)
    )

    rtn = np.log(nav[1:] / nav[:-1])
    result["区间波动率"] = np.std(rtn)
    result["年化波动率"] = result["区间波动率"] * np.sqrt(250 if freq == "D" else 52)
    result["夏普比率"] = result["年化收益率"] / result["年化波动率"]
    cummax = np.maximum.accumulate(nav)
    result["最大回撤"] = np.min((nav - cummax) / cummax)
    return result


def calc_nav_rtn(nav: np.ndarray, types: Literal["log", "simple"] = "log"):
    if types == "simple":
        rtn = nav[1:] / nav[:-1] - 1
    elif types == "log":
        rtn = np.log(nav[1:] / nav[:-1])
    else:
        raise ValueError("types参数错误")
    return np.insert(rtn, 0, np.nan)


def weekly_rtn_stats(nav: np.ndarray, date: np.ndarray[np.datetime64], tail=10):
    assert len(nav) == len(date), "nav和date长度不一致"
    nav_data = pd.DataFrame({"日期": date, "累计净值": nav})
    nav_data["rtn"] = np.log(nav_data["累计净值"]) - np.log(
        nav_data["累计净值"].shift(1)
    )
    weekly_rtn = (
        nav_data.groupby(pd.Grouper(key="日期", freq="W"))
        .apply(
            lambda x: x["rtn"].sum(skipna=True),
            include_groups=False,
        )
        .to_frame("周收益")
        .reset_index()
    )
    weekly_rtn["周收益"] = weekly_rtn["周收益"].apply(lambda x: round(x, 4))
    weekly_rtn["日期"] = weekly_rtn["日期"] - datetime.timedelta(days=2)
    weekly_rtn_table = weekly_rtn.copy()
    weekly_rtn_table = weekly_rtn_table[
        weekly_rtn_table["日期"] <= nav_data["日期"].max()
    ].tail(tail)
    weekly_rtn_table["日期"] = weekly_rtn_table["日期"].dt.strftime("%Y-%m-%d")
    weekly_rtn_table.set_index("日期", inplace=True)
    weekly_rtn_table = weekly_rtn_table.T
    return weekly_rtn_table


def win_ratio_stastics(nav: np.ndarray, date: np.ndarray[np.datetime64]):
    """
    目前只支持月度胜率统计
    """
    assert len(nav) == len(date), "nav和date长度不一致"
    nav_data = pd.DataFrame({"日期": date, "累计净值": nav})
    nav_data["rtn"] = np.log(nav_data["累计净值"]) - np.log(
        nav_data["累计净值"].shift(1)
    )
    monthly_rtn = (
        nav_data.groupby(pd.Grouper(key="日期", freq="ME"))
        .apply(
            lambda x: x["rtn"].sum(),
            include_groups=False,
        )
        .to_frame("月度收益")
        .reset_index()
    )
    monthly_rtn["year"] = monthly_rtn["日期"].dt.year
    monthly_rtn["month"] = monthly_rtn["日期"].dt.month
    monthly_rtn = monthly_rtn.pivot_table(
        index="year", columns="month", values="月度收益", aggfunc="sum"
    )
    monthly_rtn.columns = [f"{x}月" for x in monthly_rtn.columns]
    monthly_rtn.index.name = None
    monthly_rtn["年度总收益"] = monthly_rtn.apply(lambda x: np.nansum(x), axis=1)
    monthly_rtn["月度胜率"] = monthly_rtn.apply(
        lambda x: (x >= 0).sum() / (~np.isnan(x)).sum(), axis=1
    )
    # 保留四位小数
    monthly_rtn = monthly_rtn.map(lambda x: round(x, 4))
    for col in monthly_rtn.columns:
        monthly_rtn[col] = monthly_rtn[col].map(lambda x: f"{x:.3%}")

    # 将NAN变成NULL
    return monthly_rtn.replace("nan%", "")


def up_lower_bound(max_value, min_value, precision=0.1, decimal=2):
    assert (
        np.isnan(max_value) == False and np.isnan(min_value) == False
    ), "max_value or min_value is nan"
    assert (
        max_value > min_value
    ), "max_value <= min_value,but {max_value} > {min_value}"
    up_bound = max_value + precision * (max_value - min_value)
    up_bound = np.ceil(up_bound * 10**decimal) / 10**decimal

    lower_bound = min_value - precision * (max_value - min_value)
    lower_bound = np.floor(lower_bound * 10**decimal) / 10**decimal
    return up_bound, lower_bound


def nav_analysis_echarts_plot(
    date: np.ndarray[np.datetime64],
    nav: dict[str : np.ndarray],
    drawdown: dict[str : np.ndarray],
    table: pd.DataFrame = None,
    additional_table: list[pd.DataFrame] = None,
):
    date_str_list = date.astype("datetime64[D]").astype(str).tolist()
    max_nav = np.array([max(nav_i) for nav_i in nav.values()])
    min_nav = np.array([min(nav_i) for nav_i in nav.values()])
    up_bound, lower_bound = up_lower_bound(max(max_nav), min(min_nav))

    nav_line = Line(
        init_opts={
            "width": "1560px",
            "height": "600px",
            "is_horizontal_center": True,
        }
    ).add_xaxis(date_str_list)
    for key, value in nav.items():
        nav_line.add_yaxis(key, np.round(value, 4).tolist(), is_symbol_show=False)
    nav_line.set_global_opts(
        legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(font_weight="bold", font_size=20)
        ),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")
        ],
        title_opts=opts.TitleOpts("Value over Time"),
        yaxis_opts=opts.AxisOpts(min_=lower_bound, max_=up_bound),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    ).set_series_opts(
        linestyle_opts=opts.LineStyleOpts(width=3),
    )
    # 绘制最大回撤区间
    drawdown_line = Line(
        init_opts={
            "width": "1560px",
            "height": "600px",
            "is_horizontal_center": True,
        }
    ).add_xaxis(date_str_list)
    for key, value in drawdown.items():
        drawdown_line.add_yaxis(key, np.round(value, 4).tolist(), is_symbol_show=False)
    drawdown_line.set_global_opts(
        legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(font_weight="bold", font_size=20)
        ),
        datazoom_opts=[
            opts.DataZoomOpts(range_start=0, range_end=100, orient="horizontal")
        ],
        title_opts=opts.TitleOpts("max drawdown"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    ).set_series_opts(linestyle_opts=opts.LineStyleOpts(width=3))
    table = table.to_html(render_links=True)
    html = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <title>Value over Time</title>
                <style>
                body {{
                    font-family: kaiti, Fira Code;
                }}
                h1 {{
                    text-align: center;
                    margin-top: 2px;
                    font-size: 24px;
                    color: #333;
                }}
                table {{
                    margin: auto;
                    margin-bottom: 20px;
                    border-collapse: collapse;
                    width: 1500px;
                    text-align: center;

                }}
                table,
                th,
                td {{
                    border: 1px solid #8d8b8b;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #f59e00;
                    color: white;
                }}
            </style>
            </head>
            <body>
                {table}
                {nav_line.render_embed()}
                {"".join([table_i.to_html(render_links=True) for table_i in additional_table]) if additional_table is not None else ""}
                {drawdown_line.render_embed()}
            </body>
        </html>
    """
    return html


class NavAnalysisConfig(NamedTuple):
    """
    param:
        nav_data_path: 支持多个净值数据文件路径
        image_save_parh: 图片保存路径, 为None则不保存图片
    """

    begin_date: np.datetime64 = np.datetime64("2000-06-06")
    end_date: np.datetime64 = np.datetime64("2099-06-06")
    special_html_name: bool = False
    open_html: bool = True
    benchmark: Literal[
        "SHSE.000300",
        "SHSE.000905",
        "SHSE.000852",
        "SZSE.399303",
        "SHSE.000985",
    ] = None
    nav_data_path: List[Path] | Path = None
    bench_data_path: Path = None

    def dict(self):
        return self._asdict()

    def copy(self, **kwargs):
        return self._replace(**kwargs)


def ffill(arr: np.ndarray):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    return out
