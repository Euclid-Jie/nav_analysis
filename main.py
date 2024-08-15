import datetime
from utils import *

print("请选择净值数据文件，请确保列名为：日期or净值日期 / 累计净值or累计单位净值")
nav_file_paths = [Path(path_i) for path_i in getLocalFiles()]
assert len(nav_file_paths) > 0, input("未选择文件")

# 读取指数数据
index_data = pd.read_csv(nav_file_paths[0].parent.joinpath("index_data.csv"))
index_data["bob"] = pd.to_datetime(index_data["bob"]).dt.tz_localize(None)
trade_date = np.unique(index_data["bob"].values).astype("datetime64[ns]")

begin_date = np.datetime64("2000-06-06")
end_date = np.datetime64("2099-06-06")

# 读取数据并确定时间区间
nav_data_dict = {}
for path in nav_file_paths:
    nav_data = pd.read_excel(path)
    nav_data = nav_data.rename(
        columns={
            "净值日期": "日期",
            "时间": "日期",
            "累计单位净值": "累计净值",
            "实际累计净值": "累计净值",
        }
    )[["日期", "累计净值"]]
    assert (nav_data["累计净值"] <= 0.01).sum() == 0, input(
        "Error: 净值数据中存在净值为0的数据"
    )
    if nav_data["日期"].dtype == "int":
        nav_data["日期"] = pd.to_datetime(nav_data["日期"], format="%Y%m%d")
    else:
        nav_data["日期"] = pd.to_datetime(nav_data["日期"])
    nav_data = nav_data.sort_values(by="日期", ascending=True).reset_index(drop=True)
    # 选取最大的开始时间作为开始时间
    if nav_data["日期"].min() >= begin_date:
        begin_date = nav_data["日期"].min()
    if nav_data["日期"].max() <= end_date:
        end_date = nav_data["日期"].max()
    print(
        f"{path.stem}净值数据中，时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
    )
    nav_data_dict[path.stem] = nav_data

# 对时间区间进行修改
begin_date_input = input(
    f"统计时间将开始于{begin_date.strftime('%Y-%m-%d')}，如需调整请输入开始统计的日期[YYYY-MM-DD]："
)
if begin_date_input != "":
    begin_date_input = np.datetime64(begin_date_input)
    assert begin_date_input >= begin_date, input("开始统计日期早于净值数据最早日期")
    begin_date = begin_date_input

end_date_input = input(
    f"统计时间将结束于{end_date.strftime('%Y-%m-%d')}，如需调整请输入结束统计的日期[YYYY-MM-DD]："
)
if end_date_input != "":
    end_date_input = np.datetime64(end_date_input)
    assert end_date_input <= end_date, input("结束统计日期晚于净值数据最晚日期")
    end_date = end_date_input

monthly_rtn_dict = {}
if len(nav_data_dict) == 1:
    origin_date = nav_data["日期"]
    origin_date = origin_date[origin_date >= begin_date]
    origin_date = origin_date[origin_date <= end_date]
else:
    origin_date = None
for key, nav_data in nav_data_dict.items():
    nav_data = match_data(nav_data=nav_data_dict[key], trade_date=trade_date)
    nav_data = nav_data[nav_data["日期"] >= begin_date]
    nav_data = nav_data[nav_data["日期"] <= end_date]
    nav_data["rtn"] = nav_data["累计净值"].pct_change()
    nav_data_dict[key] = nav_data
    monthly_rtn = win_ratio_stastics(
        nav_data[["日期", "累计净值"]], start_date=begin_date
    )
    monthly_rtn.index = [f"{key}_{i}" for i in monthly_rtn.index]
    monthly_rtn_dict[key] = monthly_rtn

trade_date = trade_date[trade_date >= begin_date]
trade_date = trade_date[trade_date <= end_date]

html_name = input("请输入导出的html文件名：")
if html_name == "" and len(nav_file_paths) > 1:
    html_name = (
        "compare_"
        + datetime.date.today().strftime("%Y%m%d")
        + "_"
        + nav_file_paths[0].stem
        + "_etc_nav_analysis"
    )
elif html_name == "":
    html_name = (
        datetime.date.today().strftime("%Y%m%d")
        + "_"
        + nav_file_paths[0].stem
        + "_nav_analysis"
    )
else:
    pass
html_file_path = nav_file_paths[0].parent.joinpath(f"{html_name}.html")
print(f"html路径为：{html_file_path}")


if len(nav_file_paths) == 1:
    weekly_rtn = (
        list(nav_data_dict.values())[0]
        .groupby(pd.Grouper(key="日期", freq="W"))
        .apply(
            lambda x: (1 + x["rtn"]).prod() - 1,
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
    ].tail(8)
    weekly_rtn_table["日期"] = weekly_rtn_table["日期"].dt.strftime("%Y-%m-%d")
    weekly_rtn_table.set_index("日期", inplace=True)
    weekly_rtn_table = weekly_rtn_table.T
    additional_table = [pd.concat(list(monthly_rtn_dict.values())), weekly_rtn_table]
else:
    additional_table = [pd.concat(list(monthly_rtn_dict.values()))]


bench_idx = input(
    "请键入基准：[0]无基准, [1]沪深300, [2]中证500, [3]中证100, [4]国证2000, [5]中证全指"
)
if bench_idx == "" or bench_idx == "0":
    bench_mark_nav = None
else:
    bench_symbol = [
        "",
        "SHSE.000300",
        "SHSE.000905",
        "SHSE.000852",
        "SZSE.399303",
        "SHSE.000985",
    ][int(bench_idx)]
    bench_mark_nav = index_data[index_data["symbol"] == bench_symbol]
    bench_mark_nav = (
        bench_mark_nav[["bob", "close"]]
        .set_index("bob")
        .reindex(trade_date)["close"]
        .values
    )

nav_compare_analysis(
    trade_date=trade_date,
    nav_data_dict={
        key: nav_data["累计净值"].values for key, nav_data in nav_data_dict.items()
    },
    bench_mark_nav=bench_mark_nav,
    html_file_name=html_file_path,
    additional_table=additional_table,
    origin_date=origin_date,
)
os.system(f"start {html_file_path}")
