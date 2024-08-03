from unitls import *

print("请选择净值数据文件，请确保列名为：日期or净值日期 / 累计净值or累计单位净值")
nav_file_path = Path(getLocalFile())
nav_data = pd.read_excel(nav_file_path)
nav_data = nav_data.rename(columns={"净值日期": "日期", "累计单位净值": "累计净值"})[
    ["日期", "单位净值", "累计净值"]
]
nav_data["日期"] = pd.to_datetime(nav_data["日期"])
nav_data = nav_data.sort_values(by="日期", ascending=True).reset_index(drop=True)
print(
    f"净值数据中时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
)
begin_date = input("请输入开始统计的日期[YYYY-MM-DD]：")
if begin_date == "":
    begin_date = nav_data["日期"].min()
else:
    begin_date = pd.to_datetime(begin_date)
    assert begin_date >= nav_data["日期"].min(), "开始统计日期早于净值数据最早日期"

nav_data = nav_data[nav_data["日期"] >= begin_date]

html_name = input("请输入导出的html文件名：")
if html_name == "":
    html_name = (
        datetime.date.today().strftime("%Y%m%d")
        + "_"
        + nav_file_path.stem
        + "_nav_analysis"
    )
html_file_path = nav_file_path.parent.joinpath(f"{html_name}.html")
print(f"html路径为：{html_file_path}")
### 按周统计收益
nav_data["rtn"] = nav_data["累计净值"].pct_change()
weekly_rtn = (
    nav_data.groupby(pd.Grouper(key="日期", freq="W"))
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

monthly_rtn = win_ratio_stastics(
    nav_data[["日期", "累计净值"]], start_date=np.datetime64("2023-01-01")
)

bench_idx = input(
    "请键入基准：[0]无基准, [1]沪深300, [2]中证500, [3]中证100, [4]国证2000, [5]中证全指"
)
if bench_idx == "" or bench_idx == "0":
    nav_analysis(
        date=nav_data["日期"].values,
        nav=nav_data["累计净值"].values,
        html_file_name=html_file_path,
        additional_table=[monthly_rtn, weekly_rtn_table],
        plot=False,
    )

else:
    bench_symbol = [
        "",
        "SHSE.000300",
        "SHSE.000905",
        "SHSE.000852",
        "SZSE.399303",
        "SHSE.000985",
    ][int(bench_idx)]
    nav_data = add_bench_data(
        nav_data,
        bench_symbol=bench_symbol,
        index_data_path=nav_file_path.parent.joinpath("index_data.csv"),
    )
    nav_analysis(
        date=nav_data["日期"].values,
        nav=nav_data["累计净值"].values,
        bench_mark_nav=nav_data["close"].values,
        html_file_name=html_file_path,
        additional_table=[monthly_rtn, weekly_rtn_table],
        plot=False,
    )
os.system(f"start {html_file_path}")
