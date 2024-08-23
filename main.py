import datetime
from utils import *

# --- global settings ---
nav_analysis_config = NavAnalysisConfig(
    index_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
    # nav_data_path=Path(r"c:\Users\Ouwei\Desktop\nav_data\双庆需求\磐松中性进取1号.xls"),
    # begin_date=pd.to_datetime("2023-12-29"),
    open_html=True,
    image_save_parh=None,
    # benchmark="SHSE.000905",
)

if nav_analysis_config.nav_data_path == None:
    print("请选择净值数据文件，请确保列名为：日期or净值日期 / 累计净值or累计单位净值")
    nav_file_paths = [Path(path_i) for path_i in getLocalFiles()]
    # nav_file_paths = [
    #     Path("C:/Users/Ouwei/Desktop/nav_data/净值0814/市场中性/天算中性B-SXU256.xlsx")
    # ]
else:
    assert nav_analysis_config.nav_data_path.exists(), input("未找到文件夹/文件")
    if nav_analysis_config.nav_data_path.is_file():
        nav_file_paths = [nav_analysis_config.nav_data_path]
    else:
        nav_file_paths = [
            Path(path_i)
            for path_i in nav_analysis_config.nav_data_path.glob("*.xlsx|*.xls")
        ]

assert len(nav_file_paths) > 0, input("未选择文件")

# 读取指数数据
if nav_analysis_config.index_data_path is not None:
    index_data = pd.read_csv(nav_analysis_config.index_data_path)
else:
    if Path(nav_file_paths[0].parent.joinpath("index_data.csv")).exists():
        index_data = pd.read_csv(nav_file_paths[0].parent.joinpath("index_data.csv"))
    elif Path(nav_file_paths[0].parent.parent.joinpath("index_data.csv")).exists():
        index_data = pd.read_csv(
            nav_file_paths[0].parent.parent.joinpath("index_data.csv")
        )
    else:
        print(
            f"未找到指数数据文件，请将指数数据文件放在{nav_file_paths[0].parent}或{nav_file_paths[0].parent.parent}下"
        )
        print("请手动选择指数数据文件")
        index_data = pd.read_csv(getLocalFiles()[0])
index_data["bob"] = pd.to_datetime(index_data["bob"]).dt.tz_localize(None)
trade_date = np.unique(index_data["bob"].values).astype("datetime64[ns]")

begin_date = pd.to_datetime("2000-06-06")
end_date = pd.to_datetime("2099-06-06")

# 读取数据并确定时间区间
nav_data_dict = {}
for path in nav_file_paths:
    print(f"【{path.stem}】")
    nav_data = format_nav_data(path)
    # 选取最大的开始时间作为开始时间
    if nav_data["日期"].min() >= begin_date:
        begin_date = nav_data["日期"].min()
    if nav_data["日期"].max() <= end_date:
        end_date = nav_data["日期"].max()
    print(
        f"净值数据中，时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
    )
    nav_data_dict[path.stem] = nav_data

# 对时间区间进行修改
if nav_analysis_config.begin_date is not None:
    assert nav_analysis_config.begin_date >= begin_date, input(
        "开始统计日期早于净值数据最早日期"
    )
    print(f"统计时间将从{begin_date.strftime('%Y-%m-%d')}开始")
    begin_date = nav_analysis_config.begin_date

if nav_analysis_config.end_date is not None:
    assert nav_analysis_config.end_date <= end_date, input(
        "结束统计日期晚于净值数据最晚日期"
    )
    print(f"统计时间将结束于{end_date.strftime('%Y-%m-%d')}")
    end_date = nav_analysis_config.end_date

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
monthly_rtn_df = pd.concat(list(monthly_rtn_dict.values()))
for col in monthly_rtn_df.columns:
    monthly_rtn_df[col] = monthly_rtn_df[col].map(lambda x: f"{x:.3%}")

trade_date = trade_date[trade_date >= begin_date]
trade_date = trade_date[trade_date <= end_date]

if nav_analysis_config.special_html_name:
    html_name = input("请输入导出的html文件名：")
else:
    html_name = ""

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
html_file_path = Path(nav_file_paths[0].parent.joinpath(f"{html_name}.html"))
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
    ].tail(10)
    weekly_rtn_table["日期"] = weekly_rtn_table["日期"].dt.strftime("%Y-%m-%d")
    weekly_rtn_table.set_index("日期", inplace=True)
    weekly_rtn_table = weekly_rtn_table.T

    backword_analysis_df = backword_analysis(list(nav_data_dict.values())[0])
    for col in ["区间收益率", "年化收益", "年化波动率", "最大回撤"]:
        backword_analysis_df[col] = backword_analysis_df[col].map(lambda x: f"{x:.3%}")
    backword_analysis_df["夏普比率"] = backword_analysis_df["夏普比率"].apply(
        lambda x: f"{x:.3f}"
    )
    additional_table = [monthly_rtn_df, weekly_rtn_table, backword_analysis_df]
else:
    additional_table = [monthly_rtn_df]


if nav_analysis_config.benchmark is not None:
    bench_mark_nav = index_data[index_data["symbol"] == nav_analysis_config.benchmark]
    bench_mark_nav = (
        bench_mark_nav[["bob", "close"]]
        .set_index("bob")
        .reindex(trade_date)["close"]
        .values
    )
else:
    bench_mark_nav = None

nav_compare_analysis(
    trade_date=trade_date,
    nav_data_dict={
        key: nav_data["累计净值"].values for key, nav_data in nav_data_dict.items()
    },
    bench_mark_nav=bench_mark_nav,
    html_file_name=html_file_path,
    additional_table=additional_table,
    origin_date=origin_date,
    image_save_path=nav_analysis_config.image_save_parh,
)
if nav_analysis_config.open_html:
    input("导出完成，按任意键打开html文件")
    os.system(f"start {html_file_path.__str__()}")
