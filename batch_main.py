import datetime
from utils import *

# --- global settings ---
nav_analysis_config = NavAnalysisConfig(
    begin_date=pd.to_datetime("2023-12-29"),
    open_html=False,
    nav_data_path=Path(r"C:\Users\Ouwei\Desktop\nav_data\净值0814\市场中性"),
)

if nav_analysis_config.nav_data_path == None:
    print("请选择净值数据文件，请确保列名为：日期or净值日期 / 累计净值or累计单位净值")
    nav_file_paths = [Path(path_i) for path_i in getLocalFiles()]
else:
    assert nav_analysis_config.nav_data_path.exists(), input("未找到文件夹/文件")
    if nav_analysis_config.nav_data_path.is_file():
        nav_file_paths = [nav_analysis_config.nav_data_path]
    else:
        nav_file_paths = [
            Path(path_i)
            for path_i in nav_analysis_config.nav_data_path.glob("*.xlsx" or "*.xls")
        ]

assert len(nav_file_paths) > 0, input("未选择文件")

# 读取指数数据
if Path(nav_file_paths[0].parent.joinpath("index_data.csv")).exists():
    index_data = pd.read_csv(nav_file_paths[0].parent.joinpath("index_data.csv"))
elif Path(nav_file_paths[0].parent.parent.joinpath("index_data.csv")).exists():
    index_data = pd.read_csv(nav_file_paths[0].parent.parent.joinpath("index_data.csv"))
else:
    print(
        f"未找到指数数据文件，请将指数数据文件放在{nav_file_paths[0].parent}或{nav_file_paths[0].parent.parent}下"
    )
    print("请手动选择指数数据文件")
    index_data = pd.read_csv(getLocalFiles()[0])
index_data["bob"] = pd.to_datetime(index_data["bob"]).dt.tz_localize(None)
trade_date = np.unique(index_data["bob"].values).astype("datetime64[ns]")

for file_path in nav_file_paths:
    begin_date = pd.to_datetime("2023-12-29")
    end_date = pd.to_datetime("2099-06-06")
    trade_date = np.unique(index_data["bob"].values).astype("datetime64[ns]")
    # 读取数据并确定时间区间
    nav_data_dict = {}
    nav_data = pd.read_excel(file_path)
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
    assert nav_data["累计净值"].isnull().sum() == 0, input(
        "Error: 净值数据中存在累计净值为空的数据"
    )
    assert nav_data["日期"].isnull().sum() == 0, input(
        "Error: 净值数据中存在日期为空的数据"
    )
    if nav_data["日期"].duplicated(keep=False).sum() != 0:
        print(
            "Info: 净值数据中存在日期重复的数据".format(
                nav_data[nav_data["日期"].duplicated()]
            )
        )
        nav_data = nav_data.drop_duplicates(subset="日期")
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
        f"{file_path.stem}净值数据中，时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
    )

    origin_date = nav_data["日期"]
    origin_date = origin_date[origin_date >= begin_date]
    origin_date = origin_date[origin_date <= end_date]
    nav_data = match_data(nav_data=nav_data, trade_date=trade_date)
    nav_data = nav_data[nav_data["日期"] >= begin_date]
    nav_data = nav_data[nav_data["日期"] <= end_date]
    nav_data["rtn"] = nav_data["累计净值"].pct_change()
    monthly_rtn = win_ratio_stastics(
        nav_data[["日期", "累计净值"]], start_date=begin_date
    )
    monthly_rtn.index = [f"{file_path.stem}_{i}" for i in monthly_rtn.index]
    for col in monthly_rtn.columns:
        monthly_rtn[col] = monthly_rtn[col].map(lambda x: f"{x:.3%}")

    trade_date = trade_date[trade_date >= begin_date]
    trade_date = trade_date[trade_date <= end_date]

    if nav_analysis_config.special_html_name:
        html_name = input("请输入导出的html文件名：")
    else:
        html_name = (
            begin_date.strftime("%Y%m%d")
            + "_"
            + end_date.strftime("%Y%m%d")
            + "_"
            + file_path.stem
        )
    html_file_path = Path(file_path.parent.joinpath(f"{html_name}.html"))
    print(f"html路径为：{html_file_path}")

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
    ].tail(10)
    weekly_rtn_table["日期"] = weekly_rtn_table["日期"].dt.strftime("%Y-%m-%d")
    weekly_rtn_table.set_index("日期", inplace=True)
    weekly_rtn_table = weekly_rtn_table.T
    additional_table = [monthly_rtn, weekly_rtn_table]

    if nav_analysis_config.bechmark is not None:
        bench_mark_nav = index_data[
            index_data["symbol"] == nav_analysis_config.bechmark
        ]
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
        nav_data_dict={file_path.stem: nav_data["累计净值"].values},
        bench_mark_nav=bench_mark_nav,
        html_file_name=html_file_path,
        additional_table=additional_table,
        origin_date=origin_date,
    )
    if nav_analysis_config.open_html:
        input("导出完成，按任意键打开html文件")
        os.system(f"start {html_file_path.__str__()}")
