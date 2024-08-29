import datetime
from utils import *

# --- global settings ---
nav_analysis_config = NavAnalysisConfig(
    index_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
    begin_date=pd.to_datetime("2023-12-29"),
    open_html=False,
    benchmark="SHSE.000852",
    image_save_path=Path(r"C:\Users\Ouwei\Desktop\nav_data\净值库0827\image\1000增强"),
    nav_data_path=Path(r"C:\Users\Ouwei\Desktop\nav_data\净值库0827\按策略分\1000增强"),
    overwrite=False,
)

nav_file_paths = get_nav_file_paths(nav_analysis_config.nav_data_path)

# 读取指数数据
index_data = pd.read_csv(nav_analysis_config.index_data_path)
index_data["bob"] = pd.to_datetime(index_data["bob"]).dt.tz_localize(None)
trade_date = np.unique(index_data["bob"].values).astype("datetime64[ns]")

for file_path in nav_file_paths:
    print("-*-" * 24)
    print(f"【{file_path.stem}】")
    begin_date = nav_analysis_config.begin_date
    end_date = nav_analysis_config.end_date
    trade_date = np.unique(index_data["bob"].values).astype("datetime64[ns]")
    # 读取数据并确定时间区间
    nav_data_dict = {}
    nav_data = format_nav_data(file_path)
    # 选取最大的开始时间作为开始时间
    if nav_data["日期"].min() >= begin_date:
        begin_date = nav_data["日期"].min()
    if nav_data["日期"].max() <= end_date:
        end_date = nav_data["日期"].max()
    print(
        f"净值数据中，时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
    )

    origin_date = nav_data["日期"]
    origin_date = origin_date[origin_date >= begin_date]
    origin_date = origin_date[origin_date <= end_date]
    nav_data = match_data(nav_data=nav_data, trade_date=trade_date)
    nav_data = nav_data[nav_data["日期"] >= begin_date]
    nav_data = nav_data[nav_data["日期"] <= end_date]

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
    if html_file_path.exists() and nav_analysis_config.overwrite == False:
        print("文件已存在, 直接跳过")
        continue
    nav_data.to_csv(
        file_path.parent.joinpath(f"formated_nav_data_{html_name}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if nav_analysis_config.benchmark is not None:
        bench_mark_nav = index_data[
            index_data["symbol"] == nav_analysis_config.benchmark
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
        nav_data_dict={keep_chinese_chars(file_path.stem): nav_data["累计净值"].values},
        bench_mark_nav=bench_mark_nav,
        bench_mark_name=nav_analysis_config.benchmark,
        html_file_name=html_file_path,
        origin_date=origin_date,
        image_save_path=nav_analysis_config.image_save_path,
    )
