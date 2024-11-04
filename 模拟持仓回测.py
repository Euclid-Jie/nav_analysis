import pandas as pd
from pathlib import Path
import numpy as np

plan_names = [
    "高波动FOF（1亿基准）_窄表",
    "积极型FOF（1亿基准）_窄表",
    "平衡性FOF（5亿基准）_窄表",
]
for plan_name in plan_names:
    nav_data = pd.read_excel(
        r"C:\Euclid_Jie\barra\src\nav_analysis\投资比例配置测算.xlsx",
        sheet_name="产品净值1018",
    )
    nav_data = nav_data[nav_data["日期"] >= pd.to_datetime("2019-12-30")]
    nav_data["日期"] = nav_data["日期"].dt.strftime("%Y-%m-%d")
    nav_data.set_index("日期", inplace=True)
    nav_data = nav_data.ffill(limit_area="inside")
    nav_data_format = nav_data / nav_data.bfill().iloc[0].values
    ret_data = nav_data_format.pct_change(fill_method=None)
    ret_data = ret_data.T.reset_index().rename(columns={"index": "产品"})

    book1 = pd.read_excel(
        r"C:\Euclid_Jie\barra\src\nav_analysis\投资比例配置测算.xlsx",
        sheet_name=f"{plan_name}",
    )
    book1 = book1.merge(ret_data, on="产品", how="left")
    book1_valid = book1[book1.iloc[:, 5:].notna().sum(axis=1) > 0]
    rtn = book1_valid.iloc[:, 5:].values
    money = book1_valid["金额"].values[:, None]

    weight = (~np.isnan(rtn) * money).sum(axis=0)[None, :]
    muti_rtn = np.nansum((rtn * money) / weight, axis=0)

    # TODO
    # 收益率缺失，则持有现金
    # rtn_nonan = rtn.copy()
    # rtn_nonan[np.isnan(rtn_nonan)] = 0
    # # 计算加权收益率
    # muti_rtn = np.sum((rtn_nonan * money) / money.sum(), axis=0)

    muti_nav = (1 + muti_rtn).cumprod()
    muti_nav_df_1 = pd.DataFrame(
        muti_nav, index=book1_valid.columns[5:], columns=[f"{plan_name}"]
    )
    muti_nav_df_1.reset_index(drop=False).rename(
        columns={"index": "日期", f"{plan_name}": "累计净值"}
    ).to_csv(
        rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}净值.csv",
        encoding="utf-8-sig",
    )

    detail_weight_2 = pd.DataFrame(
        np.where(np.isnan(rtn), np.nan, (~np.isnan(rtn) * money)),
        index=book1_valid["产品"].values,
        columns=book1_valid.columns[5:],
    ).T
    detail_weight_2.index = pd.to_datetime(detail_weight_2.index)
    detail_weight_2[1:].reset_index(drop=False).rename(
        columns={"index": "日期"}
    ).to_csv(
        rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}_持仓明细.csv",
        encoding="utf-8-sig",
        index=False,
    )

    with pd.ExcelWriter(
        rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}_年度平均持仓金额.xlsx",
        engine="openpyxl",
    ) as writer:
        for year in range(2020, 2025):
            res = (
                detail_weight_2[
                    (detail_weight_2.index >= pd.to_datetime(f"{year}-01-01"))
                    & (detail_weight_2.index <= pd.to_datetime(f"{year}-12-31"))
                ]
                .fillna(0)
                .mean(axis=0)
                .to_frame()
                .rename(columns={0: f"{year}年日平均持仓金额"})
                .sort_values(by=f"{year}年日平均持仓金额", ascending=False)
            )
            res.reset_index(drop=False).rename(columns={"index": "产品"}).reset_index(
                drop=True
            ).to_excel(
                writer,
                sheet_name=f"{plan_name}_{year}年度平均持仓金额",
                index=False,
            )

from single_nav_analysis import SingleNavAnalysis, NavAnalysisConfig
from compare_nav_analysis import CompareNavAnalysis

for plan_name in plan_names:
    nav_analysis_config = NavAnalysisConfig(
        bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
        nav_data_path=Path(
            rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}净值.csv"
        ),
        open_html=False,
    )
    demo = SingleNavAnalysis(nav_analysis_config)
    demo.analysis()
    demo.export_html(save=True)

    nav_analysis_config = NavAnalysisConfig(
        bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
        nav_data_path=[
            Path(rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}净值.csv")
            for plan_name in plan_names
        ],
        open_html=False,
    )
    demo = CompareNavAnalysis(nav_analysis_config)
    demo.anlysis()
    demo.export_html(save=True)

    for begin_date in ["2024-08-01", "2023-12-31", "2020-12-31"]:
        nav_analysis_config = NavAnalysisConfig(
            bench_data_path=Path(
                r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"
            ),
            nav_data_path=Path(
                rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}净值.csv"
            ),
            begin_date=np.datetime64(begin_date),
            open_html=False,
        )
        demo = SingleNavAnalysis(nav_analysis_config)
        demo.analysis()
        demo.export_html(save=True)

        nav_analysis_config = NavAnalysisConfig(
            bench_data_path=Path(
                r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"
            ),
            nav_data_path=[
                Path(rf"C:\Euclid_Jie\barra\src\nav_analysis\data\{plan_name}净值.csv")
                for plan_name in plan_names
            ],
            begin_date=np.datetime64("2023-12-31"),
            open_html=False,
        )
        demo = CompareNavAnalysis(nav_analysis_config)
        demo.anlysis()
        demo.export_html(save=True)
