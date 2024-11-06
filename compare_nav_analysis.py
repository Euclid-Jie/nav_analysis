from utils import *
from single_nav_analysis import SingleNavAnalysis

class CompareNavAnalysis(SingleNavAnalysis):
    def __init__(self, nav_analysis_config: NavAnalysisConfig, strip_date:bool=True):
        self.nav_analysis_config = nav_analysis_config
        if self.nav_analysis_config.begin_date == np.datetime64("2000-06-06") and self.strip_date:
            raise ValueError("当使用strip_date时，必须指定begin_date")
        self.nav_file_paths = (
            self.nav_analysis_config.nav_data_path
            if self.nav_analysis_config.nav_data_path
            else getLocalFiles(log=False)
        )
        assert len(self.nav_file_paths) >= 2, input("至少选择两个文件")
        if self.nav_analysis_config.benchmark == "":
            self.specify_benchmark()
        self.begin_date = self.nav_analysis_config.begin_date
        self.end_date = self.nav_analysis_config.end_date
        self.strip_date = strip_date
        self.nav_data_dict = {}
        self.nav_dict = {}
        self.drawdown_dict = {}
        self.weely_display = False
        self.load_data()
        self.select_date()

    def __repr__(self) -> str:
        return f"{" AND ".join([i for i in self.nav_dict.keys()])} 净值对比分析：{self.begin_date} ~ {self.end_date}"

    def load_data(self):
        self.bench_data: pd.DataFrame = load_bench_data(
            self.nav_analysis_config.bench_data_path
        )
        for path_i in self.nav_file_paths:
            data = format_nav_data(path_i)
            if data["日期"].min() > self.begin_date and self.strip_date:
                print(f"{path_i.stem}晚于开始时间，已删除")
                continue
            self.nav_data_dict[path_i.stem] = data

    def select_date(self):
        self.trade_date, self.weekly_trade_date = generate_trading_date(
            self.begin_date, self.end_date
        )
        # TODO 优化时间区间选择逻辑, 有可能无重叠部分(suppose不应该)
        # 记录所有净值数据的时间区间中重叠的部分
        exit_trade_date = self.trade_date.copy()
        exit_trade_date = exit_trade_date[exit_trade_date >= self.begin_date]
        for name, nav_data in self.nav_data_dict.items():
            exit_trade_date = np.intersect1d(exit_trade_date, nav_data["日期"].values)
            # 选取两者中min(max)作为结束时间
            if nav_data["日期"].max() <= self.end_date:
                self.end_date = np.datetime64(nav_data["日期"].max(), "D")
            print(
                f"{name}，原始数据时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
            )
        # 选取两者中max(min)作为开始时间
        assert exit_trade_date.size > 0, input("不存在交集时间区间")
        self.begin_date = np.datetime64(exit_trade_date.min(), "D")
        assert self.begin_date < self.end_date, input("不存在交集时间区间")
        print(f"本次统计时间区间为：{self.begin_date} ~ {self.end_date}")
        self.trade_date = self.trade_date[
            (self.trade_date >= self.begin_date) & (self.trade_date <= self.end_date)
        ]
        self.weekly_trade_date = self.weekly_trade_date[
            (self.weekly_trade_date >= self.begin_date) & (self.weekly_trade_date <= self.end_date)
        ]

    def anlysis(self):
        self.metrics_table = pd.DataFrame()
        self.monthly_rtn_df = pd.DataFrame()
        self.weekly_rtn_df = pd.DataFrame()
        self.backword_analysis_df = pd.DataFrame()
        for path_i in self.nav_file_paths:
            if path_i.stem not in self.nav_data_dict.keys():
                continue
            nav_analysis_config: NavAnalysisConfig = self.nav_analysis_config.copy(
                begin_date=self.begin_date,
                end_date=self.end_date,
                nav_data_path=path_i,
                open_html=False,
            )
            single_nav_analysis = SingleNavAnalysis(nav_analysis_config)
            single_nav_analysis.analysis()
            if self.weely_display is False and single_nav_analysis.freq == "W":
                self.weely_display = True
            self.metrics_table = pd.concat(
                [self.metrics_table, single_nav_analysis.metrics_table]
            )
            self.monthly_rtn_df = pd.concat(
                [self.monthly_rtn_df, single_nav_analysis.monthly_rtn_df]
            )
            self.weekly_rtn_df = pd.concat(
                [self.weekly_rtn_df, single_nav_analysis.weekly_rtn_df]
            )
            self.backword_analysis_df = pd.concat(
                [self.backword_analysis_df, single_nav_analysis.backword_analysis_df]
            )
            for nav_name_i, nav_i in single_nav_analysis.nav_dict.items():
                match_nav_i = np.ones_like(self.trade_date, dtype=float) * np.nan
                match_nav_i[
                    np.where(np.isin(self.trade_date, single_nav_analysis.date))
                ] = nav_i
                if np.isnan(match_nav_i[0]):
                    match_nav_i[0] = 1
                self.nav_dict[nav_name_i] = ffill(match_nav_i)
            for (
                drawdown_name_i,
                drawdown_i,
            ) in single_nav_analysis.drawdown_dict.items():
                match_drawdown_i = np.ones_like(self.trade_date, dtype=float) * np.nan
                match_drawdown_i[
                    np.where(np.isin(self.trade_date, single_nav_analysis.date))
                ] = drawdown_i
                self.drawdown_dict[drawdown_name_i] = ffill(match_drawdown_i)

    def export_html(self, save=True):
        self.html = nav_analysis_echarts_plot(
            date=self.trade_date,
            nav=self.nav_dict,
            drawdown=self.drawdown_dict,
            table=self.metrics_table,
            additional_table=[
                self.monthly_rtn_df,
                self.weekly_rtn_df,
                self.backword_analysis_df,
            ],
            select_date=self.weekly_trade_date if self.weely_display else None,
        )
        if save:
            if self.nav_analysis_config.special_html_name:
                html_name = input("请输入导出的html文件名：")
            else:
               html_name = (
                    np.datetime_as_string(self.begin_date, unit="D").replace("-", "")
                    + "_"
                    + np.datetime_as_string(self.end_date, unit="D").replace("-", "")
                    + "_"
                    + "compare_nav_analysis"
                )
            html_file_path = Path(
                self.nav_file_paths[0].parent.joinpath(f"{html_name}.html")
            )
            print(f"html路径为：{html_file_path}")

            with open(html_file_path, "w", encoding="utf-8") as f:
                f.write(self.html)

            if self.nav_analysis_config.open_html:
                input("导出完成，按任意键打开html文件")
                os.system(f"start {html_file_path.__str__()}")

if __name__ == "__main__":
    nav_analysis_config = NavAnalysisConfig(
        bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
        nav_data_path=[
            Path(
                r"nav_data\STC996-永誉天泽经纶二号净值序列.xlsx"
            ),
            Path(
                r"nav_data\SXL736-永誉天泽经纶四号净值序列.xlsx"
            ),
        ],
        begin_date=np.datetime64("2023-12-29"),
        open_html=True,
        # benchmark="SHSE.000905",
    )
    # 增加参数, 短于begin_date的数据会被删除
    demo = CompareNavAnalysis(nav_analysis_config)
    demo.anlysis()
    demo.export_html()
