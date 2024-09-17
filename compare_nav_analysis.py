import datetime
from utils import *
from single_nav_analysis import SingleNavAnalysis


class CompareNavAnalysis(SingleNavAnalysis):
    def __init__(self, nav_analysis_config: NavAnalysisConfig):
        self.nav_analysis_config = nav_analysis_config
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
        self.nav_data_dict = {}
        self.nav_dict = {}
        self.drawdown_dict = {}
        self.load_data()
        self.select_date()

    def __repr__(self) -> str:
        return f"净值对比分析：{self.begin_date} ~ {self.end_date}"

    def load_data(self):
        self.bench_data: pd.DataFrame = load_bench_data(
            self.nav_analysis_config.bench_data_path
        )
        for path_i in self.nav_file_paths:
            self.nav_data_dict[path_i.stem] = format_nav_data(path_i)

    def select_date(self):
        for name, nav_data in self.nav_data_dict.items():
            # 选取最大的开始时间作为开始时间
            if nav_data["日期"].min() >= self.begin_date:
                self.begin_date = np.datetime64(nav_data["日期"].min(), "D")
            if nav_data["日期"].max() <= self.end_date:
                self.end_date = np.datetime64(nav_data["日期"].max(), "D")
            print(
                f"{name}，原始数据时间区间为：{nav_data['日期'].min().strftime('%Y-%m-%d')} - {nav_data['日期'].max().strftime('%Y-%m-%d')}"
            )
        assert self.begin_date < self.end_date, input("不存在交集时间区间")
        print(f"本次统计时间区间为：{self.begin_date} ~ {self.end_date}")
        self.trade_date = np.unique(self.bench_data["bob"].values).astype(
            "datetime64[ns]"
        )
        self.trade_date = self.trade_date[
            (self.trade_date >= self.begin_date) & (self.trade_date <= self.end_date)
        ].astype("datetime64[D]")

    def anlysis(self):
        self.metrics_table = pd.DataFrame()
        self.monthly_rtn_df = pd.DataFrame()
        self.weekly_rtn_df = pd.DataFrame()
        self.backword_analysis_df = pd.DataFrame()
        for path_i in self.nav_file_paths:
            nav_analysis_config: NavAnalysisConfig = self.nav_analysis_config.copy(
                begin_date=self.begin_date,
                end_date=self.end_date,
                nav_data_path=path_i,
                open_html=False,
            )
            single_nav_analysis = SingleNavAnalysis(nav_analysis_config)
            single_nav_analysis.analysis()
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

    def export_html(self):
        if self.nav_analysis_config.special_html_name:
            html_name = input("请输入导出的html文件名：")
        else:
            html_name = (
                datetime.date.today().strftime("%Y%m%d") + "_" + "compare_nav_analysis"
            )
        html_file_path = Path(
            self.nav_file_paths[0].parent.joinpath(f"{html_name}.html")
        )
        print(f"html路径为：{html_file_path}")

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
        )
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(self.html)

        if self.nav_analysis_config.open_html:
            input("导出完成，按任意键打开html文件")
            os.system(f"start {html_file_path.__str__()}")


if __name__ == "__main__":
    nav_analysis_config = NavAnalysisConfig(
        bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
        # nav_data_path=[
        #     Path(
        #         r"C:\Users\Ouwei\Desktop\nav_data\SGB773_麦迪生利锐联中性对冲1号.xlsx"
        #     ),
        #     Path(
        #         r"C:/Users/Ouwei/Desktop/nav_data/净值0814/市场中性/天算中性B-SXU256.xlsx"
        #     ),
        # ],
        begin_date=np.datetime64("2023-12-29"),
        open_html=True,
        benchmark="",
    )
    demo = CompareNavAnalysis(nav_analysis_config)
    demo.anlysis()
    demo.export_html()
