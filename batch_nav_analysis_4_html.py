from utils import *
from single_nav_analysis import SingleNavAnalysis

class BatchNavAnalysis(SingleNavAnalysis):
    def __init__(
        self,
        nav_analysis_config: NavAnalysisConfig,
    ):
        self.nav_analysis_config = nav_analysis_config
        self.nav_file_path = (
            self.nav_analysis_config.nav_data_path
            if self.nav_analysis_config.nav_data_path
            else getLocalFile(log=False)
        )
        assert self.nav_file_path.is_dir(), "batch analysis 以文件夹为最小分析单位"
        self.nav_file_paths = [
            Path(path_i)
            for path_i in self.nav_file_path.glob("*.xlsx" or "*.xls" or "csv")
        ]
        assert (
            len(self.nav_file_paths) > 0
        ), f"{self.nav_file_path}至少有一个净值数据文件"

        self.begin_date = self.nav_analysis_config.begin_date
        self.end_date = self.nav_analysis_config.end_date
        self.nav_data_dict = {}
        self.nav_dict = {}
        self.drawdown_dict = {}
        self.load_data()
        self.select_date()

    def load_data(self):
        self.bench_data: pd.DataFrame = load_bench_data(
            self.nav_analysis_config.bench_data_path
        )
        for path_i in self.nav_file_paths:
            print(f"开始读取{path_i.stem}净值数据")
            self.nav_data_dict[path_i.stem] = format_nav_data(path_i)

    def select_date(self):
        for _, nav_data in self.nav_data_dict.items():
            if nav_data["日期"].max() <= self.end_date:
                self.end_date = np.datetime64(nav_data["日期"].max(), "D")
        print(f"本次统计时间区间为：{self.begin_date} ~ {self.end_date}")
        self.trade_date = np.unique(self.bench_data["bob"].values).astype(
            "datetime64[ns]"
        )
        self.trade_date = self.trade_date[
            (self.trade_date >= self.begin_date) & (self.trade_date <= self.end_date)
        ].astype("datetime64[D]")

    def anlysis(self):
        for path_i in self.nav_file_paths:
            nav_analysis_config: NavAnalysisConfig = self.nav_analysis_config.copy(
                begin_date=self.begin_date,
                end_date=self.end_date,
                nav_data_path=path_i,
                open_html=False,
            )
            single_nav_analysis = SingleNavAnalysis(nav_analysis_config)
            single_nav_analysis.analysis()
            single_nav_analysis.export_html(save=True)


if __name__ == "__main__":
    for name, bench in {
        "市场中性": None,
        "量化选股": None,
        "300增强": "SHSE.000300",
        "500增强": "SHSE.000905",
        "1000增强": "SHSE.000852",
    }.items():
        nav_analysis_config = NavAnalysisConfig(
            bench_data_path=Path(
                r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"
            ),
            nav_data_path=Path(
                rf"C:\Users\Ouwei\Desktop\nav_data\净值库1101\按策略分\{name}"
            ),
            begin_date=np.datetime64("2023-12-29"),
            benchmark=bench,
        )
        demo = BatchNavAnalysis(
            nav_analysis_config,
            rewrite=False,
        )
        demo.anlysis()
