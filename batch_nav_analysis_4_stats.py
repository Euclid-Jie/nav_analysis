from utils import *
import itertools
from single_nav_analysis import SingleNavAnalysis


class BatchNavAnalysis:
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
            for path_i in itertools.chain(
                self.nav_file_path.rglob("*.xlsx"),
                self.nav_file_path.rglob("*.xls"),
                self.nav_file_path.rglob("*.csv"),
            )
        ]
        assert (
            len(self.nav_file_paths) > 0
        ), f"{self.nav_file_path}至少有一个净值数据文件"

        self.begin_date = self.nav_analysis_config.begin_date
        self.end_date = self.nav_analysis_config.end_date
        self.nav_data_dict = {}
        self.nav_dict = {}
        self.drawdown_dict = {}

    @staticmethod
    def weekly_yearly_rtn(
        nav: np.ndarray,
        date: np.datetime64,
        last_week_date: np.datetime64 = np.datetime64("2024-12-20"),
        last_year_date: np.datetime64 = np.datetime64("2023-12-29"),
    ) -> Tuple[float, float]:
        """
        计算最近一周和今年以来收益率
        """
        # TODO: 显然last_week_date\last_week_date可以自动计算, 使用 generate_trading_date
        # 最近一周收益率
        week_nav = nav[date >= last_week_date]
        if len(week_nav) == 0:
            week_rtn = np.nan
        else:
            week_rtn = week_nav[-1] / week_nav[0] - 1

        # 今年以来收益率
        year_nav = nav[date >= last_year_date]
        if len(year_nav) == 0:
            year_rtn = np.nan
        else:
            year_rtn = year_nav[-1] / year_nav[0] - 1
        return week_rtn, year_rtn

    @classmethod
    def generate_yearly_rtn_df(
        cls, single_nav_analysis: SingleNavAnalysis
    ) -> pd.DataFrame:
        """ """
        year_rtn_df = single_nav_analysis.monthly_rtn_df[["年度总收益"]]
        year_rtn_df.index = year_rtn_df.index.str.split("_").str[-1]
        year_rtn_df = year_rtn_df.T
        year_rtn_df = year_rtn_df.reset_index(drop=True)
        year_rtn_df.index = [single_nav_analysis.name]
        return year_rtn_df

    def anlysis(self):
        res = pd.DataFrame()
        for path_i in self.nav_file_paths:
            nav_analysis_config: NavAnalysisConfig = self.nav_analysis_config.copy(
                begin_date=self.begin_date,
                end_date=self.end_date,
                nav_data_path=path_i,
                open_html=False,
            )
            single_nav_analysis = SingleNavAnalysis(nav_analysis_config)
            single_nav_analysis.analysis()
            single_nav_analysis.name = (
                "超额_" if self.nav_analysis_config.benchmark else ""
            ) + single_nav_analysis.name
            week_rtn, year_rtn = self.weekly_yearly_rtn(
                single_nav_analysis.nav, single_nav_analysis.date
            )
            add_data = pd.concat(
                [
                    display_df(single_nav_analysis.metrics_df),
                    display_df(
                        pd.DataFrame(
                            {
                                "最近一周收益率": week_rtn,
                                "今年以来收益率": year_rtn,
                                "净值开始时间": single_nav_analysis.nav_data[
                                    "日期"
                                ].min(),
                                "净值结束时间": single_nav_analysis.nav_data[
                                    "日期"
                                ].max(),
                                "最新累计单位净值": single_nav_analysis.nav[-1],
                            },
                            index=[single_nav_analysis.name],
                        )
                    ),
                    self.generate_yearly_rtn_df(single_nav_analysis),
                ],
                axis=1,
            )
            add_data["管理人"] = (
                keep_chinese_chars(single_nav_analysis.name)
                .replace("超额_", "")
                .replace("中性", "")
                .replace("灵活对冲", "")
            )
            res = pd.concat(
                [
                    res,
                    add_data,
                ]
            )

        return res


if __name__ == "__main__":
    save_folder = Path(r"C:\Users\Ouwei\Desktop\nav_data_stats")
    save_folder.mkdir(exist_ok=True)
    with pd.ExcelWriter(save_folder.joinpath("all_20241227.xlsx")) as xlsx:
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
                    rf"C:\Users\Ouwei\Desktop\nav_data\净值库1227\按策略分\{name}"
                ),
                benchmark=bench,
                ingnore_null=True,
                ingnore_duplicate=True,
            )
            demo = BatchNavAnalysis(
                nav_analysis_config,
            )
            res = demo.anlysis()
            res.index.name = "管理人及产品"
            res.to_csv(
                save_folder.joinpath(f"{name}.csv"), index=True, encoding="utf-8-sig"
            )
            res["策略类型"] = name
            res.to_excel(xlsx, sheet_name=name)
