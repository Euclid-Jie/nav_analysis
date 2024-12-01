import datetime
from utils import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from attribution_analysis import AttractionAnalysis

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class SingleNavAnalysis:
    def __init__(self, nav_analysis_config: NavAnalysisConfig):
        self.nav_analysis_config = nav_analysis_config
        self.nav_file_path = (
            self.nav_analysis_config.nav_data_path
            if self.nav_analysis_config.nav_data_path
            else getLocalFile(log=False)
        )
        if self.nav_analysis_config.benchmark == "":
            self.specify_benchmark()

        self.name = self.nav_file_path.stem
        self.begin_date = self.nav_analysis_config.begin_date
        self.end_date = self.nav_analysis_config.end_date
        self.nav_dict = {}
        self.drawdown_dict = {}
        self.attraction_analysis_html = ""
        self.load_data()
        self.select_date()
        self.trade_date, self.weekly_trade_date = generate_trading_date(
            self.begin_date - np.timedelta64(10, "D"), self.end_date
        )
        # 根据数据频率对数据进行日期的规范化
        if self.freq == "D":
            self.nav_data = match_data(self.nav_data, self.trade_date)
        else:  # freq is "W"
            self.nav_data = match_data(self.nav_data, self.weekly_trade_date)

    def specify_benchmark(self):
        self.nav_analysis_config = self.nav_analysis_config.copy(
            benchmark=[
                None,
                "SHSE.000300",
                "SHSE.000905",
                "SHSE.000852",
                "SZSE.399303",
                "SHSE.000985",
            ][
                int(
                    input(
                        "请键入基准：[0]无基准, [1]沪深300, [2]中证500, [3]中证100, [4]国证2000, [5]中证全指"
                    )
                )
            ]
        )

    def __repr__(self) -> str:
        return f"{self.name}净值分析[{self.freq}]：{self.begin_date} ~ {self.end_date}"

    def load_data(self):
        if self.nav_analysis_config.benchmark is not None:
            self.bench_data: pd.DataFrame = load_bench_data(
                self.nav_analysis_config.bench_data_path
            )
            self.bench_data = self.bench_data[
                self.bench_data["symbol"] == self.nav_analysis_config.benchmark
            ][["bob", "close"]]
        else:
            self.bench_data: pd.DataFrame = None
        print(f"开始读取{self.name}净值数据")
        self.nav_data: pd.DataFrame = format_nav_data(self.nav_file_path)

        self.freq = infer_frequency(self.nav_data["日期"].values)

        print(
            f"原始数据时间区间为：{self.nav_data['日期'].min().strftime('%Y-%m-%d')} ~ {self.nav_data['日期'].max().strftime('%Y-%m-%d')}"
        )

    def select_date(self):
        # 选取最大的开始时间作为开始时间
        if self.nav_data["日期"].min() >= self.nav_analysis_config.begin_date:
            self.begin_date: np.datetime64 = np.datetime64(
                self.nav_data["日期"].min(), "D"
            )
        if self.nav_data["日期"].max() <= self.nav_analysis_config.end_date:
            self.end_date: np.datetime64 = np.datetime64(
                self.nav_data["日期"].max(), "D"
            )
        print(f"本次统计时间区间为：{self.begin_date} ~ {self.end_date}")
        if self.nav_analysis_config.benchmark is not None:
            self.bench_data = self.bench_data[
                (self.bench_data["bob"] >= self.begin_date)
                & (self.bench_data["bob"] <= self.end_date)
            ]

        self.nav_data = self.nav_data[
            (self.nav_data["日期"] >= self.begin_date)
            & (self.nav_data["日期"] <= self.end_date)
        ]

    def analysis(self):
        # NOTE nav
        self.nav = self.nav_data["累计净值"].values
        self.nav = self.nav / self.nav[0]
        self.date = self.nav_data["日期"].values.astype("datetime64[D]")

        drawdown, self.drawdown_info = drawdown_stats(self.nav, self.date)
        self.max_drawdown_info = self.drawdown_info[
            self.drawdown_info["max_drawdown"]
            == self.drawdown_info["max_drawdown"].min()
        ]
        self.max_drawdown_info.index = [f"{self.name}"]

        self.drawdown_dict.update({self.name: drawdown})
        self.nav_dict.update({self.name: self.nav})

        if self.nav_analysis_config.benchmark is not None:
            # NOTE benchmark
            self.bench_trade_date = self.bench_data["bob"].values.astype(
                "datetime64[D]"
            )
            self.bench_nav = self.bench_data["close"].values[
                np.isin(self.bench_trade_date, self.date)
            ]
            self.bench_nav = self.bench_nav / self.bench_nav[0]
            bench_drawdown, _ = drawdown_stats(self.bench_nav, self.date)
            self.nav_dict.update({self.nav_analysis_config.benchmark: self.bench_nav})
            self.drawdown_dict.update(
                {self.nav_analysis_config.benchmark: bench_drawdown}
            )
            # NOTE excess = nav / bench
            self.excess_nav = calc_excess_nav(self.nav, self.bench_nav)
            drawdown, drawdown_info = drawdown_stats(self.excess_nav, self.date)
            self.max_drawdown_info = drawdown_info[
                drawdown_info["max_drawdown"] == drawdown_info["max_drawdown"].min()
            ]
            self.max_drawdown_info.index = [f"超额_{self.name}"]
            self.nav_dict.update({f"超额_{self.name}": self.excess_nav})
            self.drawdown_dict.update({f"超额_{self.name}": drawdown})

            # metrics of excess
            self.metrics_df = pd.DataFrame(
                curve_analysis(self.excess_nav, freq=self.freq),
                index=[f"超额_{self.name}"],
            )

            # NOTE some analysis of excess
            self.monthly_rtn_df = win_ratio_stastics(self.excess_nav, self.date)
            self.monthly_rtn_df.index = [
                f"超额_{self.name}_{i}" for i in self.monthly_rtn_df.index
            ]
            self.weekly_rtn_df = weekly_rtn_stats(self.excess_nav, self.date)
            self.weekly_rtn_df.index = [
                f"超额_{self.name}_{i}" for i in self.weekly_rtn_df.index
            ]
            self.backword_analysis_df = backword_analysis(
                self.excess_nav, self.date, freq=self.freq
            )
            self.backword_analysis_df.index = [
                f"超额_{self.name}_{i}" for i in self.backword_analysis_df.index
            ]
        else:
            # metrics of nav
            self.metrics_df = pd.DataFrame(
                curve_analysis(self.nav, freq=self.freq), index=[f"{self.name}"]
            )
            # NOTE some analysis of nav
            self.monthly_rtn_df = win_ratio_stastics(self.nav, self.date)
            self.monthly_rtn_df.index = [
                f"{self.name}_{i}" for i in self.monthly_rtn_df.index
            ]
            self.weekly_rtn_df = weekly_rtn_stats(self.nav, self.date)
            self.weekly_rtn_df.index = [
                f"{self.name}_{i}" for i in self.weekly_rtn_df.index
            ]
            self.backword_analysis_df = backword_analysis(
                self.nav, self.date, freq=self.freq
            )
            self.backword_analysis_df.index = [
                f"{self.name}_{i}" for i in self.backword_analysis_df.index
            ]

        self.metrics_table = display_df(
            pd.concat(
                [
                    self.metrics_df[
                        ["年化收益率", "年化波动率", "最大回撤", "夏普比率", "卡玛比率"]
                    ],
                    self.max_drawdown_info[
                        [
                            "drawdown_start_date",
                            "max_drawdown_date",
                            "max_drawdown_days",
                            "drawdown_end_date",
                            "drawdown_fix_days",
                        ]
                    ].rename(
                        columns={
                            "drawdown_start_date": "开始日期",
                            "max_drawdown_date": "结束日期",
                            "max_drawdown_days": "持续天数",
                            "drawdown_end_date": "修复日期",
                            "drawdown_fix_days": "修复天数",
                        }
                    ),
                ],
                axis=1,
            )
        )
        self.nav_df = pd.DataFrame(self.nav_dict, index=self.date)

    def attraction_analysis(self, barra_rtn_df: pd.DataFrame, show=False, R2=True):
        # NOTE attraction analysis
        barra_rtn_df, seleclted_date = self.reindex_rtn_df(barra_rtn_df, self.date)
        attraction_analysis = AttractionAnalysis(
            nav=self.nav[np.isin(self.date, seleclted_date)],
            date=seleclted_date,
            Xs=[barra_rtn_df[col_i].values for col_i in barra_rtn_df.columns],
            Xs_name=barra_rtn_df.columns,
            windows=13,
            title=self.name,
        )
        if seleclted_date[-1] < self.date[-1]:
            print(
                f"Warning: barra data end date is {np.datetime_as_string(seleclted_date[-1], unit='D')}, please update barra data"
            )
        attraction_analysis.analyze()
        fig = attraction_analysis.plot(show=show, R2=R2)
        fig.write_html("demo.html", full_html=False)
        with open("demo.html", "r", encoding="utf-8") as f:
            self.attraction_analysis_html = f.read()
        # del tmp file
        try:
            os.remove("demo.html")
        except FileNotFoundError:
            pass

    @staticmethod
    def reindex_rtn_df(barra_rtn_df, date):
        assert "日期" in barra_rtn_df.columns, "日期列不存在"
        barra_rtn_df["日期"] = pd.to_datetime(barra_rtn_df["日期"], format="%Y-%m-%d")
        barra_rtn_df.set_index("日期", inplace=True)
        seleclted_date = np.intersect1d(date, barra_rtn_df.index)
        barra_nav_df = (barra_rtn_df + 1).cumprod().reindex(seleclted_date)
        barra_rtn_df = barra_nav_df.copy()
        for coli in barra_nav_df.columns:
            barra_rtn_df[coli] = calc_nav_rtn(barra_rtn_df[coli].values)
        barra_rtn_df = barra_rtn_df.iloc[1:]
        return barra_rtn_df, seleclted_date[1:]

    def export_html(self, save=True):
        self.html = nav_analysis_echarts_plot(
            date=self.date,
            nav=self.nav_dict,
            drawdown=self.drawdown_dict,
            table=self.metrics_table,
            additional_table=[
                self.monthly_rtn_df,
                self.weekly_rtn_df,
                self.backword_analysis_df,
            ],
            attraction_analysis_html=self.attraction_analysis_html,
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
                    + self.name
                    + "_nav_analysis"
                )
            html_file_path = Path(
                self.nav_file_path.parent.joinpath(f"{html_name}.html")
            )
            print(f"html路径为：{html_file_path}")
            with open(html_file_path, "w", encoding="utf-8") as f:
                f.write(self.html)

            if self.nav_analysis_config.open_html:
                input("导出完成，按任意键打开html文件")
                os.system(f"start {html_file_path.__str__()}")

    def plot(self, where: Literal["lower", "upper"] = "lower"):
        _, (ax1, ax2) = plt.subplots(nrows=2, figsize=(25, 14))
        # 画超额收益图
        ax1.plot(self.date, self.nav - 1, color="red", label=f"{self.name}_累计净值")
        if self.nav_analysis_config.benchmark:
            ax1.plot(
                self.date,
                self.bench_nav - 1,
                color="blue",
                label=self.nav_analysis_config.benchmark,
            )
            ax1.fill_between(
                self.date, self.excess_nav - 1, color="gray", label="超额收益"
            )
        ax1.legend(loc=f"{where} left", fontsize=15)
        ax1.tick_params(axis="x", rotation=45, labelsize=15)
        ax1.tick_params(axis="y", labelsize=15)
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.set_title("累计净值及动态回撤图", size=25)
        ax1.grid()

        # 画动态回撤图
        ax2.plot(
            self.date,
            self.drawdown_dict[self.name],
            color="red",
            label=f"{self.name}_回撤",
        )
        if self.nav_analysis_config.benchmark:
            ax2.plot(
                self.date,
                self.drawdown_dict[self.nav_analysis_config.benchmark],
                color="blue",
                label=f"{self.nav_analysis_config.benchmark}_回撤",
            )
            ax2.fill_between(
                self.date,
                self.drawdown_dict[f"超额_{self.name}"],
                color="gray",
                label="超额回撤",
            )
        ax2.legend(loc="lower left", fontsize=15)
        ax2.tick_params(axis="x", rotation=45, labelsize=15)
        ax2.tick_params(axis="y", labelsize=15)
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.grid()


if __name__ == "__main__":
    nav_analysis_config = NavAnalysisConfig(
        bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
        nav_data_path=Path(r"nav_data\ABA86A-弈倍龙杉九号A类净值序列.xlsx"),
        begin_date=np.datetime64("2023-12-29"),
        open_html=True,
        benchmark="SHSE.000905",
    )
    demo = SingleNavAnalysis(nav_analysis_config)
    demo.analysis()
    demo.attraction_analysis(
        barra_df=pd.read_csv(r"barra_data\cne6\merged_cne6_barra_factor_rtn.csv"),
        show=False,
        R2=False,
    )
    demo.export_html()
