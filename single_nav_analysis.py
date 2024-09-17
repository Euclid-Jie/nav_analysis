import datetime
from utils import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        self.name = self.nav_file_path.stem
        self.begin_date = self.nav_analysis_config.begin_date
        self.end_date = self.nav_analysis_config.end_date
        self.nav_dict = {}
        self.drawdown_dict = {}
        self.load_data()
        self.select_date()

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

        self.freq = (
            "W"
            if (self.nav_data["日期"].iloc[-1] - self.nav_data["日期"].iloc[0])
            > pd.Timedelta(len(self.nav_data["日期"]) * 365 / 251)
            else "D"
        )

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
            self.trade_date = self.bench_data["bob"].values.astype("datetime64[D]")
            self.bench_nav = self.bench_data["close"].values[
                np.isin(self.trade_date, self.date)
            ]
            self.bench_nav = self.bench_nav / self.bench_nav[0]
            bench_drawdown, _ = drawdown_stats(self.bench_nav, self.date)
            self.nav_dict.update({self.nav_analysis_config.benchmark: self.bench_nav})
            self.drawdown_dict.update(
                {self.nav_analysis_config.benchmark: bench_drawdown}
            )
            # NOTE excess = nav / bench
            self.excess_nav = self.nav / self.bench_nav
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
                        ["年化收益率", "年化波动率", "最大回撤", "夏普比率"]
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

    def export_html(self):
        if self.nav_analysis_config.special_html_name:
            html_name = input("请输入导出的html文件名：")
        else:
            html_name = (
                datetime.date.today().strftime("%Y%m%d")
                + "_"
                + self.name
                + "_nav_analysis"
            )
        html_file_path = Path(self.nav_file_path.parent.joinpath(f"{html_name}.html"))
        print(f"html路径为：{html_file_path}")

        html = nav_analysis_echarts_plot(
            date=self.date,
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
            f.write(html)

        if self.nav_analysis_config.open_html:
            input("导出完成，按任意键打开html文件")
            os.system(f"start {html_file_path.__str__()}")

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(25, 14))
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
        ax1.legend(loc="lower left", fontsize=15)
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
        nav_data_path=Path(
            r"C:/Users/Ouwei/Desktop/nav_data/净值0814/市场中性/天算中性B-SXU256.xlsx"
        ),
        begin_date=np.datetime64("2023-12-29"),
        open_html=True,
        image_save_parh=None,
        benchmark="SHSE.000905",
    )
    demo = SingleNavAnalysis(nav_analysis_config)
    demo.analysis()
    demo.export_html()
