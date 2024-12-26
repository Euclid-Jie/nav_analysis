from utils import *
from single_nav_analysis import SingleNavAnalysis
import imgkit
import re


class BatchNavAnalysis(SingleNavAnalysis):
    def __init__(
        self,
        nav_analysis_config: NavAnalysisConfig,
        config,
        options,
        rewrite: bool,
        img_save_fold: Path,
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

        self.config = config
        self.options = options
        self.rewrite = rewrite
        self.img_save_fold = img_save_fold
        self.img_save_fold.mkdir(parents=True, exist_ok=True)

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
            single_nav_analysis.name = keep_chinese_chars(single_nav_analysis.name)
            img_save_full_path = self.img_save_fold.joinpath(
                f"{single_nav_analysis.name}.jpg"
            )
            if self.rewrite or (not img_save_full_path.exists()):
                single_nav_analysis.analysis()
                single_nav_analysis.export_html(save=False)
                imgkit.from_string(
                    single_nav_analysis.html,
                    config=self.config,
                    output_path=img_save_full_path,
                    options=self.options,
                )


if __name__ == "__main__":
    nav_analysis_config = NavAnalysisConfig(
        bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
        nav_data_path=Path(
            r"C:\Users\Ouwei\Desktop\nav_data\净值库1101\按策略分\300增强"
        ),
        begin_date=np.datetime64("2023-12-29"),
        benchmark="SHSE.000300",
    )
    config = imgkit.config(
        wkhtmltoimage=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe"
    )
    options = {
        "javascript-delay": 1000,
        "crop-w": 1400,
        "crop-h": 580,
        "crop-x": 80,
        "crop-y": 90,
    }
    demo = BatchNavAnalysis(
        nav_analysis_config,
        config=config,
        options=options,
        rewrite=False,
        img_save_fold=Path(
            r"C:\Users\Ouwei\Desktop\nav_data\净值库1101\image\300增强"
        ),
    )
    demo.anlysis()
