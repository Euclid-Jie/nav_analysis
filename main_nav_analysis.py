from single_nav_analysis import SingleNavAnalysis
from compare_nav_analysis import CompareNavAnalysis
from single_nav_analysis import NavAnalysisConfig, SingleNavAnalysis
from pathlib import Path
import numpy as np


nav_analysis_config = NavAnalysisConfig(
    bench_data_path=Path(r"C:\Euclid_Jie\barra\src\nav_analysis\index_data.csv"),
    nav_data_path=Path(
        r"C:\Users\Ouwei\Desktop\管理人单页\STC996-永誉天泽经纶二号.xlsx"
    ),
    # begin_date=np.datetime64("2023-12-29"),
    open_html=False,
    # benchmark="SHSE.000852",
)
demo = SingleNavAnalysis(nav_analysis_config)
demo.analysis()
demo.export_html(save=True)
print(demo)
