import numpy as np
import pandas as pd
from utils import calc_nav_rtn
from regression import rolling_regression
from typing import List
import plotly.express as px
import plotly.graph_objects as go


class AttractionAnalysis:
    def __init__(
        self,
        nav: np.ndarray,
        date: np.ndarray[np.datetime64],
        Xs: List[np.ndarray],
        Xs_name: List[str],
        windows: int,
        title: str = "",
    ):
        assert len(date) == len(nav), "日期长度不匹配"
        assert len(Xs) > 0, "至少需要一个自变量"
        assert len(Xs) == len(Xs_name), "自变量名称长度不匹配"
        assert np.all([len(X) == len(nav) for X in Xs]), "自变量长度不匹配"
        self.nav = nav
        self.date = date
        self.Xs = Xs
        self.Xs_name = Xs_name
        self.windows = windows
        self.title = title
        self.date_for_display = date[windows - 1 :]
        self.rtn = calc_nav_rtn(nav, types="log")

    def analyze(self):
        self.betas, self.errors, self.t, self.F, self.R2 = rolling_regression(
            Y=self.rtn,
            Xs=self.Xs,
            window=self.windows,
        )
        self.betas_df = pd.DataFrame(
            self.betas,
            columns=self.Xs_name,
            index=self.date_for_display,
        )

    def plot(self, show: bool = False, R2: bool = False):
        # 将数据转换为长格式
        df_long = self.betas_df.reset_index(drop=False).melt(
            id_vars="index", var_name="name", value_name="Value"
        )
        # 绘制面积图
        fig = px.bar(
            df_long, x="index", y="Value", color="name", title=f"{self.title} Beta"
        )
        # 设置x轴标签
        fig.update_xaxes(title="日期")
        fig.update_layout(
            yaxis1=dict(
                title="敏感度",
                side="right",
            )
        )
        # 增加一条折线, 使用另一个 y 轴, 设置区间为 0-1
        fig.update_layout(
            yaxis2=dict(
                title="R2" if R2 else "nav",
                titlefont=dict(color="black"),
                tickfont=dict(color="black"),
                overlaying="y",
                side="left",
            )
        )
        # 虚线
        fig.add_trace(
            go.Scatter(
                x=self.date_for_display,
                y=self.R2 if R2 else self.nav[self.windows - 1 :],
                mode="lines",
                name="R2" if R2 else "nav",
                line=dict(color="black"),
                yaxis="y2",
            )
        )
        # 显示图表
        if show:
            fig.show()
        return fig
