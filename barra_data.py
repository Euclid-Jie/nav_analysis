import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
import requests


class BarraData:
    def __init__(self, save_folder: Path):
        self.save_folder = save_folder
        self.save_folder.mkdir(exist_ok=True)
        self.inited = self.check_data()
        self.headers = {
            "access-token": "e5eb1cf42179fbc5dc1d8364a7050964ba23fd58974092352046c4df586b1108ace4029c99b2728bfa135ab1e7b26b21",
        }
        self.url_dict = {
            "cne5": "https://pyapi.huofuniu.com/pyapi/factor/price?mod=cne5_style&sd={}&ed={}",
            "cne6": "https://pyapi.huofuniu.com/pyapi/factor/price?mod=cne6_style&sd={}&ed={}",
            "future": "https://pyapi.huofuniu.com/pyapi/factor/price?mod=future_new&plate=&sd={}&ed={}",
        }

    def check_data(self):
        # Check if data is already downloaded
        if (
            self.save_folder.joinpath("cne5.csv").exists()
            and self.save_folder.joinpath("cne6.csv").exists()
            and self.save_folder.joinpath("future.csv").exists()
        ):
            return True
        else:
            print("Data not found, please run update_data() to download data.")
            return False

    def load_data(
        self,
        type: Literal["cne5", "cne6", "future"],
        strat_date: np.datetime64 = np.datetime64("2018-06-06"),
        end_date: np.datetime64 = np.datetime64("today"),
    ):
        # Load Barra data
        if type == "cne5":
            data = pd.read_csv(self.save_folder.joinpath("cne5.csv")).dropna()
        elif type == "cne6":
            data = pd.read_csv(self.save_folder.joinpath("cne6.csv")).dropna()
        elif type == "future":
            data = pd.read_csv(self.save_folder.joinpath("future.csv")).dropna()
        else:
            raise ValueError("type should be 'cne5' or 'cne6' or 'future'")
        data["日期"] = pd.to_datetime(data["日期"])
        data = data[(data["日期"] >= strat_date) & (data["日期"] <= end_date)]
        return data.reset_index(drop=True)

    def update_data(
        self,
        type: Literal["cne5", "cne6", "future"],
        start_date: np.datetime64 = np.datetime64("2018-06-06"),
        end_date: np.datetime64 = np.datetime64("today"),
    ):
        # Update Barra data
        if not self.inited:
            url = self.url_dict[type].format(
                np.datetime_as_string(start_date, unit="D"),
                np.datetime_as_string(end_date, unit="D"),
            )
            data = requests.get(url, headers=self.headers).json()["data"]
            all_data = pd.DataFrame()
            for key, value in data.items():
                value = pd.DataFrame(value)
                value["日期"] = pd.to_datetime(value["date"])
                value.set_index("日期", inplace=True)
                value.rename(columns={"return": key}, inplace=True)
                value.drop(columns=["date"], inplace=True)
                all_data = pd.concat([all_data, value], axis=1)
            all_data.reset_index(drop=False, inplace=True)
            all_data.to_csv(
                self.save_folder / f"{type}.csv", index=False, encoding="utf-8-sig"
            )
        else:
            exited_data = self.load_data(type)
            least_date = np.datetime64(exited_data["日期"].values[-1])
            if least_date < end_date:
                url = self.url_dict[type].format(
                    np.datetime_as_string(least_date, unit="D"),
                    np.datetime_as_string(end_date, unit="D"),
                )
                data = requests.get(url, headers=self.headers).json()["data"]
                all_data = pd.DataFrame()
                for key, value in data.items():
                    value = pd.DataFrame(value)
                    value["日期"] = pd.to_datetime(value["date"])
                    value.set_index("日期", inplace=True)
                    value.rename(columns={"return": key}, inplace=True)
                    value.drop(columns=["date"], inplace=True)
                    all_data = pd.concat([all_data, value], axis=1)
                all_data.reset_index(drop=False, inplace=True)
                updated_data = pd.concat(
                    [exited_data, all_data], axis=0
                ).drop_duplicates()
                updated_data.to_csv(
                    self.save_folder / f"{type}.csv", index=False, encoding="utf-8-sig"
                )


if __name__ == "__main__":
    demo = BarraData(Path(r"C:\Euclid_Jie\barra\submodule\nav_analysis\barra_data"))
    demo.update_data("cne5")
    demo.update_data("cne6")
    demo.update_data("future")
