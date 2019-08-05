import os

import numpy as np
import pandas as pd


class CountryStatsDataset:
    __data_dir__ = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    def __init__(self):
        oecd_bli = pd.read_csv(os.path.join(self.__data_dir__, 'better_life_index.csv'), thousands=',')
        gdp_per_capita = pd.read_csv(os.path.join(self.__data_dir__, 'weo.csv'), thousands=',', na_values="n/a")

        country_stats = self._prepare_country_stats(oecd_bli, gdp_per_capita)
        self.gdp_per_capita = np.c_[country_stats['GDP per capita']]
        self.life_satisfaction = np.c_[country_stats['Life satisfaction']]

    @staticmethod
    def _prepare_country_stats(oecd_bli, gdp_per_capita):
        """
        合并 'weo.csv' 与 'better_life_index.csv' 文件，得到如下数据列：

            | Country | GDP per capita | Life satisfaction |
            |---------+----------------+-------------------|

        """
        oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
        oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')

        gdp_per_capita.rename(columns={'2015': 'GDP per capita'}, inplace=True)
        gdp_per_capita.set_index("Country", inplace=True)

        full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
        remove_indices = [0, 1, 6, 8, 33, 34, 35]
        keep_indices = list(set(range(36)) - set(remove_indices))
        return full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[keep_indices]
