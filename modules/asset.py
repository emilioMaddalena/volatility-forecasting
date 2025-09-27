from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


class Asset:
    def __init__(
        self,
        asset_name: str,
        tick_name: str,
        time_period: Tuple[str, str],
        value_type: str = "Close",
    ):
        self.asset_name = asset_name
        self.tick_name = tick_name
        self.start_date, self.end_date = time_period
        self.value_type = value_type

        self._download_prices()
        self._compute_log_returns()
        self._estimate_pointwise_volatility()

    def _download_prices(self):
        self.price = {}
        df = yf.download(self.tick_name, start=self.start_date, end=self.end_date)
        self.price = df[self.value_type]

    def _compute_log_returns(self, detrend: bool = False, mean_window: int = 30):
        self.returns = self._compute_log_difference(self.price)
        if detrend:
            self.returns = self._detrend(self.returns, mean_window=mean_window)

    def _estimate_pointwise_volatility(self):
        """Estimate via squared returns.
        
        N.B. This is a variance estimator, not standard deviation.
        """
        self.pointwise_volatility = self.returns**2

    def compute_realized_volatility(self, window: int = 30, store: bool = False) -> pd.Series:
        """Compute realized volatility over a rolling window.
        
        N.B. This is a variance estimator, not standard deviation.
        """
        realized_vol = self.returns.rolling(window=window).var()
        if store:
            self.__setattr__(f"realized_volatility_w{window}", realized_vol)
        return realized_vol

    @staticmethod
    def _compute_log_difference(series: pd.Series) -> pd.Series:
        return np.log(series / series.shift(1)).dropna()

    @staticmethod
    def _detrend(series: pd.Series, mean_window: int = 30) -> pd.Series:
        return series - series.rolling(window=mean_window).mean()
