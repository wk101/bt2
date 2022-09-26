from log import logger
import pandas as pd
from pandas import DataFrame
import numpy as np
import yfinance as yf
import pyfolio as pf
import datetime as dt


MIN_EMA_DAYS: int = 2
SLIPPAGE_IN_BASIS_POINTS: int = 5


class EMASentiment:

    def __init__(self, strategy_type: str, crypto_currency: str, _sentiment_file: str,
                 _show: bool = False, _get_price_from_csv: bool = True):
        logger.info(f'{strategy_type} initiating for {crypto_currency}!')
        self.performance = pd.DataFrame
        self.buy_hold_performance = pd.DataFrame
        self.sentiment = pd.DataFrame
        self.price = pd.DataFrame
        self.df = pd.DataFrame

        self._coin: str = crypto_currency
        self._type: str = strategy_type  # 'bull_bear_ratio'
        self._ema_days: int = MIN_EMA_DAYS
        self._slippage_basis_points = SLIPPAGE_IN_BASIS_POINTS  # A basis point is one hundred of a basis point
        self.sentiment_file = _sentiment_file

        self.show: bool = _show
        self.get_price_from_csv: bool = _get_price_from_csv

    @property
    def type(self):
        """The type property."""
        return self._type

    @type.setter
    def type(self, value: str):
        if value not in ['bull_bear_ratio', 'bull_bear_ratio_weighted', 'sentiment_ratio']:
            msg: str = f'strategy type {value} is not type bull_bear_ratio or sentiment_ratio!!!'
            logger.error(msg)
            raise Exception(msg)
        self._type = value

    @property
    def ema_days(self):
        """The ema_days property."""
        return self._ema_days

    @ema_days.setter
    def ema_days(self, value: int):
        if type(value) != int:
            msg: str = f'ema days is not an int {type(value)}'
            logger.error(msg)
            raise Exception(msg)
        elif value < MIN_EMA_DAYS:
            msg: str = f'ema days for strategy set to below two, {value}'
            logger.error(msg)
            raise Exception(msg)
        self._ema_days = value

    def execute(self):
        self._setup()
        self._calculate_simple_returns()
        self._calculate_ratio()
        self._calculate_ema_on_ratio()
        self._calculate_buy_condition()
        self._calculate_sell_condition()
        self._calculate_positions()
        self._calculate_returns()
        self._calculate_cum_returns()
        if self.show:
            self._show_tear_sheet()
        else:
            self._store_performance_in_dataframe()
            self._store_buy_and_hold_performance_in_dataframe()

    def _setup(self):
        logger.info('setup starting!!')
        if self.get_price_from_csv:
            self._get_price_from_csv()
        else:
            self._get_price_from_yahoo()
        self._get_sentiment_from_csv()
        self._set_sentiment_date_to_dataframe_index()
        self._join_price_and_sentiment()

    def _get_sentiment_from_csv(self):
        """ CSV Sentiment provided
        :return:
        """
        df = pd.read_csv(f'{self.sentiment_file}.csv')
        _filter = np.where((df['symbol'] == self._coin))
        df = df.loc[_filter]
        df = df.dropna()
        self.sentiment = df
        logger.debug(f'{len(df)} lines processed, sentiment from csv!')

    def _set_sentiment_date_to_dataframe_index(self):
        """ set date to dataframe index
            the pyfolio library needs this to create the dear down sheet
            and I would like to join price and sentiment by the data in the index field
        """
        df = self.sentiment
        df['Date'] = pd.to_datetime(df['datestr'])
        df.set_index('Date', inplace=True)
        self.sentiment = df

    def _get_price_from_yahoo(self):
        """
        Closing Price taken from Yahoo
        :return:
        """
        _start = dt.date(2020, 1, 1)
        _end = dt.date(2022, 8, 20)
        ticker = f'{self._coin}-USD'
        self.price = yf.download(ticker, start=_start, end=_end, progress=False)

    def _get_price_from_csv(self):
        pass
        self._upload_csv()
        self._unix_date_to_pandas()

    def _upload_csv(self):
        df = pd.read_csv(f'daily_prices_2.csv')
        _filter = np.where((df['symbol'] == self._coin))
        df = df.loc[_filter]
        df = df.dropna()
        self.price = df

    def _unix_date_to_pandas(self):
        """ unix date to pandas
        This also sets the date to the index
        : return:
        """
        df = self.price
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        df.set_index('Date', inplace=True)
        self.price = df

    def _join_price_and_sentiment(self):
        """join price and sentiment
        Price and sentiment left joined on date index
        self.df isc used for general working
        :return:
       """
        df = self.price.merge(self.sentiment,  left_index=True, right_index=True, how='left')
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(self.df)} lines processed, price and sentiment joined!')

    def _calculate_simple_returns(self):
        """ calculating the simple return
            :return:
        """
        df = self.df
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(df)} lines processed, returns calculated!')

    def _calculate_ratio(self):
        """ calculating the ratio
        Two methodologies, one is bull bear the other is net sentiment
        :return:
        """
        df = self.df
        if self.type == 'bull_bear_ratio':
            df['ratio'] = df['positive_sentiment']/(df['positive_sentiment'] + df['negative_sentiment'])
        elif self.type == 'bull_bear_ratio_weighted':
            df['ratio'] = df['positive_sentiment_weighted'] / (df['positive_sentiment_weighted'] +
                                                               df['negative_sentiment_weighted'])
        elif self.type == 'sentiment_ratio':
            df['ratio'] = df['net_sentiment_weighted'] / df['total_tweets_weight']
        logger.debug(f'{len(df)} lines processed, type {self.type} ratio calculated!')

    def _calculate_ema_on_ratio(self):
        """ calculating the exponential moving average
        The EMA gives a higher weight to recent values, while the SMA assigns equal weight to all values.
        :return:
        """
        df = self.df
        df['ema'] = df['ratio'].ewm(span=self.ema_days, adjust=False).mean()
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(df)} lines processed, exponential moving average calculated!')

    def _calculate_buy_condition(self):
        """ calculating buy condition
                This takes the bull bear sentiment ratio. and if the ema is higher go net long 100%
        :return:
        """
        df = self.df
        df['signal'] = np.where((df['ema'] > df['ratio']) & (df['ema'].shift(1) <= df['ratio'].shift(1)), 1, 0)
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(df)} lines processed, buy condition calculated!')

    def _calculate_sell_condition(self):
        """ calculating sell condition
        self.df = df
                This takes the bull bear sentiment ratio. and if the ema is lower go net short 100%
        :return:
        """
        df = self.df
        df['signal'] = np.where((df['ema'] < df['ratio']) & (df['ema'].shift(1) >= df['ratio'].shift(1)), -1, df['signal'])  # #
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(df)} lines processed, sell condition calculated!')

    def _calculate_positions(self):
        """ calculating positions
        return:
        """
        df = self.df
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        # shifting by 1, to account of close price return calculations
        df['position'] = df['position'].shift(1)
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(df)} lines processed, sell condition calculated!')

    def _calculate_returns(self):
        """ calculating returns
        return:
        """
        df = self.df
        df['strategy_returns'] = df['Returns'] * (df['position'])
        df = df.dropna()
        self.df = df
        logger.debug(f'{len(df)} lines processed, returns calculated!')

    def _calculate_cum_returns(self):
        """ calculates cumulative returns
        :return:
        """
        df = self.df
        df['strategy_cum_returns'] = df['strategy_returns'].cumsum()

        df['buy_and_hold'] = df['Returns'].cumsum()

        df = df.dropna()
        df.to_csv(f'data/results_{self._coin}_{self._ema_days}.csv')
        self.df = df
        # logger.debug(f'{len(df)} lines processed { df.strategy_cum_returns.iloc[-1] } returns calculated!')

    def _store_performance_in_dataframe(self):
        """ store time series performance stats in data frame
            # http://pyfolio.ml4trading.io/api-reference.html#module-pyfolio.timeseries
             return:
        """
        df = self.df
        df = pf.timeseries.perf_stats(df['strategy_cum_returns'].diff())
        df['ema days in optimization'] = self.ema_days
        df = df.dropna()
        self.performance = df
        logger.debug(f'{len(df)} lines processed, store performance in dataframe!')

    def _store_buy_and_hold_performance_in_dataframe(self):
        """ store time series performance stats in data frame
            # http://pyfolio.ml4trading.io/api-reference.html#module-pyfolio.timeseries
             return:
        """
        df = self.df
        df = pf.timeseries.perf_stats(df['buy_and_hold'].diff())
        df['ema days in optimization'] = self.ema_days
        df = df.dropna()
        self.buy_hold_performance = df
        logger.debug(f'{len(df)} lines processed, store performance in dataframe!')

    def _show_tear_sheet(self):
        """ show tears stats
            # http://pyfolio.ml4trading.io/api-reference.html#module-pyfolio.tears
            return:
        """
        df = self.df
        _slippage: str = self._slippage_basis_points
        pf.tears.create_simple_tear_sheet(df['strategy_cum_returns'].diff(), slippage=_slippage)
        logger.debug(f'{len(df)} lines processed, store performance in dataframe!')





