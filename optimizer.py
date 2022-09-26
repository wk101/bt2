from log import logger
import pandas as pd
from pandas import DataFrame
from strategy import EMASentiment


class Optimizer:

    def __init__(self, _type: str, coin: str, _sentiment_file: str):
        self.df: DataFrame = pd.DataFrame
        self.results: DataFrame = pd.DataFrame
        self.buy_hold_results: DataFrame = pd.DataFrame

        self.sentiment_file: str = _sentiment_file
        self.type: str = _type
        self._coin: str = coin
        self._input_step_size: int = 1
        self._input_start_constraint: int = 2
        self._input_end_constraint: int = 100

    def find_optimal_input_arguments(self):
        """ Find the optimal input arguments
        The input here is the number of days used in the exponential moving average.
        The number of days must be greater than two and less than thirty.
        The output we are trying to maximize for the objective strategy is the Sharpe ratio.
        There are at least two methods for calculating the ratio, we will optimize over both.
        :return:
        """
        self._initialize_input_range_constraints()
        self._initialize_objective_strategy()
        self._run_objective_strategy_with_constraints()

    def _initialize_input_range_constraints(self):
        self._range = range(self._input_start_constraint, self._input_end_constraint, self._input_step_size)

    def _initialize_objective_strategy(self):
        self.objective_strategy = EMASentiment(strategy_type=self.type, crypto_currency=self._coin,
                                               _sentiment_file=self.sentiment_file)

    def _run_objective_strategy_with_constraints(self):
        for _input in self._range:
            strategy = self.objective_strategy
            print(_input)
            strategy.ema_days = _input
            strategy.execute()
            if strategy.performance.empty:
                logger.error('error performance data frame empty')
                raise Exception('performance dataframe empty')
            if type(self.results) == type:
                logger.debug('first pass performance')
                self.results = strategy.performance.to_frame().T
            else:
                logger.debug('second pass performance')
                self.results = pd.concat([self.results, strategy.performance.to_frame().T], ignore_index=True)

            if strategy.buy_hold_performance.empty:
                logger.error('error performance data frame empty')
                raise Exception('performance dataframe empty')
            if type(self.buy_hold_results) == type:
                self.buy_hold_results = strategy.buy_hold_performance.to_frame().T
            else:
                self.buy_hold_results = pd.concat([self.buy_hold_results, strategy.buy_hold_performance.to_frame().T], ignore_index=True)

            logger.info(f'results for {self._coin} ')






