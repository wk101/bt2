from log import logger
import pandas as pd
from pandas import DataFrame
from strategy import EMASentiment


class Optimizer:

    def __init__(self, _type: str, coin: str):
        self.df: DataFrame = pd.DataFrame
        self.results: DataFrame = pd.DataFrame
        self.tears: DataFrame = pd.DataFrame
        self._type: str = _type
        self._coin: str = coin
        self._input_step_size: int = 1
        self._input_start_constraint: int = 2
        self._input_end_constraint: int = 40

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
        self.objective_strategy = EMASentiment(strategy_type=self._type, crypto_currency=self._coin)

    def _run_objective_strategy_with_constraints(self):
        for _input in self._range:
            strategy = self.objective_strategy
            strategy.ema_days = _input
            strategy.execute()
            if strategy.performance.empty:
                logger.error('error performance data frame empty')
                raise Exception('performance dataframe empty')
            if type(self.results) == type:
                self.results = strategy.performance.to_frame().T
                self.tears = strategy.tears.to_frame().T
            else:
                self.results = pd.concat([self.results, strategy.performance.to_frame().T], ignore_index=True)
                self.tears = pd.concat([self.tears, strategy.tears.to_frame().T], ignore_index=True)
            logger.info(f'results for {self._coin} ')






