from log import logger
from optimizer import Optimizer
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def main():

    coins = ['BTC', 'BAT', 'BCH', 'ADA', 'LINK', 'COMP', 'ATOM', 'CRV', 'MANA',
             'ETH', 'ETC', 'FIL', 'LTC', 'LPT', 'MKR', 'MATIC', 'DOT', 'SOL',
             'XLM', 'UNI', 'ZEC', 'ZEN']
    r = pd.DataFrame
    t = pd.DataFrame
    for coin in coins:
        logger.info(f'******************** coin {coin} ******************** ')
        print(f'******************** coin {coin} ******************** ')
        ose = Optimizer(_type='bull_bear_ratio', coin=coin)
        ose.find_optimal_input_arguments()
        ose.results['coin'] = coin
        if r.empty:
            r = ose.results
        else:
            r = pd.concat([r, ose.results])
            logger.info(f'optimized results for {coin}')

        if t.empty:
            t = ose.tears
        else:
            t = pd.concat([t, ose.tears])
            logger.info(f'optimized results for {coin}')

    r.to_csv('Results.csv')
    pass


if __name__ == '__main__':
    main()



