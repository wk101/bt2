from log import logger
from optimizer import Optimizer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def main():

    coins1 = ['BTC', 'BAT', 'BCH', 'ADA', 'LINK', 'COMP', 'ATOM', 'CRV', 'MANA',
             'ETH', 'ETC', 'FIL', 'LTC', 'LPT', 'MKR', 'MATIC', 'DOT', 'SOL',
             'XLM', 'UNI', 'ZEC', 'ZEN']
    coins = ['AAVE', 'ALGO', 'AR', 'AVAX', 'AXS', 'BAT', 'BNB', 'CHZ', 'ATOM', 'DOGE', 'ENJ',
             'FTM', 'FTT', 'SNX', 'NEAR', 'SHIB', 'XLM', 'GRT', 'SAND', 'THETA', 'UNI']

    r = pd.DataFrame
    rbh = pd.DataFrame
    for coin in coins:
        logger.info(f'******************** coin {coin} ******************** ')
        print(f'******************** coin {coin} ******************** ')
        ose = Optimizer(_type='sentiment_ratio', coin=coin, _sentiment_file='gct5')  # 'bull_bear_ratio', 'bull_bear_ratio_weighted', 'sentiment_ratio'
        ose.find_optimal_input_arguments()
        ose.results['coin'] = coin
        if r.empty:
            r = ose.results
            rbh = ose.buy_hold_results
        else:
            r = pd.concat([r, ose.results])
            rbh = pd.concat([rbh, ose.buy_hold_results])

            logger.info(f'optimized results for {coin}')

    r.to_csv(f'{ose.sentiment_file}_{ose.type}_100_results.csv')
    rbh.to_csv(f'{ose.sentiment_file}_{ose.type}_100_results_buy_hold.csv')
    pass


if __name__ == '__main__':
    main()



