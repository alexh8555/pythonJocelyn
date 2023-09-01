import pandas as pd
import os, logging
import matplotlib.pyplot as plt
import utils_model as utm
import utils_preprocessing as prep
from time import time
from datetime import datetime

''' Config '''
# TODO: support only one symbol at a time, because we have to train with only one stock at a time
symbols = ['2330']
lookback = 10 # Days to lookback
futureData = 1 # Days of Target
validate_rate = 0.1 # Ratio for validation data
USE_FUGEL = False
USE_YAHOO = True
DEBUG_PREPROCESSING = False
SKIP_TRAIN = False

''' Function Definition '''
def initialize():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s')
    plt.set_loglevel('error')

    logging.info('Initialize...')
    sysTime = time()
    logging.info('start time : ' + datetime.utcfromtimestamp(sysTime).strftime('%Y-%m-%d %H:%M:%S'))
    return sysTime

def finalize(start_sys_time):
    logging.info('Finalize...')
    logging.info('start time : ' + datetime.utcfromtimestamp(start_sys_time).strftime('%Y-%m-%d %H:%M:%S'))
    logging.info('finish time : ' + datetime.utcfromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'))
    return

''' Main Process'''
if __name__ == '__main__':
    start_sys_time = initialize()

    '''Download data'''
    # TODO: Get all stocks data
    # data = prep.get_hist_data_fugle()
    # TODO: append new data everyday
    if USE_FUGEL:
        preData = prep.preDataFugle()

        if not os.path.isfile(preData.raw):
            data = prep.get_hist_data_fugle(symbols=symbols)
            data.to_csv(preData.raw, index=False)
        else:
            logging.info('file:{0}, symbol:{1} exist!'.format(preData.raw, symbols[0]))

    else : #elif USE_YAHOO:
        preData = prep.preDataYahoo()
        if not os.path.isfile(preData.raw):
            data = prep.get_hist_data_yahoo(symbols=symbols)
            if data is not None:
                data.to_csv(preData.raw, index=False)
            else:
                print('get data fail...')
        else:
            logging.info('file:{0}, symbol:{1} exist!'.format(preData.raw, symbols[0]))

    ''' Training '''
    for symbol in symbols:
        modelName = preData.getModelName(symbol)

        if (modelName in preData.model) or (SKIP_TRAIN):
            logging.info(symbol + ' model already trained, skip!')
            continue

        # Read data from csv and prepare data
        # Available options
        # ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'change', 'symbol'] + ['ts', 'weekday']
        data = pd.read_csv(preData.raw)
        train_data = prep.preprocessing(data, list(data.columns.values))
        columes = list(train_data.columns)

        if DEBUG_PREPROCESSING:
            for j in columes:
                logging.info(j)
                logging.info(train_data[j][0:10])

            plt.title('{0}'.format(symbol))
            for col in columes:
                if (col != 'date') and (col != 'symbol'):
                    plt.plot(train_data['date'][::60], train_data[col][::60], label=col)
            plt.legend(loc='upper right')
            plt.xticks(rotation=45, ha='right')
            # plt.xlabel('date')
            # plt.ylabel('value')
            plt.show()

        # Generate training/validation data
        # train_data = pd.DataFrame.from_dict(train_data)

        # TODO: Some of the values need to be normalize, also check the sequence of data should be reverse or not?

        # Pop those not needed columes
        train_data.pop('symbol'); train_data.pop('date');
        target_data = train_data.pop('close')

        # Prepare data for training/validation
        X_train, Y_train, X_validate, Y_validate = prep.getTrainData(train_data, target_data, lookback, futureData, validate_rate)

        # train model
        model, history = utm.start_training(X_train, Y_train, X_validate, Y_validate, modelName)

    ''' Prediction '''
    for symbol in symbols:
        # TODO: add model class
        modelName = preData.getModelName(symbol)
        logging.info('Start prediction {0}, file:{1}'.format(symbol, modelName))

        # Read data from csv and prepare data
        data = pd.read_csv(preData.raw)
        test_data = prep.preprocessing(data, list(data.columns.values))
        # test_data = pd.DataFrame.from_dict(test_data)
        test_data.pop('symbol'); test_data.pop('date');

        '''
        day=0 to predict tomorrow
        day=1 to predict today for verification
        '''
        test, answer = prep.getTestData(test_data, lookback, day=0)

        utm.prediction(utm.load_pretrain(modelName), test, symbol)
        print('Answer : {0}'.format(answer))

    # finalize
    finalize(start_sys_time)