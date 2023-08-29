import pandas as pd
import os, logging
import matplotlib.pyplot as plt
import utiils_model as model
import utils_preprocessing as prep
from time import time
from datetime import datetime

''' Config '''
# TODO: support only one symbol at a time, because we have to train with only one stock at a time
symbols = ['2330']
pastData = 30 # Days of Feature
futureData = 5 # Days of Target
validate_rate = 0.1 # Ratio for validation data
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
    preData = prep.preData()

    # TODO: Get all stocks data
    # data = prep.get_hist_data()
    # TODO: append new data everyday

    if not os.path.isfile(preData.raw):
        data = prep.get_hist_data(symbols=symbols)
        data.to_csv(preData.raw, index=False)
    else:
        logging.info('file:{0}, symbol:{1} exist!'.format(preData.raw, symbols[0]))

    for symbol in symbols:
        date = datetime.utcfromtimestamp(start_sys_time)
        today = str(date.month) + str(date.day)
        modelName = today + 'model//' + today + symbol + '.h5'
        if (modelName in preData.model) and (SKIP_TRAIN):
            logging.info(symbol + ' model already trained, skip!')
            continue

        # Read data from csv and prepare data
        data = pd.read_csv(preData.raw)
        columes = list(data.columns.values)

        # Available options
        # ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'change', 'symbol'] + ['ts', 'weekday']
        train_data = prep.preprocessing(data, columes)
        columes = list(train_data.keys())

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
        train_data = pd.DataFrame.from_dict(train_data)

        # TODO: Some of the values need to be normalize, also check the sequence of data should be reverse or not?

        # Pop those not needed columes
        train_data.pop('symbol'); train_data.pop('date');
        target_data = train_data.pop('close')

        # Prepare data for training/validation
        X_train, Y_train, X_validate, Y_validate = prep.buildData(train_data, target_data, pastData, futureData, validate_rate)

        # train model
        model, history = model.start_training(model, X_train, Y_train, X_validate, Y_validate, modelName)

    # TODO: prediction
    for symbol in symbols:
        pass

    # finalize
    finalize(start_sys_time)