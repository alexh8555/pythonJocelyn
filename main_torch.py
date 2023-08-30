import pandas as pd
import os, logging
import matplotlib.pyplot as plt
import utils_model as model
import utils_preprocessing as prep
from time import time
from datetime import datetime
import utils_pytorch as pt

''' Config '''
# TODO: support only one symbol at a time, because we have to train with only one stock at a time
symbols = ['2330']
pastData = 10 # Days of Feature
futureData = 5 # Days of Target
validate_rate = 0.1 # Ratio for validation data
DEBUG_PREPROCESSING = True
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
    preData = prep.preDataFugle()

    # TODO: Get all stocks data
    # data = prep.get_hist_data()
    # TODO: append new data everyday

    if not os.path.isfile(preData.raw):
        data = prep.get_hist_data_fugle(symbols=symbols)
        data.to_csv(preData.raw, index=False)
    else:
        logging.info('file:{0}, symbol:{1} exist!'.format(preData.raw, symbols[0]))

    ''' Training '''
    for symbol in symbols:
        modelName = prep.getModelName(symbol, start_sys_time)

        if (modelName in preData.model) or (SKIP_TRAIN):
            logging.info(symbol + ' model already trained, skip!')
            continue

        # Read data from csv and prepare data
        # Available options
        # ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'change', 'symbol'] + ['ts', 'weekday']
        data = pd.read_csv(preData.raw)
        train_data = prep.preprocessing(data, list(data.columns.values))
        columes = list(train_data.keys())

        train_data = pd.DataFrame.from_dict(train_data)

        # Pop those not needed columes
        train_data.pop('symbol'); train_data.pop('date');
        x_train, x_test, y_train, y_test = pt.make_tensor(train_data)
        print('y{0}, x{1}'.format(y_train.size(),x_train.size()))

        # Generate training/validation data
        train_data = pd.DataFrame.from_dict(train_data)

        # TODO: Some of the values need to be normalize, also check the sequence of data should be reverse or not?

        # Pop those not needed columes
        train_data.pop('symbol'); train_data.pop('date');
        target_data = train_data.pop('close')

        # Prepare data for training/validation
        X_train, Y_train, X_test, Y_test = prep.getTrainData(train_data, target_data, pastData, futureData, validate_rate)

        lookback = 4
        # train-test split for time series
        X_train, Y_train = pt.convert_dataset(X_train, Y_train)
        X_test, Y_test = pt.convert_dataset(X_test, Y_test)
        # X_train, Y_train = pt.create_dataset(train, lookback=lookback)
        # X_test, Y_test = pt.create_dataset(test, lookback=lookback)

        # train model
        # model, history = model.start_training(model, X_train, Y_train, X_validate, Y_validate, modelName)
        model = pt.MyModel().float()
        pt.start_training(model, X_train, Y_train, modelName)

    ''' Prediction '''
    # for symbol in symbols:
    #     # TODO: add model class
    #     modelName = prep.getModelName(symbol, start_sys_time)
    #     logging.info('Start prediction {0}, file:{1}'.format(symbol, modelName))

    #     # Read data from csv and prepare data
    #     data = pd.read_csv(preData.raw)
    #     test_data = prep.preprocessing(data, list(data.columns.values))
    #     test_data = pd.DataFrame.from_dict(test_data)
    #     test_data.pop('symbol'); test_data.pop('date');

    #     # day=0 to predict tomorrow
    #     # day=1 to predict today for verification
    #     test, answer = prep.getTestData(test_data, pastData, day=0)

    #     model.prediction(model.load_pretrain(modelName), test, symbol)
    #     print('Answer : {0}'.format(answer))

    # finalize
    finalize(start_sys_time)