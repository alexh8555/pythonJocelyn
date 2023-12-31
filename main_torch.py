import pandas as pd
import os, logging
import matplotlib.pyplot as plt
import utils_pytorch as utm
# import utils_model as utm
import utils_preprocessing as prep
from time import time
from datetime import datetime
import torch.nn as nn
import torch
import numpy as np

''' Config '''
# TODO: support only one symbol at a time, because we have to train with only one stock at a time
symbols = ['2330']
lookback = 10 # Days of Feature
futureData = 1 # Days of Target
validate_rate = 0.1 # Ratio for validation data
num_epochs = 1 # For training
USE_FUGEL = False
USE_YAHOO = True
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

    else: # elif USE_YAHOO:
        preData = prep.preDataYahoo()
        if not os.path.isfile(preData.raw):
            data = prep.get_hist_data_yahoo(symbols=symbols)
            if data is not None:
                data.to_csv(preData.raw, index=False)
            else:
                print('get data fail...')
        else:
            logging.info('file:{0}, symbol:{1} exist!'.format(preData.raw, symbols[0]))

    for symbol in symbols:
        modelName = preData.getTorchModelName(symbol)

        if (modelName in preData.torchModel) or (SKIP_TRAIN):
            logging.info(symbol + ' model already trained, skip!')
            continue

        # Read data from csv and prepare data
        # Available options
        # ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'change', 'symbol'] + ['ts', 'weekday']
        data = pd.read_csv(preData.raw)
        train_data = prep.preprocessing(data, list(data.columns.values))
        print(train_data.head())
        # Generate training/validation data
        # train_data = pd.DataFrame.from_dict(train_data)

        # TODO: Some of the values need to be normalize, also check the sequence of data should be reverse or not?

        # Pop those not needed columes
        train_data.pop('symbol'); train_data.pop('date');
        target_data = train_data.pop('close')

        # Prepare data for training/validation
        X_train, Y_train, X_test, Y_test = prep.getTrainData(train_data, target_data, lookback, futureData, validate_rate)

        # train-test split for time series        X_train, Y_train = utm.convert_dataset(X_train, Y_train)
        # X_train, Y_train = utm.create_dataset(train, lookback=lookback)
        # X_test, Y_test = utm.create_dataset(test, lookback=lookback)

        # train model
        # model, history = model.start_training(model, X_train, Y_train, X_validate, Y_validate, modelName)
        model = utm.LSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        print(model)
        print(len(list(model.parameters())))
        for i in range(len(list(model.parameters()))):
            print(list(model.parameters())[i].size())

        X_train = torch.Tensor(X_train)
        Y_train = torch.Tensor(Y_train)
        X_test = torch.Tensor(X_test)
        Y_test = torch.Tensor(Y_test)
        print('X_train.shape = ', X_train.shape)
        print('Y_train.shape = ', Y_train.shape)
        print('X_test.shape = ', X_test.shape)
        print('Y_test.shape = ', Y_test.shape)

        ''' Training '''
        hist = np.zeros(num_epochs)

        print(f"Start Training... Total Epochs:{num_epochs}")
        for t in range(num_epochs):
            for Xt, Yt in zip(X_train, Y_train):
                optimizer.zero_grad()
                model.hidden = (torch.zeros(1,1,model.hidden_size),
                                torch.zeros(1,1,model.hidden_size))


                Y_pred = model(Xt)
                loss = criterion(Y_pred, Yt)
                loss.backward()
                optimizer.step()
            print(f"Epoch {t}, Loss: {loss.item()}")
        print(preData.getTorchModelName(symbol))
        print(preData.getTorchModelName(symbol).split('/'))

        if not os.path.exists(preData.getTorchModelName(symbol).split('/')[0]):
            os.makedirs(preData.getTorchModelName(symbol).split('/')[0])
        torch.save(model, preData.getTorchModelName(symbol))

    ''' Prediction '''
    for symbol in symbols:
        # TODO: add model class
        modelName = preData.getTorchModelName(symbol)
        logging.info('Start prediction {0}, file:{1}'.format(symbol, modelName))
        model = torch.load(modelName)

        # Read data from csv and prepare data
        data = pd.read_csv(preData.raw)
        test_data = prep.preprocessing(data, list(data.columns.values))
        # test_data = pd.DataFrame.from_dict(test_data)
        test_data.pop('symbol');
        date_bk = test_data.pop('date').values.tolist();

        # day=0 to predict tomorrow
        # day=1 to predict today for verification
        Y_pred = []

        target_data = test_data.pop('close')
        _, _, X_test, Y_test = prep.getTrainData(test_data, target_data, lookback, futureData, validate_rate)
        X_test = torch.Tensor(X_test)
        # Y_test = torch.Tensor(Y_test)

        for Xt, Yt in zip(X_test, Y_test):
            test = model(Xt).tolist()
            Y_pred.append(test)
            try:
                print(f'Pred: {test}, Answer : {Yt}')
            except:
                print(f"Pred: {test}, No Answer, it's future")

        plt.figure().subplots_adjust(bottom=0.3)
        plt.title(f'{symbol}')
        if True:
            plt.plot(date_bk[-len(Y_pred)::5], Y_pred[::5], label='Predict')
            plt.plot(date_bk[-len(Y_pred)::5], Y_test[::5], label='Answer')
        else:
            plt.plot(range(len(Y_pred)), Y_pred, label='Predict')
            plt.plot(range(len(Y_pred)), Y_test, label='Answer')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('data')
        plt.ylabel('price')
        plt.show()


    # finalize
    finalize(start_sys_time)