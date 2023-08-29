import pandas as pd
import os
import matplotlib.pyplot as plt
import logging
import utiils_model as model
import utils_preprocessing as prep

# Config
# TODO: support only one symbol at a time, because we have to train with only one stock at a time
symbols = ['2330']
pastData = 30
futureData = 5
rate = 0.1
DEBUG_PREPROCESSING = False

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s')
plt.set_loglevel('error')

if __name__ == '__main__':
    # TODO: Get all stocks data
    # data = prep.get_hist_data()
    # TODO: append new data everyday
    file_name = 'history_raw.csv'

    if not os.path.isfile(file_name):
        data = prep.get_hist_data(symbols=symbols)
        data.to_csv(file_name, index=False)
    else:
        logging.info('file:{0}, symbol:{1} exist!'.format(file_name, symbols[0]))

    # Read data from csv and prepare data
    data = pd.read_csv(file_name)
    columes = list(data.columns.values)

    # Available options
    # ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'change', 'symbol'] + ['ts']
    train_data = prep.preprocessing(data, columes)
    # print(test_list[0:10])
    columes = list(train_data.keys())

    if DEBUG_PREPROCESSING:
        for j in columes:
            logging.info(j)
            logging.info(train_data[j][0:10])

        for symbol in symbols:
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
    # TODO: Some of the values need to be normolize
    # Pop those not needed colume
    train_data.pop('symbol')
    train_data.pop('date')
    target_data = train_data.pop('close')
    X_train, Y_train, X_validate, Y_validate = prep.buildData(train_data, target_data, pastData, futureData, rate)

    # train model
    model = model.start_training(model, X_train, Y_train, X_validate, Y_validate, 'test_model')

    # TODO: prediction