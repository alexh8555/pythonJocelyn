import pandas as pd
import os
from utils_preprocessing import get_hist_data, preprocessing
import matplotlib.pyplot as plt
import logging

# Config
DEBUG_PREPROCESSING = True
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s')
plt.set_loglevel('error')

if __name__ == '__main__':
    # TODO: Get all stocks data
    # data = get_hist_data()

    # FIXME: testing with 2330 TSMC only
    file_name = 'history.csv'

    # NOTE: support only one symbol at a time, because we have to train with only one stock at a time
    symbols = ['2330']

    if not os.path.isfile(file_name):
        data = get_hist_data(symbols=symbols)
        data.to_csv(file_name, index=False)
    else:
        logging.info('file:{0}, symbol:{1} exist!'.format(file_name, symbols[0]))

    # Read data from csv and prepare data
    data = pd.read_csv(file_name)
    columes = data.columns.values.tolist()
    # print(columes) # print(list(data.columns))
    # print(data.values.tolist()[0])

    # Available options
    # ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'change', 'symbol'] + ['ts']
    train_data = preprocessing(data, columes)

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


    # TODO: make LSTM model

    # TODO: train model

    # TODO: use the model to predict
