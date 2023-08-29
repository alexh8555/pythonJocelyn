import requests, glob, datetime, time
import pandas as pd
import numpy as np

# Download Data
def get_symbols():
    '''
    取得股票代號
    '''
    symbol_link = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data'
    symbols = pd.read_csv(symbol_link)
    return symbols

def gen_calendar():
    '''
    產生日期表
    '''
    this_year = datetime.datetime.now()
    years = range(2010, this_year.year + 1)  # Fugle提供的資料從2010年
    begin = [int(str(y) + '0101') for y in years]
    end = [int(str(y) + '1231') for y in years]
    calendar = pd.DataFrame({'begin': begin,
                            'end': end})
    calendar['begin'] = pd.to_datetime(calendar['begin'], format='%Y%m%d')
    calendar['end'] = pd.to_datetime(calendar['end'], format='%Y%m%d')
    calendar[['begin', 'end']] = calendar[['begin', 'end']].astype('str')
    return calendar

def get_hist_data(symbols=[]):
    '''
    透過富果Fugle API抓取歷史資料
    '''
    if len(symbols) == 0:
        symbols = get_symbols()
    calendar = gen_calendar()
    result = pd.DataFrame()
    for i in range(len(symbols)):
        cur_symbol = symbols[i]
        symbol_result = pd.DataFrame()
        for j in range(len(calendar)):
            cur_begin = calendar.loc[j, 'begin']
            cur_end = calendar.loc[j, 'end']
            # 透過富果Fugle API抓取歷史資料
            data_link = f'https://api.fugle.tw/marketdata/v0.3/candles?symbolId=2884&apiToken=demo&from={cur_begin}&to={cur_end}&fields=open,high,low,close,volume,turnover,change'
            resp = requests.get(url=data_link)
            data = resp.json()
            candle = data['data']
            new_result = pd.DataFrame.from_dict(candle)
            symbol_result = pd.concat([symbol_result, new_result])
        symbol_result['symbol'] = cur_symbol
        result = pd.concat([result, symbol_result])
    return result


# Preproessing
convertDateToTs = lambda x: datetime.datetime.timestamp(datetime.datetime.strptime(x,"%Y-%m-%d"))
getWeekday = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").weekday()

def preprocessing(data, columes):
    out_data, sorted_out_data = {}, {}

    # Convert date to timestamp first
    out_data['ts'] = list(map(lambda x: convertDateToTs(x), data['date'].tolist()))
    out_data['weekday'] = list(map(lambda x: getWeekday(x), data['date'].tolist()))
    out_data['date'] = data['date'].tolist()

    for c in columes:
        if c != 'date':
            # Convert other data
            out_data[c] = data[c].tolist()

    # TODO: Maybe the reorder part can be better coding
    # Get sorted index from timestamp
    sort_idx = [i[0] for i in sorted(enumerate(out_data['ts']), key=lambda x:x[1])]

    # Reorder everything according to index
    for c in list(out_data.keys()):
        tmp_list = []
        for i in sort_idx:
            tmp_list.append(out_data[c][i])
        sorted_out_data[c] = tmp_list

    return sorted_out_data

def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(X, Y, validate_rate):
    X_train = X[int(X.shape[0] * validate_rate):]
    Y_train = Y[int(Y.shape[0] * validate_rate):]
    X_val = X[:int(X.shape[0] * validate_rate)]
    Y_val = Y[:int(Y.shape[0] * validate_rate)]
    return X_train, Y_train, X_val, Y_val

def buildData(data, target, pastData=30, futureData=5, validate_rate=0.1):
    X_train, Y_train = [], []
    for i in range(data.shape[0] - futureData - pastData):
        X_train.append(np.array(data.iloc[i:i + pastData]))
        Y_train.append(np.array(target.iloc[i + pastData:i + pastData + futureData]))

    X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train))
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, validate_rate)

    return X_train, Y_train, X_val, Y_val

class preData:
    def __init__(self):
        self.model = []
        self.date = datetime.datetime.utcfromtimestamp(time.time())
        self.today = str(self.date.month) + str(self.date.day)
        self.model.extend(glob.glob(self.today + 'model//**.h5'))
        self.raw = 'raw//history_raw.csv' # Maybe change to list
        # self.file.extend(glob.glob(self.today + 'npy//**.npy'))
        print('Checking what we got in local...')

    def listModel(self):
        print('{0}'.format(self.model))