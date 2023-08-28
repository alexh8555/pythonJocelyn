import requests
import pandas as pd
import datetime

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
    out_data = {}

    for c in columes:
        target = data[c].tolist()

        if c == 'date':
            # Convert date to timestamp for training
            out_data[c] = target
            out_data['ts'] = list(map(lambda x: convertDateToTs(x), target))
            out_data['weekday'] = list(map(lambda x: getWeekday(x), target))
        else:
            out_data[c] = (target)

    return out_data