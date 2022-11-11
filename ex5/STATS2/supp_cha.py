import pandas as pd
import numpy as np

def read_data():
    df = pd.read_csv('stats_challenge.csv')
    df['sofifa_id'] = df['sofifa_id'].astype('int')
    data = [list(df.iloc[0]),list(df.iloc[1]),list(df.iloc[2]),list(df.iloc[3]),list(df.iloc[4]),list(df.iloc[5]),
        list(df.iloc[6]),list(df.iloc[7]),list(df.iloc[8]),list(df.iloc[9])]
    # a = np.int64
    data = [[int(i) if isinstance(i,np.int64) else i for i in datapoint] for datapoint in data]
    return data

def read_data_full():
    df = pd.read_csv('stats_full.csv')
    data = []
    for i in range(10000):
        data.append(list(df.iloc[i]))
    data = [[int(i) if isinstance(i,np.int64) else i for i in datapoint] for datapoint in data]
    return data