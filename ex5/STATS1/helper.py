import pandas as pd
import numpy as np

# Getting data
def get_data():
    df = pd.read_csv('stats_ex.csv')
    df=df.dropna(axis=1)

    data=[]
    for i in range(10000):
        data.append(list(df.iloc[i]))
    return data