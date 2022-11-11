import numpy as np
import random

def gen_data():
    data1 = np.random.randint(10,100,size=1000).tolist()
    num = np.round(np.random.gamma(1,2,1000)*10)
    data2 = [int(i) for i in num]
    return data1, data2