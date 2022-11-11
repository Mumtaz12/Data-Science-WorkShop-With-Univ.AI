run_times_list=[[12.46, 16.15, 13.2, 17.78, 12.09, 11.85, 7.64, 15.95, 11.7, 12.28],
               [11.9, 11.24, 12.2, 12.44, 15.32, 13.76, 10.39, 12.8, 13.63, 16.73],
               [9.48, 12.7, 17.86, 11.94, 9.15, 10.68, 15.85, 15.57, 15.02, 12.3],
               [10.66, 13.06, 12.03, 14.4, 13.21, 10.76, 11.03, 10.59, 15.26, 13.55],
               [10.48, 14.28, 10.1, 11.97, 11.57, 7.62, 11.6, 8.86, 11.87, 13.82],
               [11.11, 12.58, 10.37, 10.83, 15.19, 14.82, 13.03, 15.1, 13.76, 12.82],
               [13.79, 11.01, 10.94, 11.8, 11.52, 11.07, 14.59, 12.81, 15.91, 11.61]]

import numpy as np
import pandas as pd

race_dict={}

for i in range(1,11):
    race_dict[f'lap_{i}']=np.round_((np.random.normal(loc=11, scale=2, size=7) + np.random.uniform(0.5,2,size=7)), 2)
    
runs_df=pd.DataFrame(race_dict, index=['Hargun','Sheetal','Manoj','Pavlos','Snigdha','Chaitanya','Kisalaya'])

runs_df.columns=['practice_1', 'run_1','practice_2', 'run_2','practice_3', 'run_3','practice_4', 'run_4','practice_5', 'run_5']

