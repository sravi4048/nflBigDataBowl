## Ignore this for now
import matplotlib.pyplot as plt
from pandas.io.pickle import read_pickle
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
import math
from sklearn import metrics

def main():
    for f in ["gunner_tracking.pkl", "2018trackingdata.pkl", "2019trackingdata.pkl", "2020trackingdata.pkl", 
    "processed_gunner_returner_df.pkl", "punt_returner_tracking.pkl"]:
        df = pd.read_pickle("Data/"+f)
        for col in df.columns.tolist():
            if col[0] == 'x':
                df['adj_'+col] = df.apply(lambda x: x[col] if x['playDirection'] == 'right' else 120 - x[col], axis=1)
            elif col[0] == 'o' or col[:3] == 'dir':
                df['adj_'+col] = df.apply(lambda x: 360 - x[col], axis=1)
        df['adj_playDirection'] = 'right'
        break
    
    print(df.loc[df['playDirection'] == 'left'][['playDirection', 'adj_x', 'x', 'adj_playDirection']].head(10))
main()