import matplotlib.pyplot as plt
from pandas.io.pickle import read_pickle
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
import math
#import top_gunners

def main():
    plays = pd.read_pickle("Data/combplaydata.pkl")
    plays = plays[['absoluteYardlineNumber', 'yardlineNumber', 'yardlineSide']]
    print(plays.head(20))


main()