import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def main():
    tracking_df_2018 = pd.read_pickle("Data/2018trackingdata.pkl")
    tracking_df_2019 = pd.read_pickle("Data/2019trackingdata.pkl")
    tracking_df_2020 = pd.read_pickle("Data/2020trackingdata.pkl")
    print(len(tracking_df_2018))
    comb = pd.concat([tracking_df_2018, tracking_df_2019, tracking_df_2020])
    print(len(comb))
    comb.to_pickle("Data/combinedtrackingdata.pkl")


main()