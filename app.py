"""
This module contains the main logic
"""
import math
import os
import time
from typing import Dict, List, Tuple

import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, \
    roll_time_series, \
    make_forecasting_frame

import config as cnf
import forecasting as fc
import preprocessing as pp
import utils as utils

if __name__ == "__main__":
  start = time.time()
  # set up formatting for plotting
  sns.set(font_scale=1.1, rc={'axes.facecolor': 'lightgrey'})

  # Load M4 data
  print('#### reading CSV files ####')
  if os.path.isfile('df_train_sample.csv') and\
          os.path.isfile('df_test_sample.csv'):
    df_train = pd.read_csv('df_train_sample.csv')
    df_test = pd.read_csv('df_test_sample.csv')
    print('### local sample files read ###')
    print('#### {:3.2f}s elapsed ####'.format(time.time() - start))
  else:
    train = pd.read_csv(cnf.DAILY_TRAIN)
    test = pd.read_csv(cnf.DAILY_TEST)
    elapsed = time.time()
    print('#### finished reading CSV files####')
    print('#### {:3.2f}s elapsed ####'.format(time.time() - start))
    # sample training data
    print('#### beginning sampling of time series ####')
    df_train = train.sample(frac=cnf.SAMPLING)
    # restrict test data to sampled training data
    df_test = test.iloc[df_train.index]

  # write sample to local file for faster loading
  if not os.path.isfile('df_train_sample.csv') and\
          not os.path.isfile('df_test_sample.csv'):
    df_train.to_csv('df_train_sample.csv', index=False)
    df_test.to_csv('df_test_sample.csv', index=False)

  print('#### sampling completed ####')
  print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

  # format data for tsfresh
  df_ts = pp.shorten_time_series(df_train)

  # extract features
  #extracted_features = extract_features(df_ts, column_id="V1", column_sort='timestamp')
  # impute(extracted_features)
  # TODO -set up y
  #features_filtered = select_features(extracted_features,y);

  #features = features_filtered.to_numpy()

  print('#### creating train and test data sets ####')
  X_train, y_train,\
      X_test, y_test,\
      standarizers,\
      ts_order = pp.create_train_test_datasets(df_train,
                                               df_test,
                                               lags=cnf.LAGS,
                                               steps_ahead=cnf.STEPS_AHEAD)
  print('#### train and test set created ####')
  print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

  # model = fc.fit_forecasting_model(X_train, y_train)

  # y_hat = fc.predict(model, X_test)
