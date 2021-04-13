"""
This module contains the main logic
"""

import os
import time

import numpy as np
import pandas as pd



import config as cnf
import clustering as clustering
import feature_extraction as fe
import forecasting as fc
import preprocessing as pp


if __name__ == "__main__":
    start = time.time()

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
        print('#### {:3.2f}s elapsed ####'.format(timer(features).time() - start))
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
    df_train = df_train.iloc[:25,0:50]
    df_test = df_test.iloc[:25,:]
    df_ts = pp.melt_time_series(df_train)

    features = fe.generate_features(df_ts)
    print('#### feature extraction completed ####')
    print('#### {:3.2f}s elapsed ####'.format(time.time() - start))


    # run kMeans clustering
    print('#### Starting clustering ####')
    top_kmeans_models = clustering.cluster(features.to_numpy())
    print('#### clustering completed ####')
    print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

    print('features:\n{}'.format(features))
    ts_ft_l = features.index.to_list()
    df_train_clustered = pd.DataFrame()
    for model in top_kmeans_models:
    	k = model.cluster_centers_.shape[0]
    	df_tmp = df_train.groupby('V1')\
    					.apply(clustering.get_class_label,
								model=model,
    							ts_index_l=ts_ft_l, k=k)
    	df_train_clustered = df_train_clustered.append(df_tmp)
    	# print('Cluster Center shape:\n{}'.format(model.cluster_centers_.shape))
    	# print('Labels shape:\n{}'.format(model.labels_.shape))
    	# print('No. Labels:\n{}'.format(len(model.labels_)))
    print('df_train_clustered:\n{}'.format(df_train_clustered.head()))

    #  for model in top_kmeans_model:
    #  	model.labels_

    #  	mix with df_train -> update df_test

    #  	send each to NN

    #  	compute smape, mase

    # run entire dataframe
    # smape, mase = fc.run_forecasting_process(df_train, df_test, df_ts)
    # print('sMAPE: {:.2f}\nMASE: {:.2f}'.format(smape, mase))



































