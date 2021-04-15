"""
This module contains the main logic
"""

import os
import time



import numpy as np
import pandas as pd
from tqdm import tqdm



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
	df_train = df_train.iloc[:25,0:93]
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

	print('#### Split data by identified clusters ####')
	clustered_data,\
	clustered_data_rnd = clustering.separate_data_by_k(df_train,
													df_test, df_ts,
													features,
													top_kmeans_models)
	print('#### Splitting data to clusters completed ####')
	print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

	# running clustered time series
	print('#### Starting Forecast of identified clusters ####')
	df_res_kmeans = fc.batch_forecasting(clustered_data, 'kMeans')

	print('#### Forecast of identified completed ####')
	print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

	# running randomly clustered time series
	print('#### Starting Forecast of random clusters ####')
	df_res_rnd = fc.batch_forecasting(clustered_data_rnd, 'random')

	print('#### Forecast of random clusters completed ####')
	print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

	#running all time series in single training
	print('#### Starting Forecast of whole dataset ####')
	smape_cv, mase_cv = fc.run_cv_fc_process(df_train)
	smape_m4, mase_m4 = fc.run_forecasting_process(df_train,
													df_test,
													df_ts)
	df_k_1 = pd.DataFrame({'k': 0,
						  'class_label': 0,
						  'cluster_type': None,
						  'class_size': df_train.shape[0],
						  'smape_cv': smape_cv,
						  'mase_cv': mase_cv,
						  'smape_m4': smape_m4,
						  'mase_m4': mase_m4}, index=[0])
	print('#### Forecast of whole dataset completed ####')
	print('#### {:3.2f}s elapsed ####'.format(time.time() - start))
	
	df_res = pd.concat([df_res_kmeans, df_res_rnd, df_k_1])
	df_res.reset_index(inplace=True, drop=True)

	# write results to file
	if os.path.exists(cnf.SCORES_FOLDER_PATH):
		path = cnf.SCORES_FOLDER_PATH+'/'+cnf.FORECASTING_RES_FILE_NAME
		df_res.to_csv(path, index=False)
	else:
		os.mkdir(cnf.SCORES_FOLDER_PATH)
		path = cnf.SCORES_FOLDER_PATH+'/'+cnf.FORECASTING_RES_FILE_NAME
		df_res.to_csv(path, index=False)

	print('Results:\n{}'.format(df_res.head(101)))
	print('#### Metrics computed and forecasts finished ####')
	print('#### {:3.2f}s elapsed ####'.format(time.time() - start))



































