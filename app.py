"""
This module contains the main logic
"""
from multiprocessing import Pool, cpu_count
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
	if not os.path.exists('data'):
		os.makedirs('data')
	print('#### reading M4 data files ####')
	if os.path.isfile(cnf.DATA+'df_{}_train_sample.csv'.format(cnf.RUN_TYPE)) and\
			os.path.isfile(cnf.DATA+'df_{}_test_sample.csv'.format(cnf.RUN_TYPE)):
		df_train = pd.read_csv(cnf.DATA+'df_{}_train_sample.csv'.format(cnf.RUN_TYPE))
		df_test = pd.read_csv(cnf.DATA+'df_{}_test_sample.csv'.format(cnf.RUN_TYPE))
		print('### local sample files read ###')
		print('#### {:3.2f}s elapsed ####'.format(time.time() - start))
	else:
		train = pd.read_csv(cnf.CUR_RUN_TRAIN)
		test = pd.read_csv(cnf.CUR_RUN_TEST)
		elapsed = time.time()
		print('#### finished reading CSV files####')
		print('#### {:3.2f}s elapsed ####'.format(time.time() - start))
		# sample training data if set
		if cnf.SAMPLING:
			print('#### beginning sampling of time series ####')
			df_train = train.sample(frac=cnf.SAMPLING_RATE)
			# restrict test data to sampled training data
			df_test = test.iloc[df_train.index]
				# write sample to local file for faster loading
			if not os.path.isfile(cnf.DATA+'df_{}_train_sample.csv'.format(cnf.RUN_TYPE)) and\
				not os.path.isfile(cnf.DATA+'df_{}_test_sample.csv'.format(cnf.RUN_TYPE)):
				df_train.to_csv(cnf.DATA+'df_{}_train_sample.csv'.format(cnf.RUN_TYPE),
								index=False)
				df_test.to_csv(cnf.DATA+'df_{}_test_sample.csv'.format(cnf.RUN_TYPE),
								index=False)

				print('#### sampling completed ####')
				print('#### {:3.2f}s elapsed ####'.format(time.time() - start))		
		else:
			print('#### Running with complete dataset ####')
			df_train = train
			df_test = test

	# restrict data for development
	df_train = df_train.iloc[:5,0:200]
	df_test = df_test.iloc[:5,:]

	# format data for tsfresh
	df_ts = pp.melt_time_series(df_train)

	
	# generate features from time series
	print('#### Starting feature extraction ####')
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
	num_proc = cpu_count() - 1
	p = Pool(num_proc)
	df_res_kmeans = fc.batch_forecasting_pool(clustered_data, 'kMeans', p)
	#df_res_kmeans = fc.batch_forecasting(clustered_data, 'kMeans')

	print('#### Forecast of identified completed ####')
	print('#### {:3.2f}s elapsed ####'.format(time.time() - start))

	# running randomly clustered time series
	print('#### Starting Forecast of random clusters ####')
	num_proc = cpu_count() - 1
	p = Pool(num_proc)
	df_res_rnd = fc.batch_forecasting_pool(clustered_data_rnd, 'random', p)
	#df_res_rnd = fc.batch_forecasting(clustered_data_rnd, 'random')

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
						  'cluster_type': 'all_records',
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



































