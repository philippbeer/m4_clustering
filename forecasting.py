"""
This module handles the forecasting of the M4 time series.
"""
from multiprocessing import Pool
from typing import Dict, Tuple
import statistics

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from tqdm import tqdm

import config as cnf
import error_metrics as em
import preprocessing as pp
import postprocessing as postp

# set up data structures
kmeans_data = Dict[int, Dict[int, Tuple[pd.DataFrame,
											pd.DataFrame,
											pd.DataFrame]]]


def fit_forecasting_model(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
	"""
	executes MLP forecasting model for given training data
	"""
	# Design NN
	inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
	x = tf.keras.layers.Dense(
		int(X_train.shape[1] * 1.5), activation='relu')(inputs)
	x = tf.keras.layers.Dense(
		int(X_train.shape[1] * 1.5), activation='relu')(x)
	x = tf.keras.layers.Dense(
		int(X_train.shape[1] * 1.5), activation='relu')(x)
	lout = tf.keras.layers.Dense(y_train.shape[1], activation='linear')(x)
	nn_model = tf.keras.models.Model(inputs=inputs, outputs=lout)
	print(nn_model.summary())

	# Compile NN
	opt = tf.keras.optimizers.Adam(lr=0.001)
	es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
										  mode='min',
										  verbose=2,
										  patience=5,
										  min_delta=0.0001)
	nn_model.compile(optimizer=opt, loss='mse')

	X_train_tmp, X_val_tmp, y_train_tmp, y_val_tmp\
		= train_test_split(X_train, y_train, test_size=0.2)

	nn_model.fit(X_train_tmp, y_train_tmp, epochs=100, batch_size=128, verbose=1,
				 shuffle=True, callbacks=[es], validation_data=(X_val_tmp, y_val_tmp))
	return nn_model


def predict(model, X_test: np.ndarray) -> np.ndarray:
	"""
	executes prediction process
	Params:
	-------
	model : model containing predict method
	X_test: test array generating predictions
	Returns:
	--------
	y_hat : array containing predictions
	"""
	l = [] # list for individual predictions

	for j in range(len(X_test)):
		ar_tmp = X_test[j,:].reshape(1,8)
		# generating steps ahead predictions
		for i in range(cnf.STEPS_AHEAD):
			y_hat = model.predict(ar_tmp)
			ar_tmp = shift(ar_tmp[0,:], shift=1, cval=y_hat[0,0]).reshape(1,8)
			l.append(y_hat)
	# convert l into response y_test array
	y_hat = np.concatenate(l, axis=0)
	return y_hat


def run_forecasting_process(df_train, df_test, df_ts) -> Tuple[float,float]:
	"""
	run forecasting process with the provided data
	"""
	X_train, y_train,\
		X_test, y_test,\
		standardizers,\
		ts_order = pp.create_train_test_datasets(df_train,
												 df_test,
												 lags=cnf.LAGS,
												 steps_ahead=cnf.STEPS_AHEAD)

	model = fit_forecasting_model(X_train, y_train)

	y_hat = predict(model, X_test)
	df_pred = postp.postprocess(y_test, y_hat,
				standardizers, ts_order)

	df_pred.to_csv("df_pred.csv", index=False, sep=';')
	# sMAPE
	smape = em.compute_smape(df_pred)
	# MASE
	mase = em.compute_mase(df_pred, df_ts)
	return smape, mase

def run_cv_fc_process(df_train) -> Tuple[float,float]:
	"""
	run forecasting process with the provided data
	"""

	# cross validation
	smape_l = []
	mase_l = []
	for i in range(1, cnf.KFOLD+1):
		df_train_cv = df_train.groupby('V1')\
			.apply(pp.cv_train, cur_fold=i)

		df_train_cv.reset_index(drop=True, inplace=True)

		df_test_cv = df_train.groupby('V1')\
			.apply(pp.cv_test, cur_fold=i)

		assert df_train_cv.shape[0] == df_test_cv.shape[0], 'train and test set must have same time series'
		# check if length is zero and skip run
		if df_train_cv.shape[0] == 0:
			break


		# fix order columns of cv test set
		df_test_cv = df_test_cv[['V1', 'V2', 'V3', 'V4', 'V5','V6','V7','V8','V9','V10','V11','V12','V13', 'V14', 'V15']]
		df_test_cv.reset_index(drop=True, inplace=True)

		### validate


		# create df_ts
		df_ts_cv = pp.melt_time_series(df_train)


		X_train, y_train,\
			X_test, y_test,\
			standardizers,\
			ts_order = pp.create_train_test_datasets(df_train_cv,
													 df_test_cv,
													 lags=cnf.LAGS,
													 steps_ahead=cnf.STEPS_AHEAD)

		model = fit_forecasting_model(X_train, y_train)

		y_hat = predict(model, X_test)
		df_pred_cv = postp.postprocess(y_test, y_hat,
				standardizers, ts_order)

		#df_pred.to_csv("df_pred.csv", index=False, sep=';')

		# computing error metrics on cv results
		# sMAPE
		smape = em.compute_smape(df_pred_cv)
		smape_l.append(smape)
		# MASE
		mase = em.compute_mase(df_pred_cv, df_ts_cv)
		mase_l.append(mase)

	return statistics.mean(smape_l), statistics.mean(mase_l)


def batch_forecasting(clustered_data: kmeans_data,
						cluster_type: 'str') -> pd.DataFrame:
	"""
	executes forecasting process on kMeans dictionary
	and computes error metrics
	Params:
	-------
	Returns:
	--------

	"""
	# set lists for results
	k_l = [] # hold cluster type
	class_l = [] # hold class list
	class_size_l = []
	smape_cv_l = [] # hold smape results from cv
	mase_cv_l = [] # hold mase results from cv
	smape_m4_l = [] # hold smape results from cv
	mase_m4_l = [] # hold mase results from cv

	for k, classes in tqdm(clustered_data.items()): # each is model run
		for class_label, data_dfs in classes.items():
			print('Running {} k: {}, class: {} forecast'.format(cluster_type, k, class_label))
			# get class ratio
			df_train_class = data_dfs[0]
			df_test_class = data_dfs[1]
			df_ts_class = data_dfs[2]

			smape_cv, mase_cv = run_cv_fc_process(df_train_class)

			smape_m4, mase_m4 = run_forecasting_process(df_train_class,
											df_test_class, df_ts_class)
	
			# get class ratio
			class_size = df_train_class.shape[0]

			k_l.append(k)
			class_l.append(class_label)
			class_size_l.append(class_size)
			smape_cv_l.append(smape_cv)
			mase_cv_l.append(mase_cv)
			smape_m4_l.append(smape_m4)
			mase_m4_l.append(mase_m4)

	data_d = {'k': k_l,
				'cluster_type': cluster_type,
			  'class_label': class_l,
			  'class_size': class_size_l,
			  'smape_cv': smape_cv_l,
			  'mase_cv': mase_cv_l,
			  'smape_m4': smape_m4_l,
			  'mase_m4': mase_m4_l}
	df_res = pd.DataFrame(data_d)

	return df_res


def batch_forecasting_pool(clustered_data: kmeans_data,
						cluster_type: 'str',
						p: Pool) -> pd.DataFrame:
	"""
	executes forecasting process on kMeans dictionary
	and computes error metrics
	Params:
	-------
	Returns:
	--------

	"""

	args_l = []
	for k, classes in tqdm(clustered_data.items()): # each is model run
		for class_label, data_dfs in classes.items():
			print('Running {} k: {}, class: {} forecast'.format(cluster_type, k, class_label))
			# get class ratio
			df_train_class = data_dfs[0]
			df_test_class = data_dfs[1]
			df_ts_class = data_dfs[2]

			input = (k, class_label, df_train_class, df_test_class, df_ts_class)
			args_l.append(input)

	res_data = p.starmap(forecasting_worker, args_l)

	# closing the pool
	p.close()
	p.join()
	

	df_res = pd.DataFrame(res_data, columns=['k', 'class_label', 'class_size',\
											'smape_cv', 'mase_cv',
											'smape_m4', 'mase_m4'])

	df_res['cluster_type'] = cluster_type

	return df_res


def forecasting_worker(k: int, class_label: str,
					df_train_class: pd.DataFrame,
					df_test_class: pd.DataFrame,
					df_ts_class: pd.DataFrame) -> Tuple[int, str, int,
												int, int, int, int]:
	"""
	forecasting worker that executes nn
	training for cross validation and full run
	Params:
	Returns:
	k : number of clusters
	class_label : class label of currently processed class
	class_size : size of the current class
	smape_cv : sMAPE from cross-validation
	mase_cv : MASE from cross-validation
	smape_m4 : sMAPE from forecasting entire dataframe
	mase_m4 : MASE from forecasting entire dataframe
	"""
	smape_cv, mase_cv = run_cv_fc_process(df_train_class)
	smape_m4, mase_m4 = run_forecasting_process(df_train_class,
											df_test_class, df_ts_class)

	# get class ratio
	class_size = df_train_class.shape[0]
	return k, class_label, class_size,\
			smape_cv, mase_cv, smape_m4,\
			mase_m4


























