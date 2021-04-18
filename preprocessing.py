"""
This module handles the preprocessing steps
"""
import math
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config as cnf

#pd.set_option('display.max_columns', None)

def generate_lags(df: pd.DataFrame,
				  lags: List[int]) -> pd.DataFrame:
	"""
	generate lags for passed time series
	removes all rows with NaNs
	assumes column V1 as timeseries identifier
	assumes df is sorted
	assumes value as column from which lags are computed
	Params:
	------
	df : dataframe for which lags to be generated
	lags : list of lags to be generated
	Returns:
	--------
	df : dataframe with added lags
	--------
	"""
	print('#### generating lags ####')	
	for lag in lags:
		lagged_series = df.groupby('V1').apply(lambda x: x['value'].shift(lag))	
		df['lag_{}'.format(lag)]\
		 = lagged_series.T.squeeze().reset_index(drop=True)
		
	return df


def generate_steps_ahead(df: pd.DataFrame,
							steps_ahead: int = 7) -> pd.DataFrame:
	"""
	generates the y ahead steps for corresponding dataframe
	assumes column V1 as time series identifier
	assumes df is sorted
	assumes value as column from which steps ahead are computed
	Params:
	-------
	df : dataframe for which y_train is to be generated
	steps_ahead : steps ahead to be created / defaults to 7
	Returns:
	y_train : numpy array y_train
	"""
	print('#### Setting up y_train ####')
	for i in range(1, steps_ahead + 1):
		step_ahead = df.groupby('V1')\
			.apply(lambda x: x['value'].shift(-i))
		df['step_{}'.format(i)]\
		= step_ahead.T.squeeze().reset_index(drop=True)
	return df

def melt_time_series(df: pd.DataFrame) -> pd.DataFrame:
	"""
	shorten time series to shortest series selected
	to match required shape for NN for forecasting
	assumes column V1 is available as time series identifier
	assume timestamp is stored as V2, V3, V4, ..., Vn
	Params:
	-------
	df : dataframe to be shortened
	Returns:
	-------
	sorted_df : shortened and sorted dataframe 
	"""
	print('#### Melting time series ####')
	melted = df.melt(id_vars=['V1'])
	# add timestep that allows for sorting
	melted['timestamp'] = melted['variable'].str[1:].astype(int)
	# remove all rows with NaNs
	melted.dropna(inplace=True)
	# sort each series by timestamp and
	sorted_df = melted.sort_values(by=['V1', 'timestamp'])
	sorted_df.drop('variable', inplace=True, axis=1)
	sorted_df.reset_index(drop=True, inplace=True)
	sorted_df = sorted_df[['V1', 'timestamp', 'value']]
	return sorted_df


def standardize(df: pd.DataFrame,
				scaler: MinMaxScaler = None,
				tgt_col: str = 'value') -> Tuple[pd.DataFrame, MinMaxScaler]:
	"""
	scales all value of time series between 0 - 1
	Params:
	-------
	df : dataframe to be scaled 
	Returns:
	--------
	(df, min_max_scaler) : transformed dataframe and corresponding scaler
	"""
	if isinstance(scaler, MinMaxScaler):
		x = df[tgt_col].to_numpy().reshape(-1, 1)
		x_scaled = scaler.transform(x)
		df[tgt_col] = x_scaled
		return df
	else:
		# standardize
		x = df[tgt_col].to_numpy().reshape(-1, 1)  # returns a numpy array
		min_max_scaler = MinMaxScaler()
		x_scaled = min_max_scaler.fit_transform(x)
		df[tgt_col] = x_scaled
		return df, min_max_scaler


def destandardize(df: pd.DataFrame,
				  scaler: MinMaxScaler,
				  tgt_col: str = 'value',
				  ) -> Tuple[pd.DataFrame, MinMaxScaler]:
	"""
	reverses the scaling process
	Params:
	-------
	df : dataframe to be scaled
	scaler : MinMaxScaler that allows for the inverse transformation
	Returns:
	--------
	df : inverse transformed dataframe
	"""
	# inverse transform
	cols = df.columns
	x = df[tgt_col].to_numpy().reshape(-1, 1)  # returns a numpy array
	x_descaled = scaler.inverse_transform(x)
	df[tgt_col] = x_descaled
	return df


def normalize_data(df: pd.DataFrame,
				   standardizers: Dict[str, MinMaxScaler]=None) -> pd.DataFrame:
	"""
	normalize value column of the passed dataframe
	assumes dataframe column value requires normalization
	assumes column V1 is identifier for time series
	Params:
	------
	df : dataframe which is to be normalized
	Returns:
	(df_res, standardizer) : dataframe with normalized data and dict
						  with scaler objects
	"""
	l_df_tmp = []  # keep temp dataframe
	df_res = pd.DataFrame()
	if standardizers == None:
		standardizers = dict()
		for name, group in df.groupby("V1"):
			standardized, standardizer = standardize(group)  # returns a np.ndarray
			standardizers[name] = standardizer
			l_df_tmp.append(standardized)
		df_res = pd.concat(l_df_tmp)
		return df_res, standardizers
	else:
		for name, group in df.groupby('V1'):
			scaler = standardizers[name]
			standardized = standardize(group, scaler)
			l_df_tmp.append(standardized)
		df_res = pd.concat(l_df_tmp)
		return df_res, None


def denormalize_data(df: pd.DataFrame,
					 scaler_d: dict) -> pd.DataFrame:
	"""
	denormalize value column of the passed dataframe
	assumes dataframe column value requires denormalization
	assumes column V1 is identifier for time series
	Params:
	------
	df : dataframe which is to be normalized
	Returns:
	(df_res, standardizer) : dataframe with normalized data and dict
						  with scaler objects
	"""
	l_df_tmp = []  # keep temp dataframe
	df_res = pd.DataFrame()
	for name, group in df.groupby("V1"):
		scaler = scaler_d[name]
		destandardized = destandardize(group, scaler)  # returns a np.ndarray
		l_df_tmp.append(destandardized)
		df_res = pd.concat(l_df_tmp)
	return df_res


def create_train_test_datasets(df_train: pd.DataFrame,
							   df_test: pd.DataFrame,
							   lags: List[int] = [1, 2, 3, 4, 5, 6, 7],
							   steps_ahead: int = 7)\
	-> Tuple[np.array, np.array,
			 np.array, np.array,
			 dict, dict]:
	"""
	generates X_train, y_train, X_test, y_test from test and train dataset
	- computes the lags between test and training set and removes
	overlapping laps
	records from train set
	assumes that data structure of input matches M4 data structure and naming
	convention
	Params:
	-------
	df_train : train set dataframe
	df_test : test set dataframe
	Returns:
	(X_train, y_train,
	X_test, y_test, ts_order) : tuple of datasets, standardizers, ts_order
	"""
	# compute X_train
	df_train_tmp = melt_time_series(df_train)


	df_train_scaled, standardizers = normalize_data(df_train_tmp)
	df_X_train = generate_lags(df_train_scaled, lags)
	df_X_y_train = generate_steps_ahead(df_X_train, steps_ahead)
	# identify columns related to lags or steps ahead
	#df_X_y_train.columns
	lag_cols = [col for col in df_X_y_train.columns if ('lag_' in col)]
	step_cols = [col for col in df_X_y_train.columns if ('step_' in col)]
	# drop rows where lags are not filled (beginning all series)
	df_X_y_train.dropna(subset=lag_cols, inplace=True)


	# generate df_X_test
	df_test_tmp = melt_time_series(df_test)
	#print('df_test in cttd:\n{}'.format(df_test.head(50)))
	# df_test_scaled, test_standardizers = normalize_data(df_test_tmp)
	df_X_test_ext, df_train_last = create_test_set(df_X_y_train, df_test_tmp, standardizers)
	df_X_test_val = generate_lags(df_X_test_ext, lags)
	df_X_y_test_val = generate_steps_ahead(df_X_test_val, steps_ahead)
	# remove rows with NaNs to clean up train and test set
	#print('df_X_y_test_val in cttd:\n{}'.format(df_X_y_test_val.head(50)))
	df_X_y_test_val.dropna(inplace=True)
	df_X_y_train.dropna(inplace=True)
	

	# creating X_train and y_train numpy array
	df_X_y_train.drop(['V1', 'timestamp'], axis=1, inplace=True)
	X_train = np.asarray(df_X_y_train.drop(step_cols, axis=1))
	lag_cols.append('value')  # add value to create data structure for y_train
	y_train = np.asarray(df_X_y_train.drop(lag_cols, axis=1))
	assert X_train.shape[0] == y_train.shape[0]
	# create X_test and y_test
	ts_order = df_X_y_test_val['V1'].reset_index(drop=True)
	df_X_y_test_val.drop(['V1', 'timestamp'], axis=1, inplace=True)
	#X_test_val = np.asarray(df_X_y_test_val.drop(step_cols, axis=1))
	y_test = np.asarray(df_X_y_test_val.drop(lag_cols, axis=1))

	# last train value setup
	df_train_last.reset_index(inplace=True, drop=True)
	df_train_last.drop(step_cols, axis=1, inplace=True)
	X_test = np.asarray(df_train_last.drop(['timestamp'], axis=1))

	# data shape checks
	print('X_test shape: {}'.format(X_test.shape))
	print('y_test shape: {}'.format(y_test.shape))
	# print('y_test:\n{}'.format(y_test))
	assert X_test.shape[0]*cnf.STEPS_AHEAD == y_test.shape[0], 'X_test * steps_ahead matches y_test length'
	assert X_train.shape[1] == X_test.shape[1]
	assert y_train.shape[1] == y_test.shape[1]
	return X_train, y_train, X_test, y_test, standardizers, ts_order #, X_test_val


def modify_timestamps(df: pd.DataFrame,
					  max_steps: pd.Series) -> pd.DataFrame:
	"""
	modify end time stamps to match relationsship to train set
	i.e. test set timestamp increases from last value in train set
	Params:
	-------
	df : dataframe to which to apply timestamp modification
	max_steps: pandas series containing the last available timestamp
	Returns:
	--------
	df : dataframe with modified timestamps
	"""
	assert df['V1'].nunique() == 1
	ts_id = df['V1'].unique()[0]
	first_step_test = max_steps[ts_id] + 1
	final_step = first_step_test + df.shape[0]
	s = pd.Series([i for i in range(first_step_test, final_step)])
	df['timestamp'] = s.values
	return(df)


def create_test_set(df_train: pd.DataFrame,
					df_test: pd.DataFrame,
					standardizers: dict) -> pd.DataFrame:
	"""
	add missing lags to test set and
	adjust numbering in test set to match train set
	Params:
	-------
	df_train : dataframe train set
	df_test : dataframe test set
	scaler_d : dictionary of train scalers
	Returns:
	--------
	(df_test, test_standardizer) : adjusted test set and test standardizer

	"""
	# extract records where steps ahead are incomplete
	df_train_lags = df_train[df_train.isna().any(axis=1)].copy()

	df_train_last = df_train_lags.groupby(['V1']).max()

	df_tmp = df_train_lags[['V1', 'timestamp', 'value']]

	# retrieve last step id for each time series
	max_steps = df_tmp.groupby('V1')['timestamp'].max()

	df_test = df_test.groupby('V1', as_index=False)\
		.apply(lambda x: modify_timestamps(x, max_steps))
	df_test_scaled = normalize_data(df_test, standardizers)
	df_X_y_test_val = df_tmp.append(df_test_scaled)

	# df_X_y_test_scaled = normalize_data(df_X_y_test, standardizers)
	df_X_y_test_val.sort_values(by=['V1', 'timestamp'], inplace=True)
	df_X_y_test_val.reset_index(drop=True, inplace=True)
	#print('df_X_y_test:\n{}'.format(df_X_y_test.tail(20)))
	return df_X_y_test_val, df_train_last


def cv_train(df: pd.DataFrame,
				cur_fold: int,
				kfolds: int = cnf.KFOLD, 
				steps_ahead: int = cnf.STEPS_AHEAD) -> pd.DataFrame:
	"""
	computes the kfold training set row based on total folds and current step
	checks that step_ahead steps of forecast can still be obtained from df
	Params:
	-------
	df : datafrom from groupby split process
	cur_fold : current fold from cross validation
	kfolds : total amount of folds
	steps_ahead: forecasting steps
	Returns:
	-------
	df_ret : dataframe containing the kfold training set
	"""
	df_data = df[df.notnull()].dropna(axis=1)
	ts_tot_len = df_data.shape[1]-1 # -1 to account for V1 - name
	length = math.ceil(ts_tot_len/kfolds+1)
	train_section_end = length*cur_fold+1 # +1 to account for V1 column
	test_section_end = train_section_end+steps_ahead+7 
	if test_section_end <= df_data.shape[1]:
		df_ret = df_data.iloc[:,:train_section_end]
		#print('df_ret fold: {}\nlen: {}'.format(cur_fold, ts_tot_len))

		df = df.append(df_ret)
		#print(df)
		df.reset_index(drop=True, inplace=True)
		df = df[1:]
		return df
	else:
		return None


def cv_test(df: pd.DataFrame,
				cur_fold: int,
				kfolds: int = cnf.KFOLD, 
				steps_ahead: int = cnf.STEPS_AHEAD) -> pd.DataFrame:
	"""
	computes the kfold test set row based on total folds and current step
	checks that step_ahead steps of forecast can still be obtained from df
	Params:
	-------
	df : datafrom from groupby split process
	cur_fold : current fold from cross validation
	kfolds : total amount of folds
	steps_ahead: forecasting steps
	Returns:
	-------
	df_ret : dataframe containing the kfold training set
	"""
	df_data = df[df.notnull()].dropna(axis=1)
	ts_tot_len = df_data.shape[1]-1 # -1 to account for V1 - name
	length = math.ceil(ts_tot_len/kfolds+1)
	train_section_end = length*cur_fold+1 # +1 to account for V1 column
	#  step_ahead+7 to allow for 7 steps ahead computation
	test_section_end = train_section_end + steps_ahead+7
	if test_section_end <= df_data.shape[1]:
		# print('train_sec_end: {}'.format(train_section_end))
		# print('test_sec_end: {}'.format(test_section_end))
		df_ret = df_data.iloc[:,train_section_end:test_section_end]
		# print('df_ret - cv_test: {}'.format(df_ret))
		cols = []
		for i in range(2, len(df_ret.columns)+2):
			col = 'V{}'.format(i)
			cols.append(col)
		# cols=['V2', 'V3', 'V4', 'V5','V6','V7','V8','V9','V10','V11','V12','V13', 'V14', 'V15']
		df_ret.columns = cols
		df_ret['V1'] = df_data.iloc[0,0]

		return df_ret
	else:
		return None













