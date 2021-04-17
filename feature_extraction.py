"""
This module enables the feature extraction
"""
import math
from multiprocessing import Pool
import os
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame

import config as cnf

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	extract features from time series selected for their relevance to forecasting
	dataframe assumed to be in tsfresh compatible format
	Params:
	-------
	df : dataframe from which to extract time series features
	Returns:
	-------
	features_filtered : numpy array containing the 
	"""
	if os.path.isfile('extracted_features.csv'):
		print('#### Features file exist - loading #######')
		extracted_features = pd.read_csv('extracted_features.csv')
		extracted_features.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
		extracted_features.set_index('Index', inplace=True)

		return extracted_features
	else:
		print('#### Features file does not exist - running feature extraction #######')
		# needs to be done for each time series

		l = list(df['V1'].unique()) # getting all different time series from the list
		# df_fc = pd.DataFrame()
		# y = pd.Series()
		fc_param = dict()
		print('####  creating forecasting frame #######')
		for elm in l:
			print('#### Extr. and selecting features for\
					 series {} of {} #######'\
					.format(l.index(elm)+1,len(l)))
			df_tmp = df[df['V1']==elm]
			df_fc, y = make_forecasting_frame(df_tmp['value'],kind=elm,
										   rolling_direction=1,
										   max_timeshift=7)
			
			# print('df_fc shape: {}'.format(df_fc.shape))
			# print('df_fc: {}'.format(df_fc.head(25)))
			
			extracted_features = extract_features(df_fc,
									column_id='id',
									column_sort='time',
									column_value='value',
									impute_function=impute,
									default_fc_parameters=EfficientFCParameters())

			y.index = pd.MultiIndex.from_tuples(zip(['id']*len(y.index), y.index))

			selected_features = select_features(extracted_features, y)

			fc_param_new = from_columns(selected_features)
			
			# Python 3.9 operation to unionize dictionaries
			fc_param = fc_param | fc_param_new
			fc_param_t = dict()
			# extracting
			for key in fc_param['value']:
				fc_param_t.update({key : fc_param['value'][key]})
			
		#print('fc_param: {}'.format(fc_param))
		
		print('#### Extracting relevant fts for all series ####')
		# df_fc, y = make_forecasting_frame( df['value'],
		# 									kind='V1',
		# 									rolling_direction=1,
		# 									max_timeshift=7
		# 									)
		# print('df_fc: {}'.format(df_fc.head()))
		extracted_features = extract_features(df,
											column_id='V1',
											column_sort='timestamp',
											column_value='value',
											impute_function=impute,
											default_fc_parameters=fc_param_t)

		print('extr. ft.:\n{}'.format(extracted_features))
		extracted_features.to_csv('extracted_features.csv')

	return extracted_features

def generate_features_pool(df: pd.DataFrame, p: Pool) -> pd.DataFrame:
	"""
	extract features from time series selected for their relevance to forecasting
	dataframe assumed to be in tsfresh compatible format
	Params:
	-------
	df : dataframe from which to extract time series features
	p : Pool object for multiprocessing
	Returns:
	-------
	features_filtered : numpy array containing the 
	"""
	if os.path.isfile('extracted_features.csv'):
		print('#### Features file exist - loading #######')
		extracted_features = pd.read_csv('extracted_features.csv')
		extracted_features.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
		extracted_features.set_index('Index', inplace=True)

		return extracted_features
	else:
		print('#### Features file does not exist - running feature extraction #######')
		# needs to be done for each time series

		l = list(df['V1'].unique()) # getting all different time series from the list
		df_l = []
		fc_param = dict()
		print('####  prepping dfs for feature selection #######')
		for elm in l:
			df_tmp = df[df['V1']==elm]
			df_l.append(df_tmp)

		print('#### Multiprocessing Feature selection per series ####')
		# using multiprocessing for feature selection
		fc_param_l = p.map(worker_feature_selection, df_l)

		# closing the pool
		p.close()
		p.join()
		print('#### Multiprocessing Feature selection per series finished ###')

		for param_next in fc_param_l:
			# Python 3.9 operation to unionize dictionaries
			fc_param = fc_param | param_next
			
			
			# fc_param = fc_param | fc_param_new
			fc_param_res = dict()
			# extracting
			for key in fc_param['value']:
				fc_param_res.update({key : fc_param['value'][key]})
			
		#print('fc_param: {}'.format(fc_param))
		
		print('#### Extracting relevant fts for all series ####')
		# df_fc, y = make_forecasting_frame( df['value'],
		# 									kind='V1',
		# 									rolling_direction=1,
		# 									max_timeshift=7
		# 									)
		# print('df_fc: {}'.format(df_fc.head()))
		extracted_features = extract_features(df,
											column_id='V1',
											column_sort='timestamp',
											column_value='value',
											impute_function=impute,
											default_fc_parameters=fc_param_res)

		print('extr. ft.:\n{}'.format(extracted_features))
		extracted_features.to_csv('extracted_features.csv')

	return extracted_features


def worker_feature_selection(df: pd.DataFrame) -> dict:
	"""
	creates rolling window, feature extraction and selection for
	single dataframe
	Params:
	df : dataframe in tsfresh compatible dataformat to be analyzed
	Returns:
	--------
	fc_param : dictionary with most relevant forecasting parameters
	"""
	ts_id = df.iloc[0,0]
	df_fc, y = make_forecasting_frame(df['value'],kind=ts_id,
								   rolling_direction=1,
								   max_timeshift=7)
		
	extracted_features = extract_features(df_fc,
							column_id='id',
							column_sort='time',
							column_value='value',
							impute_function=impute,
							default_fc_parameters=EfficientFCParameters())

	y.index = pd.MultiIndex.from_tuples(zip(['id']*len(y.index), y.index))

	selected_features = select_features(extracted_features, y)

	fc_param = from_columns(selected_features)

	return fc_param

def make_fc_frame(df: pd.DataFrame) -> pd.DataFrame:
	"""
	creates rolling window dataframe
	to be used inside apply of groupby
	"""
	ts_id = df.iloc[0]['V1']
	df_res, y = make_forecasting_frame(df['value'],
										kind=ts_id,
										rolling_direction=1,
										max_timeshift=cnf.MAX_TIMESHIFT)
	df_res['y'] = y
	return df_res

