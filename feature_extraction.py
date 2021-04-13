"""
This module enables the feature extraction
"""
import os
from typing import Union

import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
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
	else:
		print('#### Features file does not exist - running feature extraction #######')
		# needs to be done for each time series

		df_fc = df.groupby('V1').apply(make_fc_frame)
		y = df_fc['y']
		print('df_fc:\n{}'.format(df_fc.head(70)))
		
		extracted_features = extract_features(df_fc, column_id="kind", column_sort='time',
											column_value='value', impute_function=impute,
											default_fc_parameters=EfficientFCParameters())

		extracted_features.to_csv('extracted_features.csv')

	return extracted_features

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

