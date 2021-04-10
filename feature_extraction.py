"""
This module enables the feature extraction
"""
import os

import numpy as np
import pandas as pd
from tsfresh import extract_features, select_features 
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame

def generate_features(df: pd.DataFrame) -> np.ndarray:
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
	if os.path.isfile('features_filtered.csv'):
		print('#### Features file exist - loading #######')
		features_filtered = pd.read_csv('features_filtered.csv')
	else:
		print('#### Features file does not exist - running feature extraction #######')
		df_fc, y = make_forecasting_frame(df['value'],
										kind='daily', max_timeshift=7,
										rolling_direction=1)
		df_fc.reset_index(drop=True, inplace=True)
		# generating new dataframe that looks up target value from y via id tuple in df_fc
		extracted_features = extract_features(df_fc, column_id="id", column_sort='time',
											column_value='value', impute_function=impute)
		features_filtered = select_features(extracted_features,y.to_numpy());
		features_filtered.to_csv('features_filtered.csv', index=False)

	return features_filtered.to_numpy()