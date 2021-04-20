"""
This module enables the feature extraction
"""
import math
from multiprocessing import Pool
import os
from typing import Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame

import config as cnf

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
	if os.path.isfile(cnf.DATA+'{}_extracted_features.csv'\
						.format(cnf.RUN_TYPE)):
		print('#### Features file exist - loading #######')
		extracted_features = pd.read_csv(cnf.DATA+'{}_extracted_features.csv'\
						.format(cnf.RUN_TYPE))
		extracted_features.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
		extracted_features.set_index('Index', inplace=True)
		standard_scaler = preprocessing.StandardScaler()
        extracted_features_scaled = pd.DataFrame(standard_scaler.fit_transform(extract_features),
        	columns=extract_features.columns,
            index=extract_features.index)

		return extracted_features_scaled
	else:
		print('#### Features file does not exist - running feature extraction #######')
		# needs to be done for each time series

		l = list(df['V1'].unique()) # getting all different time series from the list
		fc_param = dict()
		print('####  creating forecasting frame ####')
		for elm in l:
			print('#### Extr. and selecting features for\
					 series {} of {} ####'\
					.format(l.index(elm)+1,len(l)))
			df_tmp = df[df['V1']==elm]
			df_fc, y = make_forecasting_frame(df_tmp['value'],kind=elm,
										   rolling_direction=1,
										   max_timeshift=7)
			
			
			extracted_features = extract_features(df_fc,
									column_id='id',
									column_sort='time',
									column_value='value',
									impute_function=impute,
									default_fc_parameters=EfficientFCParameters())

			# verify matching index structure
			if y.index[0] in extracted_features.index:
				# do nothing as the indexes are in the same structure
				pass
			else:
				# modify y index to match extracted features index
				y.index = pd.MultiIndex.from_tuples(zip(['id']*len(y.index), y.index))

			selected_features = select_features(extracted_features, y)

			fc_param_new = from_columns(selected_features)
			
			# Python 3.9 operation to unionize dictionaries
			fc_param = fc_param | fc_param_new
			fc_param_t = dict()
			# extracting
			for key in fc_param['value']:
				fc_param_t.update({key : fc_param['value'][key]})
			
		
		print('#### Extracting relevant fts for all series ####')

		extracted_features = extract_features(df,
											column_id='V1',
											column_sort='timestamp',
											column_value='value',
											impute_function=impute,
											default_fc_parameters=fc_param_t)

                standard_scaler = preprocessing.StandardScaler()
                extracted_features_scaled = pd.DataFrame(standard_scaler.fit_transform(extract_features),
                                                         columns=extract_features.columns,
                                                         index=extract_features.index)


		extracted_features.to_csv(cnf.DATA+'{}_extracted_features.csv'\
									.format(cnf.RUN_TYPE))
                extract_features_scaled.to_csv(cnf.Data+'{}_extr_features_scaled.csv'\
                                               .scaled(cnf.RUN_TYPE))

	return extracted_features_scaled

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

