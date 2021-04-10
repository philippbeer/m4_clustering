"""
This module contains the error metrics definition
"""
#import math

import numpy as np
import pandas as pd 

def compute_smape(df: pd.DataFrame) -> float:
	"""
	compute symmetric Mean Absolute Percentage
	Error for predictions in dataframe
	assumes columns true values are in column y
	assumes predictions are in columns y_hat
	Params:
	-------
	df : dataframe containing the true results and prediction
	Returns:
	smape : smape for dataset
	"""
	df = df[['y', 'y_hat']]
	ar = df.to_numpy()
	N = ar.shape[0]
	y = ar[:,0]
	y_hat = ar[:,1]
	return 1/N * np.sum(200 * np.abs(y_hat-y) / \
											(np.abs(y) + np.abs(y_hat)))


def compute_mase(df: pd.DataFrame,
				df_tr: pd.DataFrame) -> float:
	"""
	compute mean absolute scaled error for
	predictions in  dataframe
	assumes columns true values are in column y
	assumes predictions are in columns y_hat
	Params:
	-------
	df : dataframe containing the true results and prediction
	df_tr : dataframe containing the corresponding training set
	Returns:
	--------
	mase : MASE for dataset
	"""
	# df = df[['y', 'y_hat']]
	# ar = df.to_numpy()
	# N = ar.shape[0]
	# y = ar[:,0]
	# y_hat = ar[:,1]

	df_mae = df_tr.groupby('V1', as_index=False)['value']\
					.apply(lambda x: np.abs(x.diff()).sum()/(x.shape[0]-1))

	df_er = df.groupby(['V1', 'step'], as_index=False)\
			.apply(lambda x: np.mean(np.abs(x['y'].to_numpy() - x['y_hat'].to_numpy())))
	df_er.rename(columns={None : 'error'}, inplace=True)
	# df_mase = df_er.groupby(['V1', 'step'], as_index=False)\
	# 		.apply(lambda x: x['error'])
	df_mase = df_er.groupby(['V1', 'step'], as_index=False)\
			.apply(lambda x: x['error']/df_mae[df_mae['V1']==x['V1'].values[0]]['value'].values[0])
	return df_mase.mean()