"""
This module provides the methods for the processing of y_hat
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config as cnf

def postprocess(y_test: np.ndarray,
				y_hat: np.ndarray,
				standardizers: Dict[int,MinMaxScaler],
				ts_order: pd.Series) -> pd.DataFrame:
	"""
	denormalize y_test and and y_hat
	Params:
	-------
	y_test : array with with test set
	y_hat : array with predicted y values
	standardizer : dictionary of scaler objects
	ts_order : series containing the order the time series in y_hat/y_test
	Returns:
	--------
	df : dataframe
	"""
	step = 1
	df_pred = pd.DataFrame()
	for i in range(ts_order.shape[0]):
		ts_name = ts_order[i] # getting name of time series
		scaler = standardizers[ts_name] # look up scaler for time series
		y_test_rescaled = scaler.inverse_transform(y_test[i].reshape(-1,1))
		y_hat_rescaled = scaler.inverse_transform(y_hat[i].reshape(-1,1))

		d = {'V1': ts_name,
			'step': step,
			'y': y_test_rescaled.reshape(y_test_rescaled.shape[0]),
			'y_hat': y_hat_rescaled.reshape(y_hat_rescaled.shape[0])}
		df_tmp = pd.DataFrame(d, index=range(y_test_rescaled.shape[0]))

		
		df_pred = df_pred.append(df_tmp)

		# updating forecasting steps
		if step % cnf.STEPS_AHEAD == 0:
			step = 1
		else:
			step += 1

	return df_pred




	