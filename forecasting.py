"""
This module handles the forecasting of the M4 time series.
"""
import numpy as np

# Design NN

def fit_forecasting_model(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
	"""
	executes MLP forecasting model for given training data
	"""
	inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
	x = tf.keras.layers.Dense(int(X_train.shape[1] * 1.5), activation='relu')(inputs)
	x = tf.keras.layers.Dense(int(X_train.shape[1] * 1.5), activation='relu')(x)
	x = tf.keras.layers.Dense(int(X_train.shape[1] * 1.5), activation='relu')(x)
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

	X_train_tmp, y_val_tmp, y_train_tmp, y_val_tmp\
	= train_test_split(X_train, y_train, test_size=0.2)

	nn_model.fit(X_train_temp, y_train_temp, epochs=100, batch_size=128, verbose=1, 
	            shuffle=True, callbacks=[es], validation_data=(X_val_temp, y_val_temp))
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
	y_hat = model.predict(X_test)
	return y_hat