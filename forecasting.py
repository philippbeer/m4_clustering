"""
This module handles the forecasting of the M4 time series.
"""
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config as cnf
import error_metrics as em
import preprocessing as pp
import postprocessing as postp




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
    y_hat = model.predict(X_test)
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
