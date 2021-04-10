"""
This module handles the preprocessing steps
"""
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    df : dataframe with added lags
    --------
    """
    print('#### generating lags ####')
    for lag in lags:
        lagged_series = df.groupby('V1',
                                   as_index=False)\
            .apply(lambda x: x['value'].shift(lag)).values
        df['lag_{}'.format(lag)] = lagged_series
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
        step_ahead = df.groupby('V1', as_index=True)\
            .apply(lambda x: x['value'].shift(-i)).values
        df['step_{}'.format(i)] = step_ahead
    return df


def shorten_time_series(df: pd.DataFrame) -> pd.DataFrame:
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
    print('#### Shortening all series to shortest time series ####')
    melted = df.melt(id_vars=['V1'])
    # add timestep that allows for sorting
    melted['timestamp'] = melted['variable'].str[1:].astype(int)
    # remove all rows with NaNs
    melted.dropna(inplace=True)
    # get shortest time series
    shortest_series_len = melted.groupby(
        'V1')['value'].count().sort_values()[0]
    # sort each series by timestamp and
    # shorten series length of shortest (backwards from last timestamp)
    sorted_df = melted.groupby('V1', as_index=False)\
        .apply(lambda x: x.sort_values(by='timestamp', ascending=True)
               [-shortest_series_len:])
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
                   standardizers: Dict[str, MinMaxScaler]=None) -> Tuple[pd.DataFrame,
                                                                         Dict[str, MinMaxScaler]]:
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
        return df_res


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
    df_train_tmp = shorten_time_series(df_train)
    df_train_scaled, standardizers = normalize_data(df_train_tmp)
    df_X_train = generate_lags(df_train_scaled, lags)
    df_X_y_train = generate_steps_ahead(df_X_train, steps_ahead)
    # identify columns related to lags or steps ahead
    df_X_y_train.columns
    lag_cols = [col for col in df_X_y_train.columns if ('lag_' in col)]
    step_cols = [col for col in df_X_y_train.columns if ('step_' in col)]
    # drop rows where lags are not filled (beginning all series)
    df_X_y_train.dropna(subset=lag_cols, inplace=True)

    # generate df_X_test
    df_test_tmp = shorten_time_series(df_test)
    # df_test_scaled, test_standardizers = normalize_data(df_test_tmp)
    df_X_test_ext = create_test_set(df_X_y_train, df_test_tmp, standardizers)
    df_X_test = generate_lags(df_X_test_ext, lags)
    df_X_y_test = generate_steps_ahead(df_X_test, steps_ahead)
    # remove rows with NaNs to clean up train and test set
    df_X_y_test.dropna(inplace=True)
    df_X_y_train.dropna(inplace=True)

    # creating X_train and y_train numpy array
    df_X_y_train.drop(['V1', 'timestamp'], axis=1, inplace=True)
    X_train = np.asarray(df_X_y_train.drop(step_cols, axis=1))
    lag_cols.append('value')  # add value to create data structure for y_train
    y_train = np.asarray(df_X_y_train.drop(lag_cols, axis=1))
    assert X_train.shape[0] == y_train.shape[0]

    # create X_test and y_test
    ts_order = df_X_y_test['V1'].reset_index(drop=True)
    df_X_y_test.drop(['V1', 'timestamp'], axis=1, inplace=True)
    X_test = np.asarray(df_X_y_test.drop(step_cols, axis=1))
    y_test = np.asarray(df_X_y_test.drop(lag_cols, axis=1))
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]
    return X_train, y_train, X_test, y_test, standardizers, ts_order


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
    scaler_d : dictonary of train scalers
    Returns:
    --------
    (df_test, test_standardizer) : adjusted test set and test standardizer

    """
    # extract records where steps ahead are incomplete
    df_train_lags = df_train[df_train.isna().any(axis=1)].copy()
    df_tmp = df_train_lags[['V1', 'timestamp', 'value']]
    # retrieve last step id for each time series
    max_steps = df_tmp.groupby('V1')['timestamp'].max()
    df_test = df_test.groupby('V1', as_index=False)\
        .apply(lambda x: modify_timestamps(x, max_steps))
    df_test_scaled = normalize_data(df_test, standardizers)
    df_X_y_test = df_tmp.append(df_test_scaled)
    # df_X_y_test_scaled = normalize_data(df_X_y_test, standardizers)
    df_X_y_test.sort_values(by=['V1', 'timestamp'], inplace=True)
    df_X_y_test.reset_index(drop=True, inplace=True)
    return df_X_y_test
