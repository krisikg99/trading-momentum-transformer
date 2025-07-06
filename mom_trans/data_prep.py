import os

import numpy as np
import pandas as pd

from mom_trans.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
)

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252


def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """
    print(file_path)
    return (
        
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )
    
def read_changepoint_results_and_fill_na_bocd(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)\
        .fillna(method="ffill")\
        .dropna()
    assert df.shape[0] > 0, f"File {file_path} is empty after reading and filling na"
    return df


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )
def prepare_cpd_features_bocd(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na_bocd(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0].replace("_bocpd", ""))
            for f in os.listdir(folder_path)
        ]
    )
    
def prepare_cpd_features_newma(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na_bocd(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0].replace("_newma", ""))
            for f in os.listdir(folder_path)
        ]
    )

from datetime import datetime

def convert_timestamps_to_days(timestamp_list):
    """
    Converts a list of pandas.Timestamp objects to their corresponding days of the week.

    Parameters:
        timestamp_list (list): A list of pandas.Timestamp objects.

    Returns:
        list: A list of strings representing the days of the week corresponding to the input timestamps.
    """
    days_of_week = []
    for timestamp in timestamp_list:
        if isinstance(timestamp, pd.Timestamp):
            # Get the day of the week from the pandas Timestamp
            days_of_week.append(timestamp.strftime("%A"))
        else:
            # Handle non-Timestamp inputs
            days_of_week.append("Invalid timestamp")
    return days_of_week

def convert_timestamps_to_day_of_month(timestamp_list):
    """
    Converts a list of pandas.Timestamp objects to their corresponding day of the month.

    Parameters:
        timestamp_list (list): A list of pandas.Timestamp objects.

    Returns:
        list: A list of integers representing the day of the month for each input timestamp,
              or "Invalid timestamp" for non-Timestamp inputs.
    """
    days_of_month = []
    for timestamp in timestamp_list:
        if isinstance(timestamp, pd.Timestamp):
            # Get the day of the month from the pandas Timestamp
            days_of_month.append(timestamp.day)
        else:
            # Handle non-Timestamp inputs
            days_of_month.append("Invalid timestamp")
    return days_of_month

def convert_timestamps_to_week_of_year(timestamp_list):
    """
    Converts a list of pandas.Timestamp objects to their corresponding week of the year.

    Parameters:
        timestamp_list (list): A list of pandas.Timestamp objects.

    Returns:
        list: A list of integers representing the week of the year for each input timestamp,
              or "Invalid timestamp" for non-Timestamp inputs.
    """
    weeks_of_year = []
    for timestamp in timestamp_list:
        if isinstance(timestamp, pd.Timestamp):
            # Get the week of the year from the pandas Timestamp
            weeks_of_year.append(timestamp.isocalendar()[1])
        else:
            # Handle non-Timestamp inputs
            weeks_of_year.append("Invalid timestamp")
    return weeks_of_year

def convert_timestamps_to_month_of_year(timestamp_list):
    """
    Converts a list of pandas.Timestamp objects to their corresponding month of the year.

    Parameters:
        timestamp_list (list): A list of pandas.Timestamp objects.

    Returns:
        list: A list of integers representing the month of the year for each input timestamp,
              or "Invalid timestamp" for non-Timestamp inputs.
    """
    months_of_year = []
    for timestamp in timestamp_list:
        if isinstance(timestamp, pd.Timestamp):
            # Get the month of the year from the pandas Timestamp
            months_of_year.append(timestamp.month)
        else:
            # Handle non-Timestamp inputs
            months_of_year.append("Invalid timestamp")
    return months_of_year

def convert_timestamps_to_year(timestamp_list):
    """
    Converts a list of pandas.Timestamp objects to their corresponding year.

    Parameters:
        timestamp_list (list): A list of pandas.Timestamp objects.

    Returns:
        list: A list of integers representing the year for each input timestamp,
              or "Invalid timestamp" for non-Timestamp inputs.
    """
    years = []
    for timestamp in timestamp_list:
        if isinstance(timestamp, pd.Timestamp):
            # Get the year from the pandas Timestamp
            years.append(timestamp.year)
        else:
            # Handle non-Timestamp inputs
            years.append("Invalid timestamp")
    return years

def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """prepare input features for deep learning model

    Args:
        df_asset (pd.DataFrame): time-series for asset with column close

    Returns:
        pd.DataFrame: input features
    """

    df_asset = df_asset[
        ~df_asset["close"].isna()
        | ~df_asset["close"].isnull()
        | (df_asset["close"] > 1e-8)  # price is zero
    ].copy()

    # winsorize using rolling 5X standard deviations to remove outliers
    df_asset["srs"] = df_asset["close"]
    ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
    df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    # vol scaling and shift to be next day returns
    df_asset["target_returns"] = calc_vol_scaled_returns(
        df_asset["daily_returns"], df_asset["daily_vol"]
    ).shift(-1)

    def calc_normalised_returns(day_offset):
        return (
            calc_returns(df_asset["srs"], day_offset)
            / df_asset["daily_vol"]
            / np.sqrt(day_offset)
        )

    df_asset["norm_daily_return"] = calc_normalised_returns(1)
    df_asset["norm_monthly_return"] = calc_normalised_returns(21)
    df_asset["norm_quarterly_return"] = calc_normalised_returns(63)
    df_asset["norm_biannual_return"] = calc_normalised_returns(126)
    df_asset["norm_annual_return"] = calc_normalised_returns(252)

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            df_asset["srs"], short_window, long_window
        )

    # date features
    if len(df_asset):
        df_asset["day_of_week"] = df_asset.index.dayofweek
        df_asset["day_of_month"] = df_asset.index.day
        df_asset["week_of_year"] = df_asset.index.weekofyear
        df_asset["month_of_year"] = df_asset.index.month
        df_asset["year"] = df_asset.index.year
        df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    else:
        df_asset["day_of_week"] = []
        df_asset["day_of_month"] = []
        df_asset["week_of_year"] = []
        df_asset["month_of_year"] = []
        df_asset["year"] = []
        df_asset["date"] = []
        
    return df_asset.dropna()


def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features


def include_changepoint_features_bocd(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features_bocd(cpd_folder_name, lookback_window_length)[
            ["ticker", "R_standard_5","R_standard_21","R_standard_cps"]
        ]
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features

def include_changepoint_features_newma(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features_newma(cpd_folder_name, lookback_window_length)[
            ["ticker","detection_stat", "online_th","online_cp"]
        ]
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features