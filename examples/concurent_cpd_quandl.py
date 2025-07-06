import multiprocessing
import argparse
import os
from datetime import datetime, timedelta

from settings.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = 15 # len(QUANDL_TICKERS)

def create_date_range_processes(n, tickers, lookback_window_length):
    # Convert date strings to datetime objects
    start_date = datetime.strptime("1990-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2021-12-31", "%Y-%m-%d")
    
    # Calculate the total number of days in the date range
    total_days = (end_date - start_date).days
    
    # Calculate the number of days per period
    days_per_period = total_days / n
    
    all_processes = []
    
    for i in range(n):
        # Calculate period start date
        period_start = start_date + timedelta(days=int(days_per_period * i))
        
        # Calculate period end date
        period_end = start_date + timedelta(days=int(days_per_period * (i+1)))
        if i == n-1:  # Make sure the last period reaches the end date
            period_end = end_date
        
        # Format dates as strings
        period_start_str = period_start.strftime("%Y-%m-%d")
        period_end_str = period_end.strftime("%Y-%m-%d")
        
        # Create commands for all tickers for this period
        for ticker in tickers:
            cmd = f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "{period_start_str}" "{period_end_str}" "{lookback_window_length}"'
            all_processes.append(cmd)
    
    return all_processes

def main(lookback_window_length: int):
    if not os.path.exists(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length))

    all_processes = [
        f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}"'
        for ticker in QUANDL_TICKERS
    ]
    # all_processes = create_date_range_processes(
    #     n=5, # n_tickers 8 x n = 10 for 60 gb
    #     tickers=QUANDL_TICKERS,
    #     lookback_window_length=lookback_window_length
    # )
    
    # print(all_processes[0])
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
