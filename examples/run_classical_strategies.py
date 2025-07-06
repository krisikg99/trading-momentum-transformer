import os
from mom_trans.backtest import run_classical_methods

INTERVALS = [(1990, y, y + 1) for y in range(2016, 2021)]

REFERENCE_EXPERIMENT = "experiment_quandl_krisi_selected_assets_lstm_cpnone_len63_time_div_v1"

features_file_path = os.path.join(
    "data",
    "quandl_cpd_nonelbw.csv",
)

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT)
