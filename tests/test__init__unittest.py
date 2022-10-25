import numpy as np
import pandas as pd
import unittest
import hdps.__init__
from unittest.mock import patch

data_dict = {
    "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
            "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15"],
    "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    # "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    "Outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69, 93, 25, 37, 13, 78],
    "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15, 10, 36, 29, 45, 40],

    "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0, 1, 0, 0, 0, 8],
    "ICD_02": [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 5],
    "ICD_03": [8, 0, 0, 3, 8, 0, 0, 8, 0, 0, 3, 0, 8, 0, 3],
    "ICD_04": [1, 0, 0, 8, 0, 0, 0, 0, 0, 3, 4, 4, 0, 8, 2],
    "ICD_05": [8, 1, 3, 3, 0, 8, 0, 1, 0, 0, 4, 0, 1, 4, 8],
    "ICD_06": [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 1, 3, 1],
    "ICD_07": [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 1, 2, 1],
    "ICD_08": [0, 0, 8, 5, 0, 2, 0, 0, 0, 7, 0, 0, 7, 0, 3],
    "ICD_09": [0, 8, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    "ICD_00": [1, 0, 0, 2, 2, 0, 8, 0, 0, 2, 0, 5, 0, 5, 0],
    "ICD_10": [0, 1, 4, 6, 4, 0, 5, 0, 5, 4, 3, 5, 4, 5, 3],
    "ICD_11": [3, 1, 9, 5, 6, 5, 5, 8, 7, 7, 5, 8, 8, 3, 3],

    "OPS_01": [9, 4, 5, 1, 2, 4, 2, 8, 6, 5, 2, 8, 6, 7, 5],
    "OPS_02": [2, 6, 6, 9, 2, 8, 5, 5, 8, 4, 5, 0, 7, 9, 6],
    "OPS_03": [8, 1, 0, 7, 7, 2, 7, 9, 5, 4, 3, 2, 6, 6, 6],
    "OPS_04": [6, 3, 8, 4, 5, 1, 4, 1, 3, 5, 7, 4, 0, 3, 4],
    "OPS_05": [4, 6, 8, 6, 6, 4, 2, 5, 8, 3, 8, 1, 7, 0, 5],
    "OPS_06": [9, 8, 9, 2, 7, 0, 9, 3, 0, 9, 9, 9, 7, 6, 1],
    "OPS_07": [5, 6, 2, 8, 6, 3, 5, 2, 2, 2, 2, 8, 9, 0, 8],
    "OPS_08": [1, 2, 2, 5, 6, 9, 5, 8, 7, 8, 1, 9, 8, 1, 2],
    "OPS_09": [0, 8, 0, 8, 0, 0, 7, 8, 5, 1, 2, 7, 0, 0, 0],
    "OPS_00": [0, 0, 8, 0, 2, 7, 7, 0, 7, 0, 8, 2, 5, 0, 5],
    "OPS_10": [6, 5, 0, 0, 7, 6, 0, 1, 9, 4, 5, 6, 8, 2, 9],
    "OPS_11": [1, 4, 0, 0, 7, 5, 6, 9, 2, 6, 9, 9, 2, 5, 3],
}
data_df = pd.DataFrame(data=data_dict)

expected_outcome_col = data_df["Outcome"].copy(deep=True)

data_dict1 = {
    "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
            "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15"],
    "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    # "Outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69, 93, 25, 37, 13, 78],
    "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15, 10, 36, 29, 45, 40],

    "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0, 1, 0, 0, 0, 8],
    "ICD_02": [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 5],
    "ICD_03": [8, 0, 0, 3, 8, 0, 0, 8, 0, 0, 3, 0, 8, 0, 3],
    "ICD_04": [1, 0, 0, 8, 0, 0, 0, 0, 0, 3, 4, 4, 0, 8, 2],
    "ICD_05": [8, 1, 3, 3, 0, 8, 0, 1, 0, 0, 4, 0, 1, 4, 8],
    "ICD_06": [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 1, 3, 1],
    "ICD_07": [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 1, 2, 1],
    "ICD_08": [0, 0, 8, 5, 0, 2, 0, 0, 0, 7, 0, 0, 7, 0, 3],
    "ICD_09": [0, 8, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    "ICD_00": [1, 0, 0, 2, 2, 0, 8, 0, 0, 2, 0, 5, 0, 5, 0],
    "ICD_10": [0, 1, 4, 6, 4, 0, 5, 0, 5, 4, 3, 5, 4, 5, 3],
    "ICD_11": [3, 1, 9, 5, 6, 5, 5, 8, 7, 7, 5, 8, 8, 3, 3],

    "OPS_01": [9, 4, 5, 1, 2, 4, 2, 8, 6, 5, 2, 8, 6, 7, 5],
    "OPS_02": [2, 6, 6, 9, 2, 8, 5, 5, 8, 4, 5, 0, 7, 9, 6],
    "OPS_03": [8, 1, 0, 7, 7, 2, 7, 9, 5, 4, 3, 2, 6, 6, 6],
    "OPS_04": [6, 3, 8, 4, 5, 1, 4, 1, 3, 5, 7, 4, 0, 3, 4],
    "OPS_05": [4, 6, 8, 6, 6, 4, 2, 5, 8, 3, 8, 1, 7, 0, 5],
    "OPS_06": [9, 8, 9, 2, 7, 0, 9, 3, 0, 9, 9, 9, 7, 6, 1],
    "OPS_07": [5, 6, 2, 8, 6, 3, 5, 2, 2, 2, 2, 8, 9, 0, 8],
    "OPS_08": [1, 2, 2, 5, 6, 9, 5, 8, 7, 8, 1, 9, 8, 1, 2],
    "OPS_09": [0, 8, 0, 8, 0, 0, 7, 8, 5, 1, 2, 7, 0, 0, 0],
    "OPS_00": [0, 0, 8, 0, 2, 7, 7, 0, 7, 0, 8, 2, 5, 0, 5],
    "OPS_10": [6, 5, 0, 0, 7, 6, 0, 1, 9, 4, 5, 6, 8, 2, 9],
    "OPS_11": [1, 4, 0, 0, 7, 5, 6, 9, 2, 6, 9, 9, 2, 5, 3],
}
data_df1 = pd.DataFrame(data=data_dict1)


class TestCalc(unittest.TestCase):
    def test_hdps_implementation(self):
        out_output_df, out_rank_df = hdps.__init__.hdps_implementation(
            input_df=data_df, n=5, k=1, outcome="Outcome", treatment="Treatment",
            dimension_prefixes=["ICD", "OPS"], outcome_cont=True)

        assert out_output_df["Outcome"].equals(expected_outcome_col)

        with patch('hdps.__init__.process_outcome') as mocked_process_outcome:
            mocked_process_outcome.return_value = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0])
            out_output_df, out_rank_df = hdps.__init__.hdps_implementation(
                input_df=data_df, n=5, k=1, outcome="Outcome", treatment="Treatment",
                dimension_prefixes=["ICD", "OPS"], outcome_cont=True)
            mocked_process_outcome.assert_called_once()

            out_output_df, out_rank_df = hdps.__init__.hdps_implementation(
                input_df=data_df1, n=5, k=1, outcome="Outcome", treatment="Treatment",
                dimension_prefixes=["ICD", "OPS"], outcome_cont=False)
            mocked_process_outcome.assert_called_once()

        with patch('hdps.__init__.input_data_validation') as mocked_input_data_validation:
            mocked_input_data_validation.return_value = data_df1
            out_output_df, out_rank_df = hdps.__init__.hdps_implementation(
                input_df=data_df1, n=5, k=1, outcome="Outcome", treatment="Treatment",
                dimension_prefixes=["ICD", "OPS"], outcome_cont=False)
            mocked_input_data_validation.assert_called_once()
