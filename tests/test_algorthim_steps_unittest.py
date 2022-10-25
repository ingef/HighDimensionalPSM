import numpy as np
import pandas as pd
from hdps.algorithm_steps import get_non_code_cols, process_outcome, input_data_validation, \
    step_identify_candidate_empirical_covariates, step_assess_recurrence, step_prioritize_select_covariates
import unittest
from hdps.exceptions import DuplicateIdError, ColumnNotBinaryError, InvalidThresholdValueError, \
    ConvertedOutcomeNotBinaryError

data_dict = {
    "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
            "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15"],
    "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
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


#############################################################
class TestCalc(unittest.TestCase):
    def test_get_non_code_cols(self):
        col_names1 = ["Treatment", "Outcome", "test_non_code_column_name", "demo_cov_1", "predef_cov_A",
                      "ICD_01", "ICD_02", "ICD_03", "OPS_06", "OPS_07"]

        dim_prefixes1 = ["ICD", "OPS"]

        assert ["Treatment", "Outcome", "test_non_code_column_name", "demo_cov_1", "predef_cov_A"] == get_non_code_cols(
            col_names=col_names1, dimension_prefixes=dim_prefixes1)

        col_names = ["Treatment", "Outcome", "test_non_code_column_name", "ICD_01", "ICD_02", "ICD_03", "OPS_06",
                     "OPS_07"]

        dim_prefixes = ["ICD", "OPS"]

        assert ["Treatment", "Outcome", "test_non_code_column_name"] == get_non_code_cols(
            col_names=col_names, dimension_prefixes=dim_prefixes)

    def test_process_outcome(self):
        data_dict1 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15", "ID16", "ID17", "ID18", "ID19", "ID20"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "Outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}

        data_df1 = pd.DataFrame(data=data_dict1)

        expected_out_array1 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        assert np.array_equal(expected_out_array1, process_outcome(input_df=data_df1, outcome="Outcome"))

        expected_out_array2 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        assert np.array_equal(expected_out_array2, process_outcome(input_df=data_df1, outcome="Outcome",
                                                                   threshold='75p'))

        expected_out_array3 = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        assert np.array_equal(expected_out_array3,
                              process_outcome(input_df=data_df1, outcome="Outcome", threshold='median'))

        expected_out_array4 = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        assert np.array_equal(expected_out_array4, process_outcome(input_df=data_df1, outcome="Outcome", threshold=3))

        expected_out_array5 = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        assert np.array_equal(expected_out_array5, process_outcome(input_df=data_df1, outcome="Outcome", threshold=4.0))

        with self.assertRaises(InvalidThresholdValueError):
            output = process_outcome(input_df=data_df1, outcome="Outcome", threshold="25p")

        with self.assertRaises(ConvertedOutcomeNotBinaryError):
            output = process_outcome(input_df=data_df1, outcome="Outcome", threshold=21)

        with self.assertRaises(ConvertedOutcomeNotBinaryError):
            output = process_outcome(input_df=data_df1, outcome="Outcome", threshold=-1)

    def test_input_data_validation(self):
        data_dict2 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69, 93, 25, 37, 13, 78],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15, 10, 36, 29, 45, 40],

            "ICD_01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # invalid column all are zeros
            "ICD_02": [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 5],
            "ICD_03": [8, 0, 0, 3, 8, 0, 0, 8, 0, 0, 3, 0, 8, 0, 3],
            "ICD_04": [1, 0, 0, 8, 0, 0, 0, 0, 0, 3, 4, 4, 0, 8, 2],
            "ICD_05": [8, 1, 3, 3, 0, 8, 0, 1, 0, 0, 4, 0, 1, 4, 8],

            "OPS_01": [9, 4, 5, 1, 2, 4, 2, 8, 6, 5, 2, 8, 6, 7, 5],  # invalid column all are non-zeros
            "OPS_02": [2, 6, 6, 9, 2, 8, 5, 5, 8, 4, 5, 0, 7, 9, 6],
            "OPS_03": [8, 1, 0, 7, 7, 2, 7, 9, 5, 4, 3, 2, 6, 6, 6]
        }

        data_df2 = pd.DataFrame(data=data_dict2)

        expected_output_df2 = data_df2.copy(deep=True).drop(columns=["ICD_01", "OPS_01"])

        assert expected_output_df2.equals(input_data_validation(
            input_df=data_df2, treatment="Treatment", outcome="Outcome",
            not_code_columns=["PID", "demo_cov_1", "demo_cov_2", "predef_cov_A", "predef_cov_B", "predef_cov_C"]))

        data_dict3 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69, 93, 25, 37, 13, 78],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15, 10, 36, 29, 45, 40],

            "ICD_01": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            "ICD_02": [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 5],
            "ICD_03": [8, 0, 0, 3, 8, 0, 0, 8, 0, 0, 3, 0, 8, 0, 3],
            "ICD_04": [1, 0, 0, 8, 0, 0, 0, 0, 0, 3, 4, 4, 0, 8, 2],
            "ICD_05": [8, 1, 3, 3, 0, 8, 0, 1, 0, 0, 4, 0, 1, 4, 8],

            "OPS_01": [9, 4, 5, 1, 2, 4, 2, 0, 6, 5, 2, 8, 6, 7, 5],
            "OPS_02": [2, 6, 6, 9, 2, 8, 5, 5, 8, 4, 5, 0, 7, 9, 6],
            "OPS_03": [8, 1, 0, 7, 7, 2, 7, 9, 5, 4, 3, 2, 6, 6, 6]
        }

        data_df3 = pd.DataFrame(data=data_dict3)

        assert data_df3.equals(input_data_validation(
            input_df=data_df3, treatment="Treatment", outcome="Outcome",
            not_code_columns=["PID", "demo_cov_1", "demo_cov_2", "predef_cov_A", "predef_cov_B", "predef_cov_C"]))

        data_dict4 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69, 93, 25, 37, 13, 78],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15, 10, 36, 29, 45, 40],

            "ICD_01": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # invalid column all are zeros
            "ICD_02": [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 5],
            "ICD_03": [8, 0, 0, 3, 8, 0, 0, 8, 0, 0, 3, 0, 8, 0, 3],
            "ICD_04": [1, 0, 0, 8, 0, 0, 0, 0, 0, 3, 4, 4, 0, 8, 2],
            "ICD_05": [8, 1, 3, 3, 0, 8, 0, 1, 0, 0, 4, 0, 1, 4, 8],

            "OPS_01": [9, 4, 5, 1, 2, 4, 2, 8, 6, 5, 2, 8, 6, 7, 5],  # invalid column all are non-zeros
            "OPS_02": [2, 6, 6, 9, 2, 8, 5, 5, 8, 4, 5, 0, 7, 9, 6],
            "OPS_03": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # invalid column all are non-zeros (ones)
        }

        data_df4 = pd.DataFrame(data=data_dict4)

        expected_output_df4 = data_df4.copy(deep=True).drop(columns=["ICD_01", "OPS_01", "OPS_03"])

        assert expected_output_df4.equals(input_data_validation(
            input_df=data_df4, treatment="Treatment", outcome="Outcome",
            not_code_columns=["PID", "demo_cov_1", "demo_cov_2", "predef_cov_A", "predef_cov_B", "predef_cov_C"]))

        data_dict6 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10"],
            "Treatment": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15],

            "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0],
            "ICD_02": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "ICD_03": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],

            "OPS_01": [0, 1, 0, 4, 0, 7, 0, 2, 0, 1],
            "OPS_02": [2, 4, 1, 1, 0, 9, 3, 1, 1, 0],
        }

        data_df6 = pd.DataFrame(data=data_dict6)
        with self.assertRaises(ColumnNotBinaryError):
            input_data_validation(input_df=data_df6, treatment="Treatment", outcome="Outcome",
                                  not_code_columns=["PID", "demo_cov_1", "demo_cov_2", "predef_cov_A", "predef_cov_B",
                                                    "predef_cov_C"])

        data_dict6 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10"],
            # "Treatment": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            # "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            "Outcome": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15],

            "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0],
            "ICD_02": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "ICD_03": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],

            "OPS_01": [0, 1, 0, 4, 0, 7, 0, 2, 0, 1],
            "OPS_02": [2, 4, 1, 1, 0, 9, 3, 1, 1, 0],
        }

        data_df6 = pd.DataFrame(data=data_dict6)

        with self.assertRaises(ColumnNotBinaryError):
            input_data_validation(input_df=data_df6, treatment="Treatment", outcome="Outcome",
                                  not_code_columns=["PID", "demo_cov_1", "demo_cov_2", "predef_cov_A", "predef_cov_B",
                                                    "predef_cov_C"])

    def test_step_identify_candidate_empirical_covariates(self):
        data_dict6 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15],

            "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # prevalence = 4
            "ICD_02": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # prevalence = 1
            "ICD_03": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # prevalence = 9 --> 10-9 = 1
            "ICD_04": [1, 2, 2, 8, 0, 0, 0, 0, 0, 3],  # prevalence = 5
            "ICD_05": [0, 1, 0, 0, 0, 8, 0, 1, 0, 0],  # prevalence = 3

            "OPS_01": [0, 1, 0, 4, 0, 7, 0, 2, 0, 1],  # prevalence = 5
            "OPS_02": [2, 4, 1, 1, 0, 9, 3, 1, 1, 0],  # prevalence = 8 --> 10-8 = 2
            "OPS_03": [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],  # prevalence = 3
            "OPS_04": [0, 0, 1, 0, 8, 0, 1, 0, 0, 0],  # prevalence = 3
            "OPS_05": [0, 0, 1, 0, 0, 0, 0, 0, 2, 0],  # prevalence = 2
            "OPS_06": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # prevalence = 1

        }

        data_df6 = pd.DataFrame(data=data_dict6)

        dim_prefixes6 = ["ICD", "OPS"]

        output_list_6ab = step_identify_candidate_empirical_covariates(input_df=data_df6,
                                                                       dimension_prefixes=dim_prefixes6,
                                                                       n=2)
        expected_output_list6a = ["ICD_04", "ICD_01", "OPS_01", "OPS_03"]
        expected_output_list6b = ["ICD_04", "ICD_01", "OPS_01", "OPS_04"]

        assert np.array_equal(np.array(expected_output_list6a), np.array(output_list_6ab)) or \
               np.array_equal(np.array(expected_output_list6b), np.array(output_list_6ab))

        output_list_6ab = step_identify_candidate_empirical_covariates(input_df=data_df6,
                                                                       dimension_prefixes=dim_prefixes6,
                                                                       n=2, m=1)

        expected_output_list6a = ["ICD_04", "ICD_01", "OPS_01", "OPS_03"]
        expected_output_list6b = ["ICD_04", "ICD_01", "OPS_01", "OPS_04"]

        assert np.array_equal(np.array(expected_output_list6a), np.array(output_list_6ab)) or \
               np.array_equal(np.array(expected_output_list6b), np.array(output_list_6ab))

        output_list_6c = step_identify_candidate_empirical_covariates(input_df=data_df6,
                                                                      dimension_prefixes=dim_prefixes6,
                                                                      n=2, m=4)

        expected_output_list6c = ["ICD_04", "ICD_01", "OPS_01", "OPS_02"]

        assert np.array_equal(np.array(expected_output_list6c), np.array(output_list_6c))

        output_list_6d = step_identify_candidate_empirical_covariates(input_df=data_df6,
                                                                      dimension_prefixes=dim_prefixes6,
                                                                      n=1, m=1)

        expected_output_list6d = ["ICD_04", "OPS_01"]

        assert np.array_equal(np.array(expected_output_list6d), np.array(output_list_6d))

        data_dict6 = {
            "PID": ["ID01", "ID01", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15],

            "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0],
            "ICD_02": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "ICD_03": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],

            "OPS_01": [0, 1, 0, 4, 0, 7, 0, 2, 0, 1],
            "OPS_02": [2, 4, 1, 1, 0, 9, 3, 1, 1, 0],
        }

        data_df6 = pd.DataFrame(data=data_dict6)

        dim_prefixes6 = ["ICD", "OPS"]
        with self.assertRaises(DuplicateIdError):
            step_identify_candidate_empirical_covariates(input_df=data_df6, dimension_prefixes=dim_prefixes6,
                                                         n=1, m=1)

    ##############################################################################
    def test_step_assess_recurrence(self):
        data_dict7 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15", "ID16", "ID17", "ID18", "ID19", "ID20"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "Outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

            "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "ICD_02": [0, 0, 1, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],
            # [1, 5, 6, 4, 8, 3, 3] --> [1, 3, 3, 4, 5, 6, 8] --> median = 4, 5<75p<6
            "ICD_03": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "ICD_04": [1, 2, 2, 8, 9, 0, 4, 0, 3, 3, 0, 0, 0, 4, 8, 6, 7, 0, 7, 0],
            # [1, 2, 2, 8, 9, 4, 3, 3, 4, 8, 6, 7, 7] --> [1, 2, 2, 3, 3, 4, 4, 6, 7, 7, 8, 8, 9] --> median = 4,75p = 7
            "ICD_05": [0, 1, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list

            "OPS_01": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # [1] , median = 1 , 75p = 1
            "OPS_02": [2, 4, 1, 1, 0, 9, 3, 1, 1, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "OPS_03": [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "OPS_04": [0, 0, 1, 0, 8, 0, 1, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "OPS_05": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],  # [2] , median = 2 , 75p = 2
            "OPS_06": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
        }

        data_df7 = pd.DataFrame(data=data_dict7)
        selected_columns = ["ICD_02", "ICD_04", "OPS_01", "OPS_05"]

        expected_out_dict7 = {
            "ICD_02_onetime": [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            "ICD_02_median": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            "ICD_02_75p": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

            "ICD_04_onetime": [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            "ICD_04_median": [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            "ICD_04_75p": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],

            "OPS_01_onetime": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

            "OPS_05_onetime": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        }

        expected_output_df7 = pd.DataFrame(data=expected_out_dict7)
        output_df7 = step_assess_recurrence(input_df=data_df7, selected_columns=selected_columns)

        assert expected_output_df7.equals(output_df7)

        data_dict8 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10", "ID11", "ID12", "ID13", "ID14", "ID15", "ID16", "ID17", "ID18", "ID19", "ID20"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            "Outcome": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

            "ICD_01": [0, 0, 0, 4, 8, 3, 3, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "ICD_02": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 9, 0, 0, 0, 0, 0, 0, 0],
            # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 9] --> min = 1 median = 1, 75p = 1
            "ICD_03": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "ICD_04": [1, 2, 2, 8, 9, 0, 4, 0, 3, 3, 0, 0, 0, 4, 8, 6, 7, 0, 7, 0],
            # [1, 2, 2, 8, 9, 4, 3, 3, 4, 8, 6, 7, 7]--> [1, 2, 2, 3, 3, 4, 4, 6, 7, 7, 8, 8, 9]--> min=1,median=4,75p=7
            "ICD_05": [0, 1, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list

            "OPS_01": [3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 8, 8, 9, 0, 0, 0, 0, 0, 0, 0],
            # [3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 8, 8, 9] , min = 3, median = 3 , 75p = 7
            "OPS_02": [2, 4, 1, 1, 0, 9, 3, 1, 1, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "OPS_03": [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "OPS_04": [0, 0, 1, 0, 8, 0, 1, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
            "OPS_05": [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 9, 0, 0, 0, 0, 0, 0, 0],
            # [2] , min= 1, median = 2 , 75p = 2
            "OPS_06": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 8, 3, 3, 0, 0, 0],  # not in select columns list
        }

        data_df8 = pd.DataFrame(data=data_dict8)
        selected_columns = ["ICD_02", "ICD_04", "OPS_01", "OPS_05"]

        expected_out_dict8 = {
            "ICD_02_onetime": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],

            "ICD_04_onetime": [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            "ICD_04_median": [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            "ICD_04_75p": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],

            "OPS_01_onetime": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "OPS_01_75p": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],

            "OPS_05_onetime": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "OPS_05_median": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        }

        expected_output_df8 = pd.DataFrame(data=expected_out_dict8)
        output_df8 = step_assess_recurrence(input_df=data_df8, selected_columns=selected_columns)

        assert expected_output_df8.equals(output_df8)

    def test_step_prioritize_select_covariates(self):
        data_dict9 = {
            "PID": ["ID01", "ID02", "ID03", "ID04", "ID05", "ID06", "ID07", "ID08",
                    "ID09", "ID10"],
            "Treatment": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            "Outcome": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            "demo_cov_1": [19, 91, 47, 26, 36, 85, 76, 46, 30, 69],
            "demo_cov_2": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            "predef_cov_A": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            "predef_cov_B": [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
            "predef_cov_C": [2, 39, 34, 3, 26, 47, 35, 31, 44, 15],

            "ICD_somecode1": [1, 0, 4, 0, 0, 0, 1, 1, 1, 1],
            "ICD_somecode2": [1, 7, 1, 0, 0, 0, 1, 1, 0, 1],

            "OPS_somecode3": [1, 1, 1, 0, 0, 0, 1, 9, 1, 1],
            "OPS_somecode4": [1, 9, 1, 0, 0, 0, 1, 9, 1, 6]
        }

        data_df9 = pd.DataFrame(data=data_dict9)

        dim_cov_dict9 = {
            "ICD_02_onetime": [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            # "Treatment":      [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],

            "ICD_04_onetime": [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
            # Pc0 = 4 / 6 = 0.6667, pc1= 1, rrcd = (2/8)/(1/2) = 0.25/0.5 = 0.5
            # outcome:         [0, 0, 1, 0, 0, 0, 0, 1, 1, 0] ## biasmult = 1 (-0.5) + 1 / 0.6667 (-0.5) +1
            # = 0.5/0.66665 = 0.7500
            # "Treatment":     [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],## log(biasmult) = -0.1249 ## abs_log_bias_mult = 0.1249
            # calculated using calculator
            "ICD_04_median": [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            "ICD_04_75p": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],

            "OPS_01_onetime": [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "OPS_01_75p": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],

            "OPS_05_onetime": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "OPS_05_median": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        }

        dim_cov_df9 = pd.DataFrame(data=dim_cov_dict9)

        non_code_columns9 = ["PID", "demo_cov_1", "demo_cov_2", "predef_cov_A", "predef_cov_B", "predef_cov_C"]
        output_output_df, output_rank_df = step_prioritize_select_covariates(
            dim_covariates=dim_cov_df9, input_df=data_df9, treatment="Treatment", outcome="Outcome",
            k=8, not_code_columns=non_code_columns9)

        expected_abs_log_biasmult_icd04_onetime = output_rank_df.set_index("Covariates Name").loc[
            "ICD_04_onetime", "abs_log_BiasMult"]
        assert (0.1249 - 0.0001) < expected_abs_log_biasmult_icd04_onetime < (0.1249 + 0.0001)

        assert max(output_rank_df.loc[0, "abs_log_BiasMult"], output_rank_df.loc[1, "abs_log_BiasMult"],
                   output_rank_df.loc[2, "abs_log_BiasMult"]) == output_rank_df.loc[0, "abs_log_BiasMult"]

        assert max(output_rank_df.loc[1, "abs_log_BiasMult"], output_rank_df.loc[2, "abs_log_BiasMult"]) == \
               output_rank_df.loc[1, "abs_log_BiasMult"]

        assert output_rank_df.shape[0] == 8

        output_output_df1, output_rank_df1 = step_prioritize_select_covariates(
            dim_covariates=dim_cov_df9, input_df=data_df9, treatment="Treatment", outcome="Outcome",
            k=2, not_code_columns=non_code_columns9)

        assert output_rank_df1.shape[0] == 2

        expected_present_columns = non_code_columns9 + list(output_rank_df1["Covariates Name"])
        for col in expected_present_columns:
            assert col in output_output_df1.columns

        expected_absent_column = output_rank_df.loc[2, "Covariates Name"]

        assert expected_absent_column not in output_output_df1.columns
