from hdps.algorithm_steps import get_non_code_cols, step_identify_candidate_empirical_covariates, \
    step_assess_recurrence, step_prioritize_select_covariates, input_data_validation, process_outcome
from typing import Union
import pandas as pd


def HDPS_implementation(input_df: pd.DataFrame, n: int, k: int, outcome: str, treatment: str, dimension_prefixes: list, m: int =1, threshold: Union[str, float] = '75p', outcome_cont: bool =False):
    """Performs HDPS implementation for the given data.

    :param input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2', 'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns.


    :param n: int
        number of prevanlent codes to be retained in each dimension. top n of prevalent codes are selected in each dimension and rest are ignored
    :param k: int
        number of final HDPS_covariates required. top k covariates are finally selected (considering all dimensions)
    :param outcome: str
        name of the column which have outcome values.
    :param treatment: str
        name of the column which have treatment(exposure) values. This column has to be a binary column
    :param dimension_prefixes: list - list of strings
        list of name of the dimensions.
    :param m: int
        if code occur for >= m patients, that particular code is selected else dropped in each dimension. Default value for m is 1. note: m =100 as per [1] and m =1 as per [2].
    :param outcome_cont: bool
        True if outcome is continous and False if outcome is binary. Default value: False
        The HDPS implemention proposed in [1] is for binary outcome. in order to adapt the HDPS implemention for continous outcome too, below procedure is followed.

        When outcome D is Binary as per [1].
        RRcd = P(D =1 | C=1) / P(D=1|C = 0)

        Adaptation when outcome D is continuous.
        	Set threshold t which is 75th percentile value for the outcome column by default (but it could be set to median or any float or integer value if the user wishes to do so)
        	Calculate RRcd = P(D>t|C=1) / P(D>t|C=0).
        This is done by making a continuous outcome column to binary column, if value of outcome> t, then 1 else 0. After converting to binary, RRcd = P(D>t|C=1) / P(D>t|C=0) is equivalent to RRcd = P(D =1 | C=1) / P(D=1|C = 0).

    :param threshold: Union[str, float]
        applicable only if outcome_cont == True.
        a cut-off threshold to make a continous outcome to binary outcome.
        possible values for threshold are '75p', 'median', integer or float value given by the user.
        if '75p', 75th percentile value of the outcome column is taken as cut-off threshold
        if 'median', median value of the outcome column is taken as cut-off threshold
        if integer or float value, the given value is taken as cut-off threshold

    :return output_df: pandas.DataFrame
        DataFrame with columns 'PID', outcome, treatment, Demographic and Predefined covariates (if given in the input_df) and columns with HDPS covariates

    :return rank_df: pandas.DataFrame
        DataFrame with columns 'Covariates Name', 'abs_log_BiasMult' and 'rank'
        column 'Covariates Name' has names of top k HDPS covariates
        column 'abs_log_BiasMult' has the abs(log(BiasMult)) values
        column 'rank' has values that denotes the importance of HDPS covariates. Lower the number (rank) higher the importance. higher importance for covariates which has higher abs(log(BiasMult)) value.

    """
    not_code_columns = get_non_code_cols(col_names=list(input_df.columns), dimension_prefixes=dimension_prefixes)

    if outcome_cont:
        actual_outcome = input_df[outcome]
        input_df[outcome] = process_outcome(input_df=input_df, outcome=outcome, threshold=threshold)


    input_df = input_data_validation(input_df=input_df, treatment=treatment, outcome=outcome, not_code_columns=not_code_columns)

    selected_columns = step_identify_candidate_empirical_covariates(input_df=input_df,
                                                                    dimension_prefixes=dimension_prefixes, n=n, m=m)

    dim_covariates = step_assess_recurrence(input_df=input_df, selected_columns=selected_columns)

    output_df, rank_df = step_prioritize_select_covariates(dim_covariates=dim_covariates, input_df=input_df,
                                                           treatment=treatment, outcome=outcome, k=k,
                                                           not_code_columns=not_code_columns)

    if outcome_cont:
        output_df[outcome] = actual_outcome

    return output_df, rank_df