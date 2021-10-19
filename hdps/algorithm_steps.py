import math
import numpy as np
import pandas as pd
import logging
from typing import Union

logger = logging.getLogger()
logger.setLevel('INFO')


def get_non_code_cols(col_names: list, dimension_prefixes: list):
    """
    function that gives the name of the columns in the input df which
    doesn't need to underdo HDPS covariates selection

    :param col_names: list - list of strings
            list of all column names of the data frame, it may consists of names of codes with dimension name prefixes,
            demographic or predefined covariates
    :param dimension_prefixes: list - list of strings
        list of name of the dimensions.
    :return not_code_columns: list - list of strings
        list of names of columns without dimension names as prefixes
    """

    not_code_columns = [column for column in col_names if
                        not any(column.startswith(dim_name) for dim_name in dimension_prefixes)]

    return not_code_columns


def step_identify_candidate_empirical_covariates(input_df: pd.DataFrame, dimension_prefixes: list, n: int, m: int = 1):
    """
    performs selection of top n prevalent code column for each dimension

    :param input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding
        dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2',
        'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns.

    :param dimension_prefixes: list - list of strings
        list of name of the dimensions.
    :param n: int
        number of prevanlent codes to be retained in each dimension. top n of prevalent codes are selected in each
        dimension and rest are ignored
    :param m: int
        if code occur for >= m patients, that particular code is selected else dropped in each dimension. Default value
         for m is 1. note: m =100 as per [1] and m =1 as per [2].
    :return selected_columns: list - list of strings
        list of selected column names from input_df. for each dimension top n prevalent codes are selected.

    """

    col_names = input_df.columns

    # check for duplicates
    if np.unique(input_df['PID']).shape[0] == input_df['PID'].shape[0]:
        logging.info('No duplicates in PID column')
    else:
        logging.error('Duplicates in PID column')
        raise Exception('Duplicates in PID column')

    # calculating total study population count
    total_sp_count = input_df.shape[0]

    selected_columns = []
    for dim_name in dimension_prefixes:

        # getting column names for the particular dimension
        dim_cols = [col for col in col_names if col.startswith(dim_name)]

        # calculating prevalence count
        prev_count = np.count_nonzero(input_df[dim_cols], axis=0)
        dim_prevalence = pd.DataFrame(data={dim_name: dim_cols, 'prevalence_count': prev_count})
        # Sorting dim_prevalence w.r.t count in descending order (high count first)
        dim_prevalence = dim_prevalence.sort_values(by='prevalence_count', ascending=False,
                                                    ignore_index=True)
        # after grouping the count column is by default named as PID but it is actually prevalence count at this stage

        # Selection of codes - codes which have prevalence count >= m is retained others discarded
        dim_prevalence = dim_prevalence[dim_prevalence['prevalence_count'] >= m]

        # Making prevalence count symmetric -  if less than total_sp_count/2 keep the same value of count,
        # else total_sp_count - prevalence_count
        condition = dim_prevalence['prevalence_count'] < (total_sp_count / 2)
        dim_prevalence['prevalence_count'] = dim_prevalence['prevalence_count'].where(
            condition, total_sp_count - dim_prevalence['prevalence_count'])
        # note - tot - dim_prevalence['prevalence_count'] is applied where condition is false
        dim_prevalence = dim_prevalence.sort_values(by='prevalence_count', ascending=False, ignore_index=True)

        # Further Selection - Among the selected coded - top n codes where selected
        if dim_prevalence.shape[0] > n:
            dim_prevalence = dim_prevalence[:n]

        selected_columns.extend(dim_prevalence[dim_name])

    return selected_columns


def step_assess_recurrence(input_df: pd.DataFrame, selected_columns: list):
    """
    :param input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding
        dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2',
        'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns.
    :param selected_columns: list - list of strings
        list of selected column names from input_df. for each dimension top n prevalent codes are selected.
    :return dim_covariates: pandas.DataFrame
        with columns wih suffixes _ontime, _median, _75p. for each of selected_columns element, three columns with
        mentioned suffixes will be present.
        the columns with _ontime, _median, _75p are similar to _once, _sporadic, _frequent respectively in paper [1]
    """

    dim_sel_covar_counts = input_df[selected_columns]

    cov_dfs = []

    for cov in dim_sel_covar_counts.columns:

        df = pd.DataFrame(dim_sel_covar_counts[cov])
        # calculating the median of a covariates excluding 0s - if we include 0s then for most of the covariates median
        # (or/and 75th percentile) will be 0; then value for for cov_median, cov_75p will be 1 even the code occurred
        # one time which lead to identical columns (singularity matrix problem)
        median = np.median(df[df[cov] != 0])
        # calculating the third quartile of a covariates excluding 0s
        p_75 = np.percentile(df[df[cov] != 0], 75)
        # print(cov,p_75)
        df[cov + '_onetime'] = np.where(df[cov] > 0, 1, 0)
        min_value = df[df[cov] != 0][cov].min()
        if median > min_value:
            # > min_value here because if median = min_value then both covariates cov_onetime and cov_median
            # will be identical column (and result in Singular matrix)
            df[cov + '_median'] = np.where(df[cov] >= median, 1, 0)
        if (p_75 > min_value) and (median != p_75):
            # here > min_value for above reason, and != median, then cov_median and cov_75p
            # will be same (and result in Singular matrix)
            df[cov + '_75p'] = np.where(df[cov] >= p_75, 1, 0)
        cov_dfs.append(df.iloc[:, 1:])

    dim_covariates = pd.concat(cov_dfs, axis=1)

    return dim_covariates


def step_prioritize_select_covariates(dim_covariates: pd.DataFrame, input_df: pd.DataFrame, treatment: str,
                                      outcome: str, k: int, not_code_columns: list):
    """
    :param dim_covariates: pandas.DataFrame
        with columns wih suffixes _ontime, _median, _75p. for each of selected_columns element, three columns with
        mentioned suffixes will be present.
        the columns with _ontime, _median, _75p are similar to _once, _sporadic, _frequent respectively in paper [1]

    :param input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding
        dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2',
        'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns

    :param treatment: str
        name of the column which have treatment(exposure) values. This column has to be a binary column


    :param k: int
        number of final HDPS_covariates required. top k covariates are finally selected (considering all dimensions)

    :param not_code_columns: list - list of strings
        list of names of columns without dimension names as prefixes

    :param outcome: str
        name of the column which have outcome values

    :return output_df: pandas.DataFrame
        DataFrame with columns 'PID', outcome, treatment, Demographic and Predefined covariates
        (if given in the input_df) and columns with HDPS covariates

    :return rank_df: pandas.DataFrame
        DataFrame with columns 'Covariates Name', 'abs_log_BiasMult' and Rank
        column 'Covariates Name' has names of top k HDPS covariates
        column 'abs_log_BiasMult' has the abs(log(BiasMult)) values
        column 'rank' has values that denotes the importance of HDPS covariates. Lower the number (rank) higher the
        importance. higher importance for covariates which has higher abs(log(BiasMult)) value.
    """

    # list of covariates' names (columns)
    covariates_list = dim_covariates.columns

    dim_covariates_treat_outcome = pd.concat([input_df[['PID', treatment, outcome]], dim_covariates], axis=1)
    # Calculation of BiasMult and abs_log_BiasMult
    cov_bias_mult_list = []
    for covariate in covariates_list:
        # creating the contingency table between treatment and a particular covariate in the selected covariates
        cont_tab = pd.crosstab(dim_covariates_treat_outcome[covariate], dim_covariates_treat_outcome[treatment])

        # calculating PC0 and PC1
        p_c0 = cont_tab.loc[1, 0] / cont_tab[0].sum()  # for treatment 0
        p_c1 = cont_tab.loc[1, 1] / cont_tab[1].sum()  # for treatment 1

        # Calculating rrcd
        cont_tab_co = pd.crosstab(dim_covariates_treat_outcome[covariate], dim_covariates_treat_outcome[outcome])

        rrcd = (cont_tab_co.loc[1, 1]/cont_tab_co.loc[1, :].sum()) / (cont_tab_co.loc[0, 1]/cont_tab_co.loc[0, :].sum())

        # calculating absolute value of log of BiasMult
        bias_mult = (p_c1 * (rrcd - 1) + 1) / (p_c0 * (rrcd - 1) + 1)

        abs_log_bias_mult = abs(math.log10(bias_mult))
        # here log is log to base 10 # followed as reference to R implementation of HDPS

        cov_bias_mult_list.append((covariate, bias_mult, abs_log_bias_mult))

    # creating a df with columns 'Covariates Name', 'BiasMult', 'abs_log_BiasMult' , rows are the selected covariates
    cov_bias_mult_df = pd.DataFrame(cov_bias_mult_list, columns=['Covariates Name', 'BiasMult', 'abs_log_BiasMult'])
    # sorting the df in descending order with respect to abs_log_BiasMult
    cov_bias_mult_df = cov_bias_mult_df.sort_values(by='abs_log_BiasMult', ascending=False, ignore_index=True)

    # selecting the top k  covariates with higher abs_log_BiasMult value
    if cov_bias_mult_df.shape[0] > k:
        cov_bias_mult_df = cov_bias_mult_df[:k]

    # name of the k selected covariates
    sel_covariate_names = list(cov_bias_mult_df['Covariates Name'])

    # df with selected covariates, abs_log_BiasMult and rank
    rank_df = cov_bias_mult_df[['Covariates Name', 'abs_log_BiasMult']]
    rank_df['Rank'] = np.arange(1, (rank_df.shape[0] + 1))

    # filtering those k columns
    dim_covariates_sel = dim_covariates_treat_outcome[sel_covariate_names]

    # output df
    output_df = pd.concat([input_df[not_code_columns], dim_covariates_sel], axis=1)

    logging.info('List of selected HDPS covarities (with higher to lower values of absolute log BiasMult): ' +
                 str(sel_covariate_names))

    return output_df, rank_df


def input_data_validation(input_df: pd.DataFrame, treatment: str, outcome: str,
                          not_code_columns: list):
    """
    performs validation of input_df columns. Removes invalid code columns.

    :param input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding
        dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2',
        'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns

    :param treatment: str
        name of the column which have treatment(exposure) values. This column has to be a binary column

    :param outcome: str
        name of the column which have outcome values

    :param not_code_columns: list - list of strings
        list of names of columns without dimension names as prefixes

    :return: input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding
        dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2',
        'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns. This is an updated input_df where invalid code
        columns are removed.
    """

    for column in [treatment, outcome]:
        if set(input_df[column].unique()) != {0, 1}:
            error_msg = f"Treatment column and outcome column (converted outcome column when outcome is continuous)" \
                        f"needs to be binary with both expressions, but {column} has entries" \
                        f"{input_df[column].unique()}"
            logging.error(error_msg)
            raise Exception(error_msg)

    code_column_df = input_df.drop(columns=not_code_columns)
    invalid_code_columns = []

    for col in code_column_df.columns():
        code_unique_value = input_df[col].unique()

        # check for zero entry presence and at least one non-zero entry presence
        if (code_unique_value.all()) or (not code_unique_value.any()):
            invalid_code_columns.append(col)

    if len(invalid_code_columns) > 0:
        logging.warning("Some code column(s) is/are invalid. The invalid code columns are ignored. The code column is "
                        "expected to have at least one zero value and one non-zero value")
        logging.warning("List of ignored invalid code columns: " + str(invalid_code_columns))

        input_df = input_df.drop(columns=invalid_code_columns)

    return input_df


def process_outcome(input_df: pd.DataFrame, outcome: str, threshold: Union[str, float] = '75p'):
    """
    converts continuous outcome to binary outcome

    :param input_df: pandas.DataFrame
        Data frame with mandatory columns - 'PID', outcome, treatment, codes (like ICD, OPS) with corresponding
        dimension name as prefix - examples: 'DimensionName1_ICDcodeName1', 'DimensionName1_ICDcodeName2',
        'DimensionName1_ICDcodeName1', 'DimensionName2_OPScodeName1', 'DimensionName2_OPScodeName1' and
        other optional columns of predefined and demographic columns

    :param outcome: str
        name of the column which have outcome values

    :param threshold: Union[str, float]
        applicable only if outcome_cont == True.
        a cut-off threshold to make a continous outcome to binary outcome.
        possible values for threshold are '75p', 'median', integer or float value given by the user.
        if '75p', 75th percentile value of the outcome column is taken as cut-off threshold
        if 'median', median value of the outcome column is taken as cut-off threshold
        if integer or float value, the given value is taken as cut-off threshold

    :return: ndarray
        converted outcome column in binary form
    """

    logging.info('Processing Outcome')
    if threshold == '75p':
        threshold_value = np.percentile(input_df[outcome], 75)
        logging.info('Threshold is 75 percentile: ' + str(threshold_value))
    elif threshold == 'median':
        threshold_value = np.median(input_df[outcome])
        logging.info('Threshold is median: ' + str(threshold_value))
    elif isinstance(threshold, (int, float)):
        threshold_value = threshold
        logging.info('Threshold is a value given: ' + str(threshold_value))
    else:
        logging.error("Invalid value given for 'threshold' parameter")
        raise Exception("Invalid value given for 'threshold' parameter")
    # converting continuous outcome column to binary
    return np.where(input_df[outcome] > threshold_value, 1, 0)
