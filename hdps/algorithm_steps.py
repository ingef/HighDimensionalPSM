import math
import numpy as np
import pandas as pd
import logging


def get_non_code_cols(col_names, Dimension_prefixes):
    """function that gives the name of the columns in the input df which
    doesn't need to underdo HDPS covariates selection"""
    not_code_columns= [column for column in col_names if
                        not any(column.startswith(dim_name) for dim_name in Dimension_prefixes)]

    return not_code_columns


def step_identify_candidate_empirical_covariates(input_df, Dimension_prefixes, n, m=1):
    """In each dimension, it select the top n prevalent code columns"""
    col_names = input_df.columns

    # check for duplicates
    if np.unique(input_df['PID']).shape[0] == input_df['PID'].shape[0]:
        logging.info('No duplicates in PID column')
    else:
        logging.error('Duplicates in PID column')
        raise Exception('Duplicates in PID column')

    # calculating total study population count
    total_SP_count = input_df.shape[0]
    #print('Total study population count is ', total_SP_count)

    selected_columns = []
    for dim_name in Dimension_prefixes:

        # getting column names for the particular dimension
        dim_cols = [col for col in col_names if col.startswith(dim_name)]

        # calculating prevalance count
        prev_count = np.count_nonzero(input_df[dim_cols], axis=0)
        dim_prevalance = pd.DataFrame(data={dim_name: dim_cols, 'prevalance_count': prev_count})
        # Sorting dim_prevalance w.r.t count in decending order (high count first)
        dim_prevalance = dim_prevalance.sort_values(by='prevalance_count', ascending=False,
                                                    ignore_index=True)  # after grouping the count column is by default named as PID but it is actually prevalance count at this stage

        # Selection of codes - codes which have prevalance count >= m is retained others discarded
        dim_prevalance = dim_prevalance[dim_prevalance['prevalance_count'] >= m]

        # Making prevalance count symentric -  if less than total_SP_count/2 keep the same value of count, else total_SP_count - prevalance_count
        condition = dim_prevalance['prevalance_count'] < (total_SP_count / 2)
        dim_prevalance['prevalance_count'] = dim_prevalance['prevalance_count'].where(condition,
                                                                                      total_SP_count - dim_prevalance[
                                                                                          'prevalance_count'])  # note - tot - dim_prevalance['prevalance_count'] is applied where condition is false
        dim_prevalance = dim_prevalance.sort_values(by='prevalance_count', ascending=False, ignore_index=True)

        # Further Selection - Among the selected coded - top n codes where selected
        if dim_prevalance.shape[0] > n:
            dim_prevalance_sel = dim_prevalance[:n]

        # print(dim_prevalance_sel)
        selected_columns.extend(dim_prevalance_sel[dim_name])

    return selected_columns


def step_assess_recurrence(input_df, selected_columns):
    dim_sel_covar_counts = input_df[selected_columns]

    cov_dfs = []

    for cov in dim_sel_covar_counts.columns:

        df = pd.DataFrame(dim_sel_covar_counts[cov])
        # calculating the median of a covariates excluding 0s - if we include 0s then for most of the covariates median (or/and 75th percentile) will be 0; then value for for cov_median, cov_75p will be 1 even the code occured one time which lead to identical columns (singularity matrix problem)
        median = np.median(df[df[cov] != 0])
        # calculating the third quartile of a covariates excluding 0s
        p_75 = np.percentile(df[df[cov] != 0], 75)
        # print(cov,p_75)
        df[cov + '_onetime'] = np.where(df[cov] > 0, 1, 0)
        if median > 1:  # >1 here because if median = 1 then both covariates cov_onetime and cov_median will be identical column (and result in Singular matrix)
            df[cov + '_median'] = np.where(df[cov] >= median, 1, 0)
        if (p_75 > 1) and (
                median != p_75):  # here >1 for above reason, and != median, then cov_median and cov_75p will be same (and result in Singular matrix)
            df[cov + '_75p'] = np.where(df[cov] >= p_75, 1, 0)
        cov_dfs.append(df.iloc[:, 1:])

    dim_covariates = pd.concat(cov_dfs, axis=1)

    return dim_covariates


def step_prioritize_select_covariates(dim_covariates, input_df, treatment, outcome, k, not_code_columns):
    # list of covariates' names (columns)
    covariates_list = dim_covariates.columns

    dim_covariates_treat_outcome = pd.concat([input_df[['PID', treatment, outcome]], dim_covariates], axis=1)
    # Calculation of BiasMult and abs_log_BiasMult
    Cov_BiasMult_List = []
    for covariate in covariates_list:
        # creating the contigency table between treatment and a particular covariate in the selected covariates
        cont_tab = pd.crosstab(dim_covariates_treat_outcome[covariate], dim_covariates_treat_outcome[treatment])

        # calculating PC0 and PC1
        P_c0_doac = cont_tab.loc[1, 0] / cont_tab[0].sum()  # for treatment 0
        P_c1_vka = cont_tab.loc[1, 1] / cont_tab[1].sum()  # for treatment 1 (Alphabetical order)

        # Calculating RRcd
        cont_tab_co = pd.crosstab(dim_covariates_treat_outcome[covariate], dim_covariates_treat_outcome[outcome])
        RRcd = (cont_tab_co.loc[1, 1]/cont_tab_co.loc[1, :].sum()) / (cont_tab_co.loc[0, 1]/cont_tab_co.loc[0, :].sum())

        # calculating absolute value of log of BiasMult
        BiasMult = (P_c1_vka * (RRcd - 1) + 1) / (P_c0_doac * (RRcd - 1) + 1)

        abs_log_BiasMult = abs(
            math.log10(BiasMult))  # here log is log to base 10 # followed as reference to R implementation of HDPS

        Cov_BiasMult_List.append((covariate, BiasMult, abs_log_BiasMult))

    # creating a df with columns 'Covariates Name', 'BiasMult', 'abs_log_BiasMult' , rows are the selected covariates
    Cov_BiasMult_df = pd.DataFrame(Cov_BiasMult_List, columns=['Covariates Name', 'BiasMult', 'abs_log_BiasMult'])
    # sorting the df in desending order with respect to abs_log_BiasMult
    Cov_BiasMult_df = Cov_BiasMult_df.sort_values(by='abs_log_BiasMult', ascending=False, ignore_index=True)

    # selecting the top k  covariates with higher abs_log_BiasMult value
    if Cov_BiasMult_df.shape[0] > k:
        Cov_BiasMult_df = Cov_BiasMult_df[:k]

    # name of the k selected covariates
    sel_covariate_names = list(Cov_BiasMult_df['Covariates Name'])

    # df with selected covariates, abs_log_BiasMult and rank
    rank_df = Cov_BiasMult_df[['Covariates Name', 'abs_log_BiasMult']]
    rank_df['Rank'] = np.arange(1, (k + 1))

    # filtering those k columns
    dim_covariates_sel = dim_covariates_treat_outcome[sel_covariate_names]

    # output df
    output_df = pd.concat([input_df[not_code_columns], dim_covariates_sel], axis=1)

    print('List of selected HDPS covarities (with higher to lower values of absolute log BiasMult): ',
          sel_covariate_names)

    return output_df, rank_df


'''
References
[1] Schneeweiss, Sebastian & Rassen, Jeremy & Glynn, Robert & Avorn, Jerry & Mogun, Helen & Brookhart, M. (2009). High-Dimensional Propensity Score Adjustment in Studies of Treatment Effects Using Health Care Claims Data. Epidemiology (Cambridge, Mass.). 20. 512-22. 10.1097/EDE.0b013e3181a663cc. 
[2] Sam Lendle, "lendle/hdps: High-dimensional propensity score algorithm". link: https://rdrr.io/github/lendle/hdps/
[3] ohn Tazare & Ian Douglas & Elizabeth Williamson, 2019. "hdps: Implementation of high-dimensional propensity score approaches in Stata," London Stata Conference 2019 05, Stata Users Group. Link: https://www.stata.com/meeting/uk19/slides/uk19_tazare.pdf
[4] Schneeweiss S. Automated data-adaptive analytics for electronic healthcare data to study causal treatment effects. Clin Epidemiol. 2018;10:771-788. Published 2018 Jul 6. doi:10.2147/CLEP.S166545
[5] Wyss R, Fireman B, Rassen JA, Schneeweiss S. Erratum: High-dimensional Propensity Score Adjustment in Studies of Treatment Effects Using Health Care Claims Data. Epidemiology. 2018 Nov;29(6):e63-e64. doi: 10.1097/EDE.0000000000000886. Erratum for: Epidemiology. 2009 Jul;20(4):512-22. PMID: 29958191. 
'''