from hdps.algorithm_steps import *

id_column = "PID"
n_selected_per_dimension = 3
k_selected_total = 2
m_min_code_occurrence = 1
dimension_prefixes = ["ICD", "ATC", "OPS"]
col_names = [id_column, "treatment", "outcome", "ICD_1", "ICD_2", "ICD_3", "ICD_4", "ICD_5",
             "ATC_1", "ATC_2", "ATC_3", "ATC_4", "ATC_5"]
input_df = pd.DataFrame([
    ["id_1", 0, 0, 0, 1, 0, 0, 3, 1, 1, 1, 1, 0],
    ["id_2", 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    ["id_3", 0, 1, 1, 0, 1, 0, 5, 1, 1, 3, 1, 2],
    ["id_4", 0, 0, 1, 1, 1, 0, 2, 0, 1, 4, 1, 0],
    ["id_5", 0, 1, 0, 1, 3, 0, 0, 1, 2, 4, 1, 2],
    ["id_6", 1, 1, 1, 0, 5, 0, 2, 0, 2, 3, 1, 2],
    ["id_7", 1, 0, 1, 0, 2, 0, 1, 0, 2, 1, 1, 1],
    ["id_8", 1, 1, 4, 1, 4, 0, 1, 1, 1, 2, 1, 0],
    ["id_9", 1, 1, 0, 1, 0, 0, 3, 1, 1, 2, 1, 1],
    ["id_10", 1, 1, 1, 0, 0, 0, 2, 1, 1, 2, 1, 5]
    ], columns=col_names)
non_code_cols = [id_column, "treatment", "outcome"]

selected_columns = ['ATC_1', 'ATC_3', 'ATC_5', 'ICD_2', 'ICD_3', 'ICD_1']


def test_get_non_code_cols():
    assert get_non_code_cols(col_names, dimension_prefixes) == non_code_cols


def test_step_identify_candidate_empirical_covariates():

    sel_columns = step_identify_candidate_empirical_covariates(input_df=input_df,
                                                               dimension_prefixes=dimension_prefixes,
                                                               n=n_selected_per_dimension,
                                                               m=m_min_code_occurrence)

    assert set(sel_columns) == set(selected_columns)


def test_assess_recurrence():
    df = input_df[[id_column, "treatment", "outcome", *selected_columns]]
    dim_cov = step_assess_recurrence(df, selected_columns)

    assert set(dim_cov.columns) == {'ATC_1_onetime', 'ATC_3_75p', 'ATC_3_median',
                               'ATC_3_onetime', 'ATC_5_median', 'ATC_5_onetime', 'ICD_1_onetime', 'ICD_2_onetime',
                               'ICD_3_75p', 'ICD_3_median', 'ICD_3_onetime'}


def test_step_prioritize_select_covariates():
    df = input_df[[id_column, "treatment", "outcome", *selected_columns]]
    dim_cov = step_assess_recurrence(df, selected_columns)
    df, dim_cov = input_data_validation(
        input_df=df, dim_covariates=dim_cov, treatment="treatment", outcome="outcome", not_code_columns=non_code_cols)
    df, rank_df = step_prioritize_select_covariates(dim_covariates=dim_cov, input_df=df,
                                                    treatment="treatment", outcome="outcome",
                                                    k=k_selected_total, not_code_columns=non_code_cols)

    assert rank_df.loc[0]["Covariates Name"] == "ICD_3_75p"
    assert rank_df.loc[0]["Rank"] == 1
    assert rank_df.loc[1]["Covariates Name"] == "ICD_2_onetime"
    assert rank_df.loc[1]["Rank"] == 2

