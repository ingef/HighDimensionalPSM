from hdps.algorithm_steps import get_non_code_cols, step_identify_candidate_empirical_covariates, \
    step_assess_recurrence, step_prioritize_select_covariates


def HDPS_implementation(Input_df, n, k, outcome, treatment, Dimension_prefixes, m=1):
    not_code_columns = get_non_code_cols(col_names=Input_df.columns, Dimension_prefixes=Dimension_prefixes)

    selected_columns = step_identify_candidate_empirical_covariates(Input_df=Input_df,
                                                                    Dimension_prefixes=Dimension_prefixes, n=n, m=m)

    dim_covariates = step_assess_recurrence(Input_df=Input_df, selected_columns=selected_columns)

    output_df, rank_df = step_prioritize_select_covariates(dim_covariates=dim_covariates, Input_df=Input_df,
                                                           treatment=treatment, outcome=outcome, k=k,
                                                           not_code_columns=not_code_columns)

    return output_df, rank_df
