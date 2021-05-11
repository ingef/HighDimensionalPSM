from hdps.algorithm_steps import get_non_code_cols, step_identify_candidate_empirical_covariates, \
    step_assess_recurrence, step_prioritize_select_covariates


def HDPS_implementation(input_df, n, k, outcome, treatment, Dimension_prefixes, m=1, threshold='75p', outcome_cont=False):
    not_code_columns = get_non_code_cols(col_names=input_df.columns, Dimension_prefixes=Dimension_prefixes)

    selected_columns = step_identify_candidate_empirical_covariates(input_df=input_df,
                                                                    Dimension_prefixes=Dimension_prefixes, n=n, m=m)

    dim_covariates = step_assess_recurrence(input_df=input_df, selected_columns=selected_columns)

    output_df, rank_df = step_prioritize_select_covariates(dim_covariates=dim_covariates, input_df=input_df,
                                                           treatment=treatment, outcome=outcome, k=k,
                                                           not_code_columns=not_code_columns, threshold=threshold,
                                                           outcome_cont=outcome_cont )


    return output_df, rank_df