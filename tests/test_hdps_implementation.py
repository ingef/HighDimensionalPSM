import pandas as pd
from hdps import hdps_implementation

id_column = "PID"
n_selected_per_dimension = 3
k_selected_total = 2
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


def test_hdps_implementation():
    df, rank_df = hdps_implementation(input_df, n_selected_per_dimension, k_selected_total, "outcome", "treatment",
                                      dimension_prefixes)

    assert rank_df.loc[0]["Covariates Name"] == "ICD_3_75p"
    assert rank_df.loc[0]["Rank"] == 1
    assert rank_df.loc[1]["Covariates Name"] == "ICD_2_onetime"
    assert rank_df.loc[1]["Rank"] == 2