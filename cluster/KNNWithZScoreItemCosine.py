from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreItemCosine = KNN_WithZScore(sim_name="cosine", user_based=False)
KNNWithZScoreItemCosine.log_hyperparameters_to_json()
