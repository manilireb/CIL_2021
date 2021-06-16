from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreUserCosine = KNN_WithZScore(sim_name="cosine", user_based=True)
KNNWithZScoreUserCosine.log_hyperparameters_to_json()
