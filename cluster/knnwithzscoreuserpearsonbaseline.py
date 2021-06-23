from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreUserPearsonBaseline = KNN_WithZScore(sim_name="pearson_baseline", user_based=True)

KNNWithZScoreUserPearsonBaseline.log_hyperparameters_to_json()
