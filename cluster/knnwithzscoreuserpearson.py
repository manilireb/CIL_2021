from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreUserPearson = KNN_WithZScore(sim_name="pearson", user_based=True)

KNNWithZScoreUserPearson.log_hyperparameters_to_json()
