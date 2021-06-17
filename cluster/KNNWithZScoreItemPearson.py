from KNN_Methods.KNN_WithZScore import KNN_WithZScore

KNNWithZScoreItemPearson = KNN_WithZScore(sim_name="pearson", user_based=False)
KNNWithZScoreItemPearson.log_hyperparameters_to_json()
