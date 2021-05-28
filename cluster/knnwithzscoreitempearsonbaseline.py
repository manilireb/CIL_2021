from KNN_Methods.KNN_WithZScore import KNN_WithZScore

knn = KNN_WithZScore(sim_name="pearson_baseline", user_based=False)
knn.log_hyperparameters_to_json()
