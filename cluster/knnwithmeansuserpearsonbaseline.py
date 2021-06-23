from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansUserPearsonBaseline = KNN_WithMeans(sim_name="pearson_baseline", user_based=True)
KNNWithMeansUserPearsonBaseline.log_hyperparameters_to_json()
