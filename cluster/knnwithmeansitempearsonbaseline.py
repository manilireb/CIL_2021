from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansItemPearsonBaseline = KNN_WithMeans(sim_name="pearson_baseline", user_based=False)

KNNWithMeansItemPearsonBaseline.log_hyperparameters_to_json()
