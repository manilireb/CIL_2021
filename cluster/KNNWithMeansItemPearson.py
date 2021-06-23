from KNN_Methods.KNN_WithMeans import KNN_WithMeans

KNNWithMeansItemPearson = KNN_WithMeans(sim_name="pearson", user_based=False)

KNNWithMeansItemPearson.log_hyperparameters_to_json()
