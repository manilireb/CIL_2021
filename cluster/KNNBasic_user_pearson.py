from KNN_Methods.KNN_Basics import KNN_Basic

KNNBasic_user_pearson = KNN_Basic(sim_name="pearson", user_based=True)
KNNBasic_user_pearson.log_hyperparameters_to_json()
