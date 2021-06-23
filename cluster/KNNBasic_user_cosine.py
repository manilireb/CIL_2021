from KNN_Methods.KNN_Basics import KNN_Basic

KNNBasic_user_cosine = KNN_Basic(sim_name="cosine", user_based=True)
KNNBasic_user_cosine.log_hyperparameters_to_json()
