import sys

sys.path.append("../")

from KNN_Methods.KNN_Basics import KNN_Basic

"""
Please do not run this file on your local machine because it would probably take several hours to complete.
Just use it as a reference on how one can use the defined classes.
"""

KNNBasic_user_cosine = KNN_Basic(sim_name="cosine", user_based=True)
KNNBasic_user_msd = KNN_Basic(sim_name="msd", user_based=True)
KNNBasic_user_pearson = KNN_Basic(sim_name="pearson", user_based=True)
KNNBasic_user_pearson_baseline = KNN_Basic(sim_name="pearson_baseline", user_based=True)


KNNBasic_item_cosine = KNN_Basic(sim_name="cosine", user_based=False)
KNNBasic_item_msd = KNN_Basic(sim_name="msd", user_based=False)
KNNBasic_item_pearson = KNN_Basic(sim_name="pearson", user_based=False)
KNNBasic_item_pearson_baseline = KNN_Basic(sim_name="pearson_baseline", user_based=False)


KNNBasic_user_cosine.log_hyperparameters_to_json()
KNNBasic_user_msd.log_hyperparameters_to_json()
KNNBasic_user_pearson.log_hyperparameters_to_json()
KNNBasic_user_pearson_baseline.log_hyperparameters_to_json()


KNNBasic_item_cosine.log_hyperparameters_to_json()
KNNBasic_item_msd.log_hyperparameters_to_json()
KNNBasic_item_pearson.log_hyperparameters_to_json()
KNNBasic_item_pearson_baseline.log_hyperparameters_to_json()
