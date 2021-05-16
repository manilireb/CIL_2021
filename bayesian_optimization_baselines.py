
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt

from helpers import load_data, preprocess_data
from helpers import init_MF, compute_error


from helpers import update_user_feature, update_item_feature
from helpers import build_index_groups




# ## Implementing baselines

# ### SGD


def matrix_factorization_SGD(train, test, num_features, lambda_user, lambda_item, gamma, reg_step):
    """matrix factorization by SGD."""
    
    # define parameters
    num_epochs = 30
    errors = [0]
    
    # set seed
    np.random.seed(1)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= reg_step
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        
        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nz_test)
    
    return rmse


# ### ALS



def ALS(train, test, num_features, lambda_user, lambda_item):
    """Alternating Least Squares (ALS) algorithm."""
    
    # define parameters
    num_epochs = 30
    error_list = [0, 0]
    
    # set seed
    np.random.seed(1)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    
    for it in range(num_epochs): 
    
        # update user feature & item feature
        user_features = update_user_feature(
            train, item_features, lambda_user,
            nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(
            train, user_features, lambda_item,
            nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        error_list.append(error)

    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)

    return rmse



# ## Load the data


path_dataset = "Datasets/data_train.csv"
ratings = load_data(path_dataset)

# ## Hyperparameter Tuning

# In[11]:


from timeit import default_timer as timer
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold


# ### SGD


tuning_params = dict()

tuning_params = {
  'num_features': [5,50],
  'lambda_user': [0.0005,0.7],
  'lambda_item' : [0.0005,0.7],
  'gamma' : [0.005,0.01],
  'reg_step' : [1.1,1.2]
 }

def BO_func(num_features, lambda_user, lambda_item, gamma, reg_step):
    
    cv = 3
    kf = KFold(n_splits=cv, random_state=1, shuffle=True)
    test_RMSE_list = []
    
    for train_index, val_index in kf.split(ratings):
        ratings_train, ratings_val = ratings[train_index], ratings[val_index]
        test_RMSE = matrix_factorization_SGD(ratings_train, ratings_val, num_features = int(num_features),
                                     lambda_user = lambda_user, 
                                     lambda_item = lambda_item, gamma = gamma,
                                     reg_step = reg_step)
        test_RMSE_list.append(test_RMSE)
    
    mean_test_rmse = np.mean(test_RMSE_list)
    
    return -mean_test_rmse

optimizer = BayesianOptimization(
  f = BO_func,
  pbounds = tuning_params,
  verbose = 10,
  random_state = 5, 
 )

start = timer()
optimizer.maximize(
  init_points = 2,
  n_iter = 4
 )
end = timer()
print("execution time: {} hours.".format((end - start)/3600))


# ### ALS

tuning_params = dict()

tuning_params = {
  'num_features': [5,50],
  'lambda_user': [0.0005,0.7],
  'lambda_item' : [0.0005,0.7],
 }

def BO_func(num_features, lambda_user, lambda_item):
    
    cv = 3
    kf = KFold(n_splits=cv, random_state=1, shuffle=True)
    test_RMSE_list = []
    
    for train_index, val_index in kf.split(ratings):
        ratings_train, ratings_val = ratings[train_index], ratings[val_index]
        test_RMSE = ALS(ratings_train, ratings_val, num_features = int(num_features), 
                lambda_user = lambda_user, lambda_item = lambda_item)
        test_RMSE_list.append(test_RMSE)
    
    mean_test_rmse = np.mean(test_RMSE_list)
    
    return -mean_test_rmse

optimizer = BayesianOptimization(
  f = BO_func,
  pbounds = tuning_params,
  verbose = 10,
  random_state = 5, 
 )

start = timer()
optimizer.maximize(
  init_points = 2,
  n_iter = 4
 )
end = timer()
print("execution time: {} hours.".format((end - start)/3600))

