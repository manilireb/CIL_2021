# ML_Project2
## Dependencies
Before running the code make sure the following packeages are installed:
* numpy 
* scipy 
* matplotlib.pyplot 
* surprise 
* pandas 
* seaborn 
* csv  
* json 
* bayes_opt 

Caution: In some parts of the code parallel computing is used, it is set to use all vCPUs, however the maximum of parallel jobs is 3, when running the program make sure not to run any other demanding tasks!

## Running the code

### `run.py`
The `run.py` script creates the submission file. Only the Ensemble method is run and the submission csv is created. This is due to the fact that training the models and generate the prediction for the ensemble model takes a consideral amount of time. 

### Running the full project
Because running the full project takes a considerable amount of time, the project is split up in several scripts. If the full code is to be run it has to be done in the following order:

1. `bayesian_optimization_baselines.py`
2. `submission_tuned_baseline`
3. `parser.py` 
4. `hyperparamter_tuning.py`
5. `testset_prediction.py`
6. `Ensemble.py`

This is important, because all the scripts produce outputfiles that are used by other scripts. This will create the same final output as `run.py`.


## Structure 

The output of `run.py` goes to `submission`.

`Notebooks` contains the jupyter notebooks on which python scripts are based. They were used to develop the project.

`Scripts` contains the python scripts. Furthermore it includes the folder `outputs` which contains the output files of the scripts. `run.py` loads these outputfiles to generate the submisson. It also includes the `Datasets` folder which holds the raw data sets called `data_train.csv` and `sample_submission.csv`.

### `Scripts`
The following python scritps are included in the folder:
* `helpers.py` this script includes some functions that are used by `bayesian_optimization_baselines.py` and `submission_tuned_baseline`.
* `bayesian_optimization_baselines.py` this scripts performes the hyperparameter tuning for the ALS and SGD matrix factorization.
* `submission_tuned_baseline` this script, performs the prediction on the test set with the tuned ALS and SGD matrix factorization methods and creates two csv files that can be uploaded to aicrowd.
* `parser.py` this script takes the raw dataset `data_train.csv` and `sample_submission.csv` as inputs and returns `trainset.csv` and `testset.csv` which are formated to be used by the surprise package.
* `hyperparamter_tuning.py` this script uses bayesian optimization cross validation to optimize the hyperparameters of the machine learning algorithms. For each machine learning algorithm it returns a json file with the hyperparameters and RMSE for each iteration of the bayesian optimization.
* `testset_prediction.py` creates a csv file for each tuned model, containing the predicted ratings on the testset, that can be uploaded to aicrowd.
* `Ensemble.py` in this script the ridge regression whith the best method from each class of models is performed to get an ensemble prediction on the testset. Please consult the report for a detailed description.

## Further information
The code was tested on Windows 10 as well as Linux (Pop!-OS 19.4). It is advised to have at least 4 vCPUs to have one free thread left over when running the code.
