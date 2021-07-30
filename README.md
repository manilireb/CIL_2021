# Collaborative Filtering - Computational Intelligence Lab 2020


## Description 
This project was done as part of the Computational Intelligence Lab 2021 at ETH Zurich (see [Course website](http://da.inf.ethz.ch/teaching/2021/CIL/)).  The goal was to build a recommender system on user-movie ratings (see [Kaggle Competition](https://www.kaggle.com/c/cil-collaborative-filtering-2021)).  
Take a look at the [report](TODO) for the details of our approach.

## Setup
The experiments were run and tested with `Python 3.6.1`.  
  
Clone project: 
```
git clone https://github.com/manilireb/CIL_2021.git
```
Before running make sure that the source directory is recognized by your PYTHONPATH, e.g. do
```
export PYTHONPATH=/path_to_source_directory/CIL_2021:$PYTHONPATH
export PYTHONPATH=/path_to_source_directory/CIL_2021/src:$PYTHONPATH
```
Install the virtual environment

```
cd CIL_2021
python3 -m venv ~/CIL_2021/venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Setup on Leonhard Cluster

Most of the experiments were run on the [Leonhard Cluster](https://scicomp.ethz.ch/wiki/Leonhard). If you like to reproduce the experiments, set up the cluster as follows:
```
cd home
git clone https://github.com/manilireb/CIL_2021.git
cd CIL_2021
pip install --upgrade pip
python3 -m venv ~/CIL_2021/venv
source ./init_leonhard.sh 
```

## Reproduce Experiments
All our experiments for the hyperparameter tuning and the model evaluation are logged in the [logs](https://github.com/manilireb/CIL_2021/tree/main/logs) folder. The scripts that were producing these logs can be found in the [cluster](https://github.com/manilireb/CIL_2021/tree/main/cluster) folder.  
The scripts that produce the submission files for Kaggle can be found in the [src](https://github.com/manilireb/CIL_2021/tree/main/src) folder.  
To reproduce the final submissin run [this](https://github.com/manilireb/CIL_2021/blob/main/src/Ensemble/mlp_ridge_regressor_all.py) file.

## Not Using Git
If you downloaded the this repo without using git you have to initiate a git repo with 
```
git init
```
inside the folder. The reason is that we are using the `get_git_root()` function to navigate through the folders.

## Problems
For problems or further questions, don't hesitate to ask at manuel.reber@inf.ethz.ch.
