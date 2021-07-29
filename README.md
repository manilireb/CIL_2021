# Collaborative Filtering - Computational Intelligence Lab 2020


## Description 
This project was done as part of the Computational Intelligence Lab 2021 at ETH Zurich (see [Course website](http://da.inf.ethz.ch/teaching/2021/CIL/)).  The goal was to build a recommender system on user-movie ratings (see [Kaggle Competition](https://www.kaggle.com/c/cil-collaborative-filtering-2021)).

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
All our experiments for the hyperparameter tuning are logged in the [logs](https://github.com/manilireb/CIL_2021/tree/main/logs) folder. The scripts that were producing these loggs can be found in the [cluster](https://github.com/manilireb/CIL_2021/tree/main/cluster) folder.  
The scripts that produce the submission files for Kaggle can be found in the [src](https://github.com/manilireb/CIL_2021/tree/main/src) folder.  
To reproduce the final submissin run [this](https://github.com/manilireb/CIL_2021/blob/main/src/Ensemble/mlp_ridge_regressor_all.py).






### Creating the environment

Create the [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment, which contains the basic packages:

```
conda env create -n cil -f environment.yml
```
Or if you already have created the environment, update it using:

```
conda env update -n cil -f environment.yml
```

Activate the environment:

```
conda activate cil
```


### Pre-Commit Hooks  

After creating the environment run  
```
pip install pre-commit 
```
This runs the [pre-commit-hooks](https://pre-commit.com/hooks.html) specified in the [.pre-commit-config.yaml](.pre-commit-config.yaml) file before each push.

### Running
Before running make sure that the source directory is recognized by your PYTHONPATH, for example do:
```
export PYTHONPATH=/path_to_source_directory/CIL_2021:$PYTHONPATH
export PYTHONPATH=/path_to_source_directory/CIL_2021/src:$PYTHONPATH
```
### Using Leonhard Cluster  
Make sure you are connected to VPN before using the cluster.  
To run code on the Leonhard cluster follow these steps.
#### Login
```
ssh _your_nethz_username_@login.leonhard.ethz.ch
```
#### Setup
If you're logged in for the first time, go to your home directory and clone the repository.
```
cd home
git clone https://github.com/manilireb/CIL_2021.git
```
You should now have a folder called `CIL_2021`. Go to this folder and pull the latest version 
```
cd CIL_2021
git pull origin main 
```
If you're logged in for the first time, you have to create a virtual environment 
```
module load python_cpu/3.6.1
python3 -m venv ~/CIL_2021/venv
pip install --upgrade pip
```
Now run the `init_leonhard.sh` script 
```
source ./init_leonhard.sh 
```
Now you should be ready to run code on the cluster
#### Execute code
Run your batch jobs using
```
bsub -R "rusage[mem=64000]" -n 5 -W 5:00 python _your_file_.py
```
The command above submits a jobs using 5 cores with each 64000 MB with a time limit of 5 hours.  
The more resources are required, the longer it takes to schedule your job.  
Check some useful informations on your job using 
```
bjobs
```

