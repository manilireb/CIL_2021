# CIL_2021

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
python 3 -m venv ~/CIL_2021/venv
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
The more resources are required, the it takes to schedule your job.  
check some useful informations on your job using 
```
bjobs
```

