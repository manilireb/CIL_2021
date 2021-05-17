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
