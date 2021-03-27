# CIL_2021

### Creating the environment

Create the [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment, which contains the basic packages:

```
    conda env create -n cil -f environment.yml
```

Activate the environment:

```
    conda activate cil
```


### Formatting Code 

Use .[black](https://github.com/psf/black) to format your code before pushing it.  

```
    black {source_file_or_directory}
```
