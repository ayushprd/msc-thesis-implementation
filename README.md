# GBMAP

This document describes how to run the experiments described in the thesis for developing and evaluating Gradient Boosting Mapping (GBMAP) which is a supervised dimensionality reduction and feature creation method.

## Installing

Here's how to install GBMAP as a python package (required for running the experiments).
```bash
cd gbmao
pip install -e .
```


Install requirements.
```bash
cd gbmap_code
pip install -r requirements.txt
```


## Run All experiments

Run all experiments.
```bash
cd experiments/thesis
bash run_experiments.sh
```
## Run specific experiments

cd  into the directory below to run specific experiments.
```bash
cd experiments/thesis
```

### Run Scaling experiment
```bash
python gbmap_scaling.py
```

### Run regression and classification experiments 
```bash
python regression_comparison.py -c reg
python regression_comparison.py -c knn
python classification_comparison.py -c cls
python classification_comparison.py -c knn
```

### Run feature creation experiments 
```bash
python feature_creation.py -c reg
python feature_creation.py -c cls
```
