# GBMAP

Gradient Boosting Mapping (GBMAP) is a supervised dimensionality reduction and feature creation method. This document describes instruction how to run the experiments described in the thesis.

## Installing

Here's how to install GBMAP as a python package (required for running the experiments).
```bash
cd gbmap_code
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
cd gbmap_code/experiments/thesis
bash run_experiments.sh
```
## Run specific experiments

Go to and run the experiments listed below.
```bash
cd gbmap_code/experiments/thesis
```

### Run Scaling experiment (Table 3)
```bash
python gbmap_scaling.py
```

### Run regression and classification experiments (Table 4)
```bash
python regression_comparison.py -c reg
python regression_comparison.py -c knn
python classification_comparison.py -c cls
python classification_comparison.py -c knn
```

### Run feature creation experiments (Figure 1 and Table 5)
```bash
python feature_creation.py -c reg
python feature_creation.py -c cls
```
