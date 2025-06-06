#!/usr/bin/env bash

# get script path
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# regression features
python "$SCRIPTPATH/feature_creation.py" -c reg

# classification features
python "$SCRIPTPATH/feature_creation.py" -c cls

# regression comparison agains others
python "$SCRIPTPATH/regression_comparison.py" -c reg

# classification comparison against others
python "$SCRIPTPATH/classification_comparison.py" -c cls

# knn regression comparison 
python "$SCRIPTPATH/regression_comparison.py" -c knn

# knn classification comparison 
python "$SCRIPTPATH/classification_comparison.py" -c knn

# scaling  
python "$SCRIPTPATH/gbmap_scaling.py" 

# Out-of-Distribution Detection regression  
python "$SCRIPTPATH/drift.py" 

# Out-of-Distribution Detection classification  
python "$SCRIPTPATH/drift.py" -c