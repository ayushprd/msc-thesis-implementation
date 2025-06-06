from gbmap.embedding.gbnn_four import GBMAP4
from gbmap_knn import KNNReg, KNNCls, KNNWithGBMAPReg, KNNWithGBMAPCls
from consts import RANDOM_SEED

import numpy as np

from scipy.stats import uniform

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


REGRESSION_FULL = {
    "results_filename": "regression",
    "model_config": {
        "linreg": {
            "model": LinearRegression,
            "random_search_iters": 0,
            "params": {},
        },
        "xgboost": {
            "model": XGBRegressor,
            "random_search_iters": 100,
            "params": {
                "n_estimators": range(100, 2001),
                "max_depth": range(1, 6),
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                "random_state": [RANDOM_SEED],
            },
        },
        "gbmap": {
            "model": GBMAP4,
            "random_search_iters": 100,
            "params": {
                "n_boosts": range(2, 151),
                "softplus_scale": range(1, 21),
                "optim_maxiter": [200, 400],
                "penalty_weight": uniform(loc=0, scale=1e-2),
                "random_state": [RANDOM_SEED],
            },
        },
    },
}


REGRESSION_KNN = {
    "results_filename": "regression_knn",
    "model_config": {
        "knn": {
            "model": KNeighborsRegressor,
            "random_search_iters": 0,
            "params": {"n_neighbors": 10, "metric": "minkowski", "algorithm": "brute"},
        },
        "knnwithgbmap": {
            "model": KNNWithGBMAPReg,
            "random_search_iters": 50,
            "params": {
                "n_boosts": range(2, 51),
                "softplus_scale": range(1, 21),
                "optim_maxiter": [200, 400],
                "penalty_weight": uniform(loc=0, scale=1e-2),
                "random_state": [RANDOM_SEED],
            },
        },
    },
}


REGRESSION_GBMAP = {
    "results_filename": "regression_gbmap",
    "model_config": {
        "gbmap": {
            "model": GBMAP4,
            "random_search_iters": 100,
            "params": {
                "n_boosts": range(2, 151),
                "softplus_scale": range(1, 21),
                "optim_maxiter": [200, 400],
                "penalty_weight": uniform(loc=0, scale=1e-2),
                "random_state": [RANDOM_SEED],
            },
        },
    },
}


CLASSIFICATION_FULL = {
    "results_filename": "classification",
    "model_config": {
        "logreg": {
            "model": LogisticRegression,
            "random_search_iters": 100,
            "params": {
                "penalty": ["l2"],
                "C": uniform(loc=1, scale=1e5),
                "max_iter": [500],
                "random_state": [RANDOM_SEED],
            },
        },
        "xgboost": {
            "model": XGBClassifier,
            "random_search_iters": 100,
            "params": {
                "n_estimators": range(100, 2001, 1),
                "max_depth": range(1, 6),
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                "random_state": [RANDOM_SEED],
            },
        },
        "gbmap": {
            "model": GBMAP4,
            "random_search_iters": 100,
            "params": {
                "n_boosts": range(2, 151),
                "softplus_scale": range(1, 21),
                "optim_maxiter": [200, 400],
                "penalty_weight": uniform(loc=0, scale=1e-2),
                "is_classifier": [True],
                "random_state": [RANDOM_SEED],
            },
        },
    },
}


CLASSIFICATION_KNN = {
    "results_filename": "classification_knn",
    "model_config": {
        "knn": {
            "model": KNNCls,
            "random_search_iters": 0,
            "params": {},
        },
        "knnwithgbmap": {
            "model": KNNWithGBMAPCls,
            "random_search_iters": 50,
            "params": {
                "n_boosts": range(2, 51),
                "softplus_scale": range(1, 21),
                "optim_maxiter": [200, 400],
                "penalty_weight": uniform(loc=0, scale=1e-2),
                "random_state": [RANDOM_SEED],
                "is_classifier": [True],
            },
        },
    },
}


CLASSIFICATION_GBMAP = {
    "results_filename": "classification_gbmap",
    "model_config": {
        "gbmap": {
            "model": GBMAP4,
            "random_search_iters": 100,
            "params": {
                "n_boosts": range(2, 151),
                "softplus_scale": range(1, 21),
                "optim_maxiter": [200, 400],
                "penalty_weight": uniform(loc=0, scale=1e-2),
                "is_classifier": [True],
                "random_state": [RANDOM_SEED],
            },
        },
    },
}

FEATURE_CREATION_REG = {
    "results_filename": "features_reg",
    "embedding_dims": [2, 8, 12, 16, 32],
    "embedding_config": {
        "PCA": {"random_state": RANDOM_SEED},
        "GBMAP4": {
            "random_state": RANDOM_SEED,
            "penalty_weight": 1e-3,
            "optim_maxiter": 400,
            "optim_tol": 1e-3,
            "softplus_scale": 3,
        },
        "IVIS": {"supervision_metric": "mae", "n_epochs_without_progress": 5},
    },
    "model_config": {
        "linreg": {"model": LinearRegression, "params": {}},
    },
}


FEATURE_CREATION_CLS = {
    "results_filename": "features_cls",
    "embedding_dims": [2, 8, 12, 16, 32],
    "embedding_config": {
        "PCA": {"random_state": RANDOM_SEED},
        "GBMAP4": {
            "random_state": RANDOM_SEED,
            "penalty_weight": 1e-3,
            "optim_maxiter": 400,
            "optim_tol": 1e-3,
            "softplus_scale": 3,
            "is_classifier": True,
        },
        "LOL": {"svd_solver": "full", "random_state": RANDOM_SEED},
        "IVIS": {"supervision_metric": "mae", "n_epochs_without_progress": 5},
    },
    "model_config": {
        "logreg": {
            "model": LogisticRegression,
            "params": {"penalty": "l2", "C": 100, "random_state": RANDOM_SEED},
        },
    },
}
