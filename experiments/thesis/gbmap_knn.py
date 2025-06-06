from gbmap.embedding.common import knnpred, caverage
from gbmap.embedding.gbnn_four import GBMAP4

import numpy as np
from jax import numpy as jnp

# SAMPLING_SIZE = 10_000


class KNNReg:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = 10

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Predict using k-NN

        # if self.X_train.shape[0] > SAMPLING_SIZE:
        #    self.X_train = self.X_train[:SAMPLING_SIZE, :]
        #    self.y_train = self.y_train[:SAMPLING_SIZE]

        y_pred = knnpred(
            jnp.array(X_test),
            jnp.array(self.X_train),
            jnp.array(self.y_train),
            k=self.k,
        )
        return y_pred


class KNNCls:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = 10

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Predict using k-NN

        # if self.X_train.shape[0] > SAMPLING_SIZE:
        #    self.X_train = self.X_train[:SAMPLING_SIZE, :]
        #    self.y_train = self.y_train[:SAMPLING_SIZE]

        y_pred = knnpred(
            jnp.array(X_test),
            jnp.array(self.X_train),
            jnp.array(self.y_train),
            average=caverage,
            k=self.k,
        )
        y_pred = jnp.where(y_pred > 0, 1, -1)
        return y_pred


class KNNWithGBMAPReg:
    def __init__(
        self,
        n_boosts=10,
        softplus_scale=1,
        optim_maxiter=100,
        penalty_weight=1e-3,
        random_state=None,
    ):
        self.gbmap = GBMAP4(
            n_boosts=n_boosts,
            softplus_scale=softplus_scale,
            optim_maxiter=optim_maxiter,
            penalty_weight=penalty_weight,
            random_state=random_state,
        )
        self.X_train = None
        self.y_train = None
        self.k = 10

    def fit(self, X_train, y_train):
        self.gbmap.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet")

        X_train_dist_gbmap = self.gbmap.transform(self.X_train)
        X_test_dist_gbmap = self.gbmap.transform(X_test)

        # if X_train_dist_gbmap.shape[0] > SAMPLING_SIZE:
        #    X_train_dist_gbmap = X_train_dist_gbmap[:SAMPLING_SIZE, :]
        #    self.y_train = self.y_train[:SAMPLING_SIZE]

        # Predict using k-NN
        y_pred = knnpred(
            jnp.array(X_test_dist_gbmap),
            jnp.array(X_train_dist_gbmap),
            jnp.array(self.y_train),
            k=self.k,
        )
        return y_pred


class KNNWithGBMAPCls:
    def __init__(
        self,
        n_boosts=10,
        softplus_scale=1,
        optim_maxiter=100,
        penalty_weight=1e-3,
        random_state=None,
    ):
        self.gbmap = GBMAP4(
            n_boosts=n_boosts,
            softplus_scale=softplus_scale,
            optim_maxiter=optim_maxiter,
            penalty_weight=penalty_weight,
            random_state=random_state,
            is_classifier=True,
        )
        self.X_train = None
        self.y_train = None
        self.k = 10

    def fit(self, X_train, y_train):
        self.gbmap.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been trained yet")

        X_train_dist_gbmap = self.gbmap.transform(self.X_train)
        X_test_dist_gbmap = self.gbmap.transform(X_test)

        # if X_train_dist_gbmap.shape[0] > SAMPLING_SIZE:
        #    X_train_dist_gbmap = X_train_dist_gbmap[:SAMPLING_SIZE, :]
        #    self.y_train = self.y_train[:SAMPLING_SIZE]

        # Predict using k-NN
        y_pred = knnpred(
            jnp.array(X_test_dist_gbmap),
            jnp.array(X_train_dist_gbmap),
            jnp.array(self.y_train),
            average=caverage,
            k=self.k,
        )
        y_pred = jnp.where(y_pred > 0, 1, -1)
        return y_pred
