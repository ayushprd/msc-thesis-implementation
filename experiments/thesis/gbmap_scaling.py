from gbmap.embedding.gbnn_four import GBMAP4
from gbmap.data.datasets import make_synth
import common
from consts import RANDOM_SEED
from sklearn.decomposition import PCA
from lol import LOL
import jax.numpy as jnp
from jaxopt import LBFGS
from ivis import Ivis
import tensorflow as tf
from jax import random, jit
import time
import os
import traceback
from jax.nn import softplus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from stopit import ThreadingTimeout, TimeoutException

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

key = random.PRNGKey(42)


def loss_quadratic(y, yp):
    return jnp.average((y - yp) ** 2)


def init_network_params0(p, b):
    w = jnp.zeros(p)
    a = jnp.zeros(1)
    b = jnp.array(b)
    return (a, b, w)


# Forward pass of one weak learner
def predict1(params, x, scale=5.0):
    a, b, w = params
    return a + b * softplus(scale * jnp.dot(x, w.T)) / scale


def init_network_params1(p, b, key):
    a_key, w_key = random.split(key, 2)
    # First layer weights and biases
    w = random.normal(w_key, (p,))
    a = random.normal(a_key)
    b = jnp.array(b)
    return (a, b, w)


# One boosting iteration: learn weak learner
def learn1(x, y, y0, params0, loss=loss_quadratic, ridge=1e-3):
    a, b, w = params0
    par0 = a, w

    def objective_fn(par):
        a, w = par
        params = a, b, w
        return loss(y, y0 + predict1(params, x)) + ridge * jnp.average(w**2)

    lbfgs = LBFGS(fun=objective_fn, maxiter=500)
    opt_result = lbfgs.run(init_params=par0)
    # Extract optimized parameters
    a, w = opt_result.params
    return (a, b, w)


# m boosting iterations
def learn(
    m,
    x,
    y,
    key,
    y0=None,
    loss=loss_quadratic,
    ridge=1e-3,
    randomise=True,
    precompile=False,
):
    if y0 is None:
        y0 = jnp.zeros(y.shape)

    keys = random.split(key, m)

    @jit
    def learnf(y0, b, key):
        params0 = (
            init_network_params1(x.shape[1], b, key)
            if randomise
            else init_network_params0(x.shape[1], b)
        )
        params = learn1(x, y, y0, params0, loss, ridge)
        predict = predict1(params, x)
        lossv = loss(y, y0 + predict)
        return params, lossv

    if precompile:
        learnf(y0, 1.0, keys[0])

    params = [None] * m
    losses = [None] * m

    start_time = time.time()
    for j in range(m):
        params_plus, loss_plus = learnf(y0, 1.0, keys[j])
        params_minus, loss_minus = learnf(y0, -1.0, keys[j])
        params[j] = params_minus if loss_minus < loss_plus else params_plus
        losses[j] = (loss_minus if loss_minus < loss_plus else loss_plus).item()
        y0 = y0 + predict1(params[j], x)
    time_duration = time.time() - start_time
    # print(f"duration = {time_duration:.3f}")

    return params, jnp.array(losses), time_duration


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


results_dir = os.path.join(common.script_abs_path(), "results", "gbmap_scaling")
create_dir(results_dir)


# Make predictions of a dummy regression model
def dummy_regression(ytrain, ytest):
    return jnp.full(ytest.shape, jnp.average(ytrain))


# L1 embedding (without abs)
def embed(params, x):
    return jnp.array([predict1(par, x) for par in params]).T


def predict(params, x, y0=0.0):
    # f = y0
    # for par in params:
    #    f = f + predict1(par, x)
    return y0 + jnp.sum(embed(params, x), axis=1)


def embed_data(X, y, method, params, n_dims):
    X_dr = None
    dr_time = np.nan
    dummy_loss = np.nan
    gbmap_loss = np.nan
    dummy_loss = 0
    gbmap_loss = 0
    try:
        with ThreadingTimeout(600) as to_ctx:
            if method == "GBMAP":
                params, loss, time_duration = learn(n_dims, X, y, key, precompile=True)
                dr_time = time_duration
                gbmap_loss = loss_quadratic(y, predict(params, X))
                dummy_loss = loss_quadratic(y, dummy_regression(y, y))
            elif method == "PCA":
                start_time = time.time()
                dr_model = PCA(n_components=n_dims, **params)
                X_dr = dr_model.fit_transform(X)
                dr_time = time.time() - start_time
            elif method == "LOL":
                start_time = time.time()
                dr_model = LOL(n_components=n_dims, **params)
                dr_model.fit(X, y)
                X_dr = dr_model.transform(X)
                dr_time = time.time() - start_time
            elif method == "IVIS":
                start_time = time.time()
                dr_model = Ivis(embedding_dims=n_dims, **params)
                dr_model.fit(X, y)
                X_dr = dr_model.transform(X)
                dr_time = time.time() - start_time
            else:
                raise ValueError("Got an unknown embedding method '{}'".format(method))
    except Exception as e:
        if isinstance(e, TimeoutException):
            print(f"Timeout occurred for {method}")
        else:
            print(f"Error in embedding data using {method}: {e}")
            traceback.print_exc()

    if to_ctx.state == to_ctx.TIMED_OUT:
        print(f"time out for {method}")

    return X_dr, dr_time, dummy_loss, gbmap_loss


def measure_embedding_time(
    dataset_size, data_dim, embedding_dim, method, classification=True, repeats=1
):
    times = []
    dummy_losses = []
    gbmap_lossess = []
    skip_remaining_repeats = (
        False  # Flag to skip the remaining repeats for the current configuration
    )
    skip_same_model_higher_size_dims = (
        False  # Flag to skip executing the same model with higher size or dims
    )

    for i in range(repeats):
        if skip_remaining_repeats or skip_same_model_higher_size_dims:
            break

        X, y = make_synth(dataset_size, data_dim, classification=classification)
        params = {}
        if method == "GBMAP":
            params.update(
                {
                    "penalty_weight": 1e-5,
                    "is_classifier": classification,
                    "random_state": RANDOM_SEED + i,
                }
            )
        elif method == "PCA":
            pass
        elif method == "TSNE":
            params.update(
                {"method": "barnes_hut", "random_state": RANDOM_SEED + i, "n_jobs": -1}
            )
        elif method == "LOL":
            params.update({"svd_solver": "full", "random_state": RANDOM_SEED + i})
        elif method == "IVIS":
            params.update(
                {
                    "supervision_metric": "mae",
                    "n_epochs_without_progress": 5,
                    "verbose": 0,
                }
            )
        elif method == "SPCA":
            pass

        try:
            _, dr_time, dummy_loss, gbmap_loss = embed_data(
                X, y, method, params, embedding_dim
            )
        except Exception as e:
            if isinstance(e, TimeoutException):
                print(f"Timeout occurred for {method}")
                skip_remaining_repeats = True  # Set the flag to skip the remaining repeats for this configuration
                break
            else:
                print(f"Error in embedding data using {method}: {e}")
                traceback.print_exc()
                continue  # Continue with the next repeat

        times.append(dr_time)
        dummy_losses.append(dummy_loss)
        gbmap_lossess.append(gbmap_loss)

    mean_time = np.mean(times)
    mean_dummy_loss = np.mean(dummy_losses)
    mean_gbmap_loss = np.mean(gbmap_lossess)
    print(
        "Finished embedding {} size={} dims={}: time:{} dummy_loss:{} gbmap_loss: {} ".format(
            method, dataset_size, data_dim, mean_time, mean_dummy_loss, mean_gbmap_loss
        )
    )
    return mean_time, mean_dummy_loss, mean_gbmap_loss


def scaling_data_size(results_dir, classification=True, method="GBMAP3"):
    task = "regression" if not classification else "classification"
    print(f"Running varying dataset size experiment for {task} using {method}")
    dims = 25
    data_points = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    times_dataset_size = []
    times_dummy_loss = []
    times_gbmap_loss = []
    for size in data_points:
        times, dummy_loss, gbmap_loss = measure_embedding_time(
            dataset_size=size,
            data_dim=dims,
            embedding_dim=2,
            method=method,
            classification=classification,
            repeats=1,
        )
        times_dataset_size.append(times)
        times_dummy_loss.append(dummy_loss)
        times_gbmap_loss.append(gbmap_loss)

    return {
        "size": data_points,
        "dim": [dims for _ in data_points],
        "time": times_dataset_size,
        "dummy_loss": times_dummy_loss,
        "gbmap_loss": times_gbmap_loss,
    }


def scaling_dims(results_dir, classification=True, method="GBMAP3"):
    task = "regression" if not classification else "classification"
    print(f"Running varying data dimension experiment for {task} using {method}")
    size = 10_000
    features = [100, 200, 400, 800, 1600]
    times_dataset_size = []
    times_dummy_loss = []
    times_gbmap_loss = []
    for dims in features:
        times, dummy_loss, gbmap_loss = measure_embedding_time(
            dataset_size=size,
            data_dim=dims,
            embedding_dim=2,
            method=method,
            classification=classification,
            repeats=1,
        )
        times_dataset_size.append(times)
        times_dummy_loss.append(dummy_loss)
        times_gbmap_loss.append(gbmap_loss)

    return {
        "dim": features,
        "size": [size for _ in features],
        "time": times_dataset_size,
        "dummy_loss": times_dummy_loss,
        "gbmap_loss": times_gbmap_loss,
    }


def collect_results(methods, scaling_function, classification, results_dir, type_str):
    result_list = []
    for method in methods:
        result = scaling_function(
            results_dir=results_dir, classification=classification, method=method
        )

        for i in range(len(result["time"])):
            time_result = result["time"][i]
            if np.isnan(time_result):
                time_result = "timeout"

            if type_str == "size":
                row = {
                    "model": method,
                    "size": result["size"][i],
                    "dim": result["dim"][i],
                    "time": time_result,
                    "dummy_loss": result["dummy_loss"][i],
                    "gbmap_loss": result["gbmap_loss"][i],
                }
            else:
                row = {
                    "model": method,
                    "size": result["size"][i],
                    "dim": result["dim"][i],
                    "time": time_result,
                    "dummy_loss": result["dummy_loss"][i],
                    "gbmap_loss": result["gbmap_loss"][i],
                }
            result_list.append(row)

    df = pd.DataFrame(result_list)
    df.to_csv(os.path.join(results_dir, f"scaling_{type_str}.csv"), index=False)
    return df


if __name__ == "__main__":
    start_time = time.time()
    methods = ["PCA", "IVIS", "GBMAP", "LOL"]
    df_size = collect_results(methods, scaling_data_size, True, results_dir, "size")
    df_dims = collect_results(methods, scaling_dims, True, results_dir, "dims")
    df_size = pd.read_csv(os.path.join(results_dir, "scaling_size.csv"))
    df_dim = pd.read_csv(os.path.join(results_dir, "scaling_dims.csv"))

    linestyle = [
        ("-", None, "black"),
        (":", None, "black"),
        ("--", None, "black"),
        ("-", None, "blue"),
        ("-", None, "red"),
        ("--", None, "grey"),
    ]
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
    for model, style in zip(df_size["model"].unique(), cycle(linestyle)):
        model_df = df_size[df_size["model"] == model]
        axs[0].plot(
            model_df["size"],
            model_df["time"],
            label=model,
            linestyle=style[0],
            color=style[2],
        )
    axs[0].set_xlabel("Dataset Size")
    axs[0].set_ylabel("Time (s)")
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    for model, style in zip(df_dim["model"].unique(), cycle(linestyle)):
        model_df = df_dim[df_dim["model"] == model]
        axs[1].plot(
            model_df["dim"],
            model_df["time"],
            label=model,
            linestyle=style[0],
            color=style[2],
        )
    axs[1].set_xlabel("Data Dimension")
    axs[1].set_ylabel("Time (s)")
    axs[1].legend()
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "size_dims.pdf"))
    print("total time: {}".format(time.time() - start_time))
