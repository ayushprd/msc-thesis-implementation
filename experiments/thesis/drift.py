from common import (
    regression_datasets_mini,
    safe_write_dir_path,
    classification_datasets_mini,
)
import pickle
import argparse
from gbmap.embedding.common import (
    F1max,
    knnpred,
    to_binary,
    caverage,
    find_variables,
    test_feature,
    split4,
    loss_logistic,
)
from common import result_csv_save_path, fig_result_save_path
from gbmap.embedding.gbnn_four import GBMAP4
from gbmap.data.datasets import make_synth_drift
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import traceback
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax.nn import softplus
from jax import device_put
from jax import random
import pandas as pd
import time
from consts import WRITE_DIR
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, DMatrix
from sklearn.ensemble import RandomForestRegressor

MODEL_RANDOM_SEED = 49


def eval_reg(X, y, data_name, n_boosts, key, knn=5, fontsize=12):
    print("Start running concept drift for regression datasets...")
    X = device_put(X)
    y = device_put(y)

    key1, key2 = random.split(key, 2)
    feature = find_variables(key1, X, y)

    n_irr = 0
    std_irr = 10.0
    ia1, ia2, ib1, ib2 = split4(key1, X, feature=feature)
    X1 = jnp.hstack(
        (
            std_irr * random.normal(key2, (X.shape[0], n_irr)),
            jnp.delete(X, (feature), axis=1),
        )
    )

    gbmap = GBMAP4(
        n_boosts=n_boosts,
        random_state=MODEL_RANDOM_SEED,
        optim_maxiter=200,
    )
    gbmap.fit(X1[ia1, :], y[ia1])
    yhat = gbmap.predict(X1)

    ib = jnp.concatenate((ib1, ib2))
    lv = (yhat - y) ** 2
    sigma = jnp.quantile(lv[ia2], 0.95).item()

    Z1 = gbmap.transform(X1)
    yhat_knn = knnpred(Z1, Z1[ia1, :], y[ia1], k=knn)
    drift = jnp.abs(yhat_knn - yhat)

    id = jnp.concatenate((ia2, ib1, ib2))
    fpr, tpr, _ = roc_curve(lv[id] > sigma, drift[id])
    auc = roc_auc_score(lv[id] > sigma, drift[id])
    max_f1, threshold = F1max(ind=drift[id], error=lv[id], thr_error=sigma)

    print(f"max F1 = {max_f1:.4f}  auc = {auc:.4f}")

    # For drawing
    min_x = jnp.min(drift[id])
    max_x = jnp.max(drift[id])
    middle_x = (min_x + max_x) / 2
    min_y = jnp.min(lv[id])
    max_y = jnp.max(lv[id])
    middle_y = (min_y + max_y) / 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.axhline(sigma)
    plt.axvline(threshold)
    plt.scatter(drift[ia2], lv[ia2], label="a2", alpha=0.5)
    plt.scatter(drift[ib], lv[ib], label="b", alpha=0.5, marker="x")
    plt.text(middle_x + 0.01, sigma + 0.01, f"{sigma:.1f}", fontsize=fontsize)
    plt.text(threshold + 0.01, middle_y + 0.01, f"{threshold:.1f}", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Drift indicator", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.title("Drift indicator vs true loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplot(1, 2, 2)
    plt.axline((0, 0), slope=1.0, color="lightgray")
    plt.plot(fpr, tpr)
    plt.text(0.5, 0.2, f"AUC = {auc:.3f}", fontsize=fontsize + 5)
    plt.xlabel("False positive rate", fontsize=fontsize)
    plt.ylabel("True positive rate", fontsize=fontsize)
    plt.title("ROC curve", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig_filename = data_name + "_gbmap.png"
    fig_path = fig_result_save_path(fig_filename)
    plt.savefig(fig_path, bbox_inches="tight")

    # Original distance indicator
    disto = jnp.sqrt(jnp.sum((X1[id, None, :] - X1[None, ia1, :]) ** 2, axis=2))
    kni = jnp.argsort(disto, axis=1)[:, knn]
    drifto = disto[jnp.arange(id.shape[0]), kni]
    fpro, tpro, _ = roc_curve(lv[id] > sigma, drifto[id])
    auco = roc_auc_score(lv[id] > sigma, drifto[id])
    max_f1o, thresholdo = F1max(ind=drifto[id], error=lv[id], thr_error=sigma)

    print(f"max F1 = {max_f1o:.4f}  auc = {auco:.4f}")

    min_xo = jnp.min(drifto[id])
    max_xo = jnp.max(drifto[id])
    middle_xo = (min_xo + max_xo) / 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(drifto[ia2], lv[ia2], label="a2", alpha=0.5)
    plt.scatter(drifto[ib], lv[ib], label="b", alpha=0.5, marker="x")
    plt.text(middle_xo + 0.01, sigma + 0.01, f"{sigma:.1f}", fontsize=fontsize)
    plt.text(thresholdo + 0.01, middle_y + 0.01, f"{thresholdo:.1f}", fontsize=fontsize)
    plt.axhline(sigma)
    plt.axvline(thresholdo)
    plt.legend(fontsize=fontsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Drift indicator", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.title("Drift indicator vs true loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplot(1, 2, 2)
    plt.plot(fpro, tpro)
    plt.text(0.5, 0.2, f"AUC = {auco:.3f}", fontsize=fontsize + 5)
    plt.axline((0, 0), slope=1.0, color="lightgray")
    plt.xlabel("False positive rate", fontsize=fontsize)
    plt.ylabel("True positive rate", fontsize=fontsize)
    plt.title("ROC curve", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig_filename = data_name + "_euclidean.png"
    fig_path = fig_result_save_path(fig_filename)
    plt.savefig(fig_path, bbox_inches="tight")

    # pickle
    pkl_filename = data_name + ".pkl"
    pkl_path = fig_result_save_path(pkl_filename)
    data = {
        "drift": drift,
        "drifto": drifto,
        "lv": lv,
        "ia1": ia1,
        "ia2": ia2,
        "ib": ib,
    }
    with open(pkl_path, "wb") as file:
        pickle.dump(data, file)

    return {
        "data-name": data_name,
        "max-f1": max_f1,
        "auc": auc,
        "max-f1o": max_f1o,
        "auco": auco,
    }


def eval_cls(X, y, data_name, n_boosts, key, knn=5, fontsize=12):
    print("Start running concept drift for classification datasets...")
    X = device_put(X)
    y = device_put(y)

    key1, key2 = random.split(key, 2)
    feature = find_variables(key1, X, y, loss=loss_logistic)

    n_irr = 0
    std_irr = 10.0
    ia1, ia2, ib1, ib2 = split4(key1, X, feature=feature)
    X1 = jnp.hstack(
        (
            std_irr * random.normal(key2, (X.shape[0], n_irr)),
            jnp.delete(X, (feature), axis=1),
        )
    )

    gbmap = GBMAP4(
        n_boosts=n_boosts,
        random_state=MODEL_RANDOM_SEED,
        optim_maxiter=200,
        is_classifier=True,
    )
    gbmap.fit(X1[ia1, :], y[ia1])
    yhat = gbmap.predict(X1, get_score=True)

    ib = jnp.concatenate((ib1, ib2))
    # lv = (yhat - y) ** 2
    lv = softplus(-yhat * y)
    sigma = jnp.quantile(lv[ia2], 0.95).item()

    Z1 = gbmap.transform(X1)
    # yhat_knn = knnpred(Z1, Z1[ia1, :], y[ia1], k=knn, average=caverage)
    yhat_knn = knnpred(Z1, Z1[ia1, :], yhat[ia1], k=knn)
    drift = jnp.abs(yhat_knn - yhat)

    id = jnp.concatenate((ia2, ib1, ib2))
    fpr, tpr, _ = roc_curve(lv[id] > sigma, drift[id])
    auc = roc_auc_score(lv[id] > sigma, drift[id])
    max_f1, threshold = F1max(ind=drift[id], error=lv[id], thr_error=sigma)

    print(f"max F1 = {max_f1:.4f}  auc = {auc:.4f}")

    # for drawing

    min_x = jnp.min(drift[id])
    max_x = jnp.max(drift[id])
    middle_x = (min_x + max_x) / 2
    min_y = jnp.min(lv[id])
    max_y = jnp.max(lv[id])
    middle_y = (min_y + max_y) / 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.axhline(sigma)
    plt.axvline(threshold)
    plt.scatter(drift[ia2], lv[ia2], label="a2", alpha=0.5)
    plt.scatter(drift[ib], lv[ib], label="b", alpha=0.5, marker="x")
    plt.text(middle_x, sigma + 0.01, f"{sigma:.1f}", fontsize=fontsize)
    plt.text(threshold + 0.01, middle_y, f"{threshold:.1f}", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Drift indicator", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.title("Drift indicator vs true loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplot(1, 2, 2)
    plt.axline((0, 0), slope=1.0, color="lightgray")
    plt.plot(fpr, tpr)
    plt.text(0.5, 0.2, f"AUC = {auc:.3f}", fontsize=fontsize + 3)
    plt.xlabel("False positive rate", fontsize=fontsize)
    plt.ylabel("True positive rate", fontsize=fontsize)
    plt.title("ROC curve", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig_filename = data_name + "_gbmap_cls.png"
    fig_path = fig_result_save_path(fig_filename)
    plt.savefig(fig_path, bbox_inches="tight")

    # Original distance indicator
    disto = jnp.sqrt(jnp.sum((X1[id, None, :] - X1[None, ia1, :]) ** 2, axis=2))
    kni = jnp.argsort(disto, axis=1)[:, knn]
    drifto = disto[jnp.arange(id.shape[0]), kni]
    fpro, tpro, _ = roc_curve(lv[id] > sigma, drifto[id])
    auco = roc_auc_score(lv[id] > sigma, drifto[id])
    max_f1o, thresholdo = F1max(ind=drifto[id], error=lv[id], thr_error=sigma)

    print(f"max F1 = {max_f1o:.4f}  auc = {auco:.4f}")

    min_xo = jnp.min(drifto[id])
    max_xo = jnp.max(drifto[id])
    middle_xo = (min_xo + max_xo) / 2

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(drifto[ia2], lv[ia2], label="a2", alpha=0.5)
    plt.scatter(drifto[ib], lv[ib], label="b", alpha=0.5, marker="x")
    plt.text(middle_xo, sigma + 0.01, f"{sigma:.1f}", fontsize=fontsize)
    plt.text(thresholdo + 0.01, middle_y, f"{thresholdo:.1f}", fontsize=fontsize)
    plt.axhline(sigma)
    plt.axvline(thresholdo)
    plt.legend(fontsize=fontsize)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Drift indicator", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.title("Drift indicator vs true loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.subplot(1, 2, 2)
    plt.plot(fpro, tpro)
    plt.text(0.5, 0.2, f"AUC = {auco:.3f}", fontsize=fontsize + 3)
    plt.axline((0, 0), slope=1.0, color="lightgray")
    plt.xlabel("False positive rate", fontsize=fontsize)
    plt.ylabel("True positive rate", fontsize=fontsize)
    plt.title("ROC curve", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig_filename = data_name + "_euclidean_cls.png"
    fig_path = fig_result_save_path(fig_filename)
    plt.savefig(fig_path, bbox_inches="tight")

    # pickle
    pkl_filename = data_name + ".pkl"
    pkl_path = fig_result_save_path(pkl_filename)
    data = {
        "drift": drift,
        "drifto": drifto,
        "lv": lv,
        "ia1": ia1,
        "ia2": ia2,
        "ib": ib,
    }
    with open(pkl_path, "wb") as file:
        pickle.dump(data, file)

    return {
        "data-name": data_name,
        "max-f1": max_f1,
        "auc": auc,
        "max-f1o": max_f1o,
        "auco": auco,
    }


def main():
    parser = argparse.ArgumentParser()
    help_text = "Select the experiment config from ['reg', 'cls']"
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="reg",
        help=help_text,
    )
    args = parser.parse_args()
    if args.config == "reg":
        classification = False
    elif args.config == "cls":
        classification = True
    else:
        raise ValueError("Unknown experiment config {}".format(args.config))

    fontsize = 16
    res_table = []
    if classification:
        datasets = classification_datasets_mini()
    else:
        datasets = regression_datasets_mini()
    key = random.PRNGKey(42)
    n_boosts = 20

    for dataset_name in datasets.keys():
        X, y = datasets[dataset_name]["loader"](**datasets[dataset_name]["params"])
        if classification:
            res = eval_cls(X, y, dataset_name, n_boosts, key, knn=5, fontsize=fontsize)
        else:
            res = eval_reg(X, y, dataset_name, n_boosts, key, knn=5, fontsize=fontsize)
        res_table.append(res)

    results_df = pd.DataFrame(res_table)
    print(results_df)
    # get a path for results.csv file
    save_path = result_csv_save_path(csv_filename="drift")
    # save results to csv
    results_df.to_csv(save_path, index=False)
    print("Results saved to {}".format(save_path))


def stats(result_path):
    res = pd.read_csv(result_path)
    res = res.loc[:, ["data-name", "auc", "auco"]]
    res.rename(
        columns={"auc": "gbmap", "auco": "euclid", "data-name": "dataset"}, inplace=True
    )
    print(res.to_latex(float_format=f"%.{2}f", index=False))


if __name__ == "__main__":
    main()
    # file_name = "drift_2024-01-30-1556.csv"
    # results_dir = safe_write_dir_path(WRITE_DIR)
    # results_path = os.path.join(results_dir, file_name)
    # stats(results_path)
