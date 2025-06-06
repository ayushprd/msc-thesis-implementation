from common import (
    features_reg_datasets,
    features_cls_datasets,
    result_csv_save_path,
    cv_result_save_path,
)
from consts import RANDOM_SEED
from experiment_configs import FEATURE_CREATION_REG, FEATURE_CREATION_CLS

from gbmap.embedding.gbnn_four import GBMAP4

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold

import os
import argparse
import traceback

import numpy as np
import pandas as pd

import time

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
)

from sklearn.decomposition import PCA
from lol import LOL
from ivis import Ivis
import tensorflow as tf  # for IVIS


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)  # for IVIS

DOWNSAMPLE = 60_000


def random_search(model_class, model_params, X, y, max_iters, csv_filename=None):
    # downsample the data if it has too many datapoints
    if X.shape[0] > DOWNSAMPLE:
        idx = np.random.choice(X.shape[0], DOWNSAMPLE, replace=False)
        X_cv = X[idx, :]
        y_cv = y[idx]
    else:
        X_cv = X
        y_cv = y

    results = []
    scores = []
    param_sampler = ParameterSampler(
        model_params, n_iter=max_iters, random_state=RANDOM_SEED
    )
    print("Starting random search for {} iterations".format(len(param_sampler)))
    time_start = time.time()
    for i, params in enumerate(param_sampler):
        print("[Progress {}/{}]".format(i + 1, max_iters))
        try:
            # eval using stratified 5-fold cv
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            kf.get_n_splits(X_cv, y_cv)

            cv_scores = []
            for _, (train_index, test_index) in enumerate(kf.split(X_cv, y_cv)):
                # train split
                x_train = X_cv[train_index, :]
                y_train = y_cv[train_index]
                # test split
                x_test = X_cv[test_index, :]
                y_test = y_cv[test_index]
                # train
                model_cv = model_class(**params)
                model_cv.fit(x_train, y_train)
                # eval
                pred_test = model_cv.predict(x_test)
                cv_scores.append(mean_squared_error(y_test, pred_test))

            # store the mean cv score for the parameters
            scores.append(np.mean(cv_scores))
            # store parameters
            results.append(params)
        except Exception:
            # exception occurred, save the params and -1 score
            print("Exception while evaluating params: {}".format(params))
            traceback.print_exc()
            print("Skipping...")
            scores.append(1e-6)
            results.append(params)

    time_end = time.time()
    print("Finished random search, runtime: {:.1f} (s)".format(time_end - time_start))
    # find the best model
    best_idx = np.argmin(scores)
    best_params = results[best_idx]

    if csv_filename is not None:
        # create a DataFrame
        results_df = pd.DataFrame(results)
        results_df["mse"] = scores
        # save results
        save_path = cv_result_save_path(csv_filename)
        results_df.to_csv(save_path, index=False)

    return best_params


def embed_data(X_train, X_test, y_train, method, params, n_dims):
    start = time.time()
    if method == "GBMAP4":
        model = GBMAP4(n_boosts=n_dims, **params)
        model.fit(X_train, y_train)
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
    elif method == "PCA":
        model = PCA(n_components=n_dims, **params)
        model.fit(X_train, y_train)
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
    elif method == "LOL":
        model = LOL(n_components=n_dims, **params)
        model.fit(X_train, y_train)
        Z_train = model.transform(X_train)
        if Z_train.shape[1] == n_dims:
            Z_test = model.transform(X_test)
        else:
            print(
                "LOL produced d={} embedding when was asked to use {} components.".format(
                    Z_train.shape[1], n_dims
                )
            )
            print("Trying again with {} components...".format(n_dims + 1))
            model = LOL(n_components=n_dims + 1, **params)
            model.fit(X_train, y_train)
            Z_train = model.transform(X_train)
            if Z_train.shape[1] == n_dims:
                Z_test = model.transform(X_test)
            else:
                raise ValueError(
                    "LOL produced embedding with wrong dimensions, could not recover by asking k+1 embedding."
                )

    elif method == "IVIS":
        model = Ivis(embedding_dims=n_dims, **params)
        model.fit(X_train, y_train)
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
    elif method == "Original":
        # no embedding, for original data comparison
        Z_train = X_train
        Z_test = X_test
    else:
        raise ValueError("Got an unknown embedding methdod '{}'".format(method))

    embed_time = time.time() - start
    return Z_train, Z_test, embed_time


def fit_eval_model_reg(
    model_class,
    model_params,
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name,
    embedding_method,
    model_name,
    embedding_params,
    fold_idx,
    time,
):
    n_features = X_train.shape[1]
    model_params_str = ", ".join(f"{k}={v}" for k, v in model_params.items())
    embedding_params_str = ", ".join(f"{k}={v}" for k, v in embedding_params.items())
    try:
        # configure the supervised model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        train_mse = mean_squared_error(y_train, pred_train)
        train_r2 = r2_score(y_train, pred_train)
        test_mse = mean_squared_error(y_test, pred_test)
        test_r2 = r2_score(y_test, pred_test)

        results = {
            "dataset": dataset_name,
            "features": embedding_method,
            "data_dims": n_features,
            "model": model_name,
            "fold": fold_idx,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "embed_time": time,
            "embedding_params": embedding_params_str,
            "model_params": model_params_str,
        }
    except Exception:
        traceback.print_exc()
        print("Continuing...")

        results = {
            "dataset": dataset_name,
            "features": embedding_method,
            "data_dims": n_features,
            "model": model_name,
            "fold": fold_idx,
            "test_mse": 1e6,
            "test_r2": -1,
            "train_mse": 1e6,
            "train_r2": -1,
            "embed_time": 1e6,
            "embedding_params": embedding_params_str,
            "model_params": model_params_str,
        }
    return results


def fit_eval_model_cls(
    model_class,
    model_params,
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name,
    embedding_method,
    model_name,
    embedding_params,
    fold_idx,
    time,
):
    n_features = X_train.shape[1]  # Correctly calculate the number of features
    model_params_str = ", ".join(f"{k}={v}" for k, v in model_params.items())
    embedding_params_str = ", ".join(f"{k}={v}" for k, v in embedding_params.items())

    try:
        # configure the supervised model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        train_acc = accuracy_score(y_train, pred_train)
        train_pres = precision_score(y_train, pred_train)
        train_recall = recall_score(y_train, pred_train)
        test_acc = accuracy_score(y_test, pred_test)
        test_pres = precision_score(y_test, pred_test)
        test_recall = recall_score(y_test, pred_test)

        results = {
            "dataset": dataset_name,
            "features": embedding_method,
            "data_dims": n_features,
            "model": model_name,
            "fold": fold_idx,
            "train_acc": train_acc,
            "train_pres": train_pres,
            "train_recall": train_recall,
            "test_acc": test_acc,
            "test_pres": test_pres,
            "test_recall": test_recall,
            "embed_time": time,
            "embedding_params": embedding_params_str,
            "model_params": model_params_str,
        }
    except Exception:
        traceback.print_exc()
        print("Continuing...")
        results = {
            "dataset": dataset_name,
            "features": embedding_method,
            "data_dims": n_features,
            "model": model_name,
            "fold": fold_idx,
            "train_acc": -1,
            "train_pres": -1,
            "train_recall": -1,
            "test_acc": -1,
            "test_pres": -1,
            "test_recall": -1,
            "embed_time": 1e6,
            "embedding_params": embedding_params_str,
            "model_params": model_params_str,
        }
    return results


def evaluate_models(
    models_dict, embedding_params, datasets, embedding_dims, is_reg, save_path
):
    embedding_methods = embedding_params.keys()

    iter = 1
    n_experiments = (
        len(models_dict) * len(datasets) * len(embedding_methods) * len(embedding_dims)
    )

    for dataset_name in datasets.keys():
        X, y = datasets[dataset_name]["loader"](**datasets[dataset_name]["params"])

        for embedding_method in embedding_methods:
            embedding_paramsi = embedding_params.get(embedding_method, None)

            if embedding_method == "GBMAP4":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=1
                )
                params_dict = {
                    "n_boosts": [embedding_dims[-1]],
                    "softplus_scale": range(1, 21),
                    "optim_maxiter": [400],
                    "penalty_weight": [1e-3],
                    "random_state": [RANDOM_SEED],
                }
                embedding_paramsi = random_search(
                    model_class=GBMAP4,
                    model_params=params_dict,
                    X=X_train,
                    y=y_train,
                    max_iters=10,
                    csv_filename="reg_cv_{}_{}.csv".format(dataset_name, model_name),
                )
                # delete n_boosts (it will be defined later)
                embedding_paramsi.pop("n_boosts")

            for n_dims in embedding_dims:
                print(
                    "[Progress {i}/{n}]: Evaluating: dataset={d}, features={e}, dims={k}".format(
                        i=iter,
                        n=n_experiments,
                        d=dataset_name,
                        e=embedding_method,
                        k=n_dims,
                    )
                )

                for model_name in models_dict.keys():
                    model_class = models_dict[model_name]["model"]
                    model_params = models_dict[model_name]["params"]

                    for i in range(5):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.3, random_state=RANDOM_SEED + i
                        )

                        try:
                            Z_train, Z_test, embed_time = embed_data(
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                method=embedding_method,
                                params=embedding_paramsi,
                                n_dims=n_dims,
                            )
                        except Exception:
                            traceback.print_exc()
                            print("Skipping...")
                            break

                        if is_reg:
                            res = fit_eval_model_reg(
                                model_class=model_class,
                                model_params=model_params,
                                X_train=Z_train,
                                X_test=Z_test,
                                y_train=y_train,
                                y_test=y_test,
                                dataset_name=dataset_name,
                                embedding_method=embedding_method,
                                model_name=model_name,
                                embedding_params=embedding_paramsi,
                                fold_idx=i,
                                time=embed_time,
                            )
                        else:
                            res = fit_eval_model_cls(
                                model_class=model_class,
                                model_params=model_params,
                                X_train=Z_train,
                                X_test=Z_test,
                                y_train=y_train,
                                y_test=y_test,
                                dataset_name=dataset_name,
                                embedding_method=embedding_method,
                                model_name=model_name,
                                embedding_params=embedding_paramsi,
                                fold_idx=i,
                                time=embed_time,
                            )
                        df = pd.DataFrame([res])
                        print("####RESULTS####")
                        print(df)

                        df.to_csv(
                            save_path,
                            mode="a",
                            header=not os.path.exists(save_path),
                            index=False,
                        )
                        # results.append(res)

                    iter += 1


def main():
    help_txt = """Select the experiment config from ['reg', 'vers'],
    where 'reg' is used to compare different embedding methods producing regression features and
    'version' is used to compare gbmap versions against each other.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="reg", help=help_txt)
    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do a test run with only one embedding dimension.",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calculate baseline results (fit learner with original features).",
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Run the experiment on selected datasets, provide a list delimeted by space, e.g., 'data1 data2'",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--omit",
        help="Omit datasts by providing a list delimeted by space, e.g., 'data1 data2'",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--dims",
        help="Embedding dimensions separated by commas",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    # experiment params
    is_reg = False

    if args.config == "reg":
        config_dict = FEATURE_CREATION_REG
        is_reg = True
    elif args.config == "cls":
        config_dict = FEATURE_CREATION_CLS
    else:
        raise ValueError("Unknown experiment config {}".format(args.config))

    embedding_dims = config_dict["embedding_dims"]

    if args.dims is not None:
        dims_str = args.dims.split(",")
        embedding_dims = [int(m) for m in dims_str]

    print(embedding_dims)

    # load datasets
    if is_reg:
        datasets = features_reg_datasets(test=args.test)
    else:
        datasets = features_cls_datasets(test=args.test)

    if args.data is not None:
        selected = [dataset for dataset in args.data.split(" ")]
        selected_datasets = {}
        for key in datasets.keys():
            if key not in selected:
                continue
            selected_datasets[key] = datasets[key]
        datasets = selected_datasets

    if args.omit is not None:
        selected = [dataset for dataset in args.omit.split(" ")]
        selected_datasets = {}
        for key in datasets.keys():
            if key in selected:
                continue
            selected_datasets[key] = datasets[key]
        datasets = selected_datasets

    print(datasets.keys())

    embedding_params = config_dict["embedding_config"]
    models_dict = config_dict["model_config"]
    results_filename = config_dict["results_filename"]

    # get a path for results.csv file
    save_path = result_csv_save_path(csv_filename=results_filename)

    if args.baseline:
        # run the experiment on the original data
        evaluate_models(
            models_dict=models_dict,
            embedding_params={"Original": {}},
            datasets=datasets,
            embedding_dims=[None],
            is_reg=is_reg,
            save_path=save_path,
        )
    # evaluate embedding methods
    evaluate_models(
        models_dict=models_dict,
        embedding_params=embedding_params,
        datasets=datasets,
        embedding_dims=embedding_dims,
        is_reg=is_reg,
        save_path=save_path,
    )

    print("Results saved to {}".format(save_path))


if __name__ == "__main__":
    main()
