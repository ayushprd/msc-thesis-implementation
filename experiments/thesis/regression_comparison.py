from common import (
    regression_datasets,
    regression_knn_datasets,
    result_csv_save_path,
    cv_result_save_path,
)
from consts import RANDOM_SEED
from experiment_configs import REGRESSION_FULL, REGRESSION_GBMAP, REGRESSION_KNN

import time
import argparse
import traceback
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold


np.random.seed(RANDOM_SEED)

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


def train_eval_results(
    model_class, params, X_train, X_test, y_train, y_test, dataset_name, model_name
):
    # for params in param_sets:
    model = model_class(**params)
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())

    print("Evaluating model {} with params {}".format(model_name, params_str))
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_mse = mean_squared_error(y_train, pred_train)
        train_r2 = r2_score(y_train, pred_train)
        test_mse = mean_squared_error(y_test, pred_test)
        test_r2 = r2_score(y_test, pred_test)
        results = {
            "dataset": dataset_name,
            "model": model_name,
            "test_mse": test_mse,
            "test_r2": test_r2,
            "train_mse": train_mse,
            "train_r2": train_r2,
            "train_time": training_time,
            "params": params_str,
        }
    except Exception:
        traceback.print_exc()

        print("Continuing...")
        results = {
            "dataset": dataset_name,
            "model": model_name,
            "test_mse": 1e6,
            "test_r2": -1,
            "train_mse": 1e6,
            "train_r2": -1,
            "train_time": 1e6,
            "params": params_str,
        }
    return results


def evaluate_models(models_dict, datasets, test_size, save_path, downsample=None):
    """
    Evaluate multiple models on multiple datasets with multiple sets of parameters.

    Args:
        model_classes (list): List of model classes to evaluate.
        model_init_params (dict): Dictionary with model classes as keys and lists of initialization params as values.
        datasets (list): List of datasets to evaluate the models on.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        pandas.DataFrame: DataFrame containing the evaluation results for each model and dataset.
    """

    iter = 1
    n_experiments = len(datasets) * len(models_dict)

    for dataset_name in datasets.keys():
        try:
            # extract the datapoint
            X, y = datasets[dataset_name]["loader"](**datasets[dataset_name]["params"])

        except Exception:
            traceback.print_exc()
            print("Could not load dataset {}, skipping...".format(dataset_name))
            continue

        for model_name in models_dict.keys():
            model_class = models_dict[model_name]["model"]
            params_dict = models_dict[model_name]["params"]
            random_search_iters = models_dict[model_name]["random_search_iters"]

            print(
                "[Progress {}/{}]: Dataset: {} Evaluating {})".format(
                    iter, n_experiments, dataset_name, model_name
                )
            )

            if downsample is not None:
                if X.shape[0] > downsample:
                    idx = np.random.choice(X.shape[0], downsample, replace=False)
                    X = X[idx, :]
                    y = y[idx]

                print("dataset size:", X.shape)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=RANDOM_SEED
            )

            if random_search_iters > 0:
                # many params do a random search with CV to pick as suitable one
                params_eval = random_search(
                    model_class=model_class,
                    model_params=params_dict,
                    X=X_train,
                    y=y_train,
                    max_iters=random_search_iters,
                    csv_filename="reg_cv_{}_{}.csv".format(dataset_name, model_name),
                )
            else:
                # no parameters to select, evaluate
                params_eval = params_dict

            res = train_eval_results(
                model_class=model_class,
                params=params_eval,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            # results.append(res)
            df = pd.DataFrame([res])
            print("####RESULTS####")
            print(df)

            df.to_csv(
                save_path,
                mode="a",
                header=not os.path.exists(save_path),
                index=False,
            )
            iter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="reg",
        help="Select the experiment config from ['reg', 'gbmap', 'knn']",
    )
    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do a test run with only one random search iteration.",
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
        "-i",
        "--iters",
        help="Override random search iterations.",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    if args.omit is not None and args.data is not None:
        raise ValueError("Both --omit and --data cannot be selected at the same time.")

    test_size = 0.3
    downsample = None

    if args.config == "reg":
        config_dict = REGRESSION_FULL
    elif args.config == "gbmap":
        config_dict = REGRESSION_GBMAP
    elif args.config == "knn":
        config_dict = REGRESSION_KNN
        downsample = 10_000
    else:
        raise ValueError("Unknown experiment config {}".format(args.config))

    models_dict = config_dict["model_config"]

    if args.test == True:
        # if test is set
        print("Running a test with 1 random search iterations.")
        for model_name in models_dict.keys():
            models_dict[model_name]["random_search_iters"] = 1

    if args.iters is not None:
        print(
            "Running the experiment with {} random search iterations.".format(
                args.iters
            )
        )
        for model_name in models_dict.keys():
            models_dict[model_name]["random_search_iters"] = args.iters

    # dataset dict
    if args.config == "knn":
        datasets = regression_knn_datasets(test=args.test)
    else:
        datasets = regression_datasets(test=args.test)

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

    # get a path for results.csv file
    save_path = result_csv_save_path(csv_filename=config_dict["results_filename"])

    evaluate_models(
        models_dict=models_dict,
        datasets=datasets,
        test_size=test_size,
        save_path=save_path,
        downsample=downsample,
    )

    print("Results saved to {}".format(save_path))


if __name__ == "__main__":
    main()
