from common import classification_datasets, result_csv_save_path, cv_result_save_path
from consts import RANDOM_SEED
from experiment_configs import (
    CLASSIFICATION_FULL,
    CLASSIFICATION_GBMAP,
    CLASSIFICATION_KNN,
)

import time
import argparse
import traceback
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import StratifiedKFold

# from sklearn.model_selection import RepeatedStratifiedKFold


np.random.seed(RANDOM_SEED)

DOWNSAMPLE = 60_000


def random_search(model, model_params, X, y, max_iters, csv_filename=None):
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
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
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
                model_cv = model(**params)
                model_cv.fit(x_train, y_train)
                # eval
                pred_test = model_cv.predict(x_test)
                cv_scores.append(accuracy_score(y_test, pred_test))

            # store the mean cv score for the parameters
            scores.append(np.mean(cv_scores))
            # store parameters
            results.append(params)
        except Exception:
            # exception occurred, save the params and -1 score
            print("Exception while evaluating params: {}".format(params))
            traceback.print_exc()
            print("Skipping...")
            scores.append(-1)
            results.append(params)

    time_end = time.time()
    print("Finished random search, runtime: {:.1f} (s)".format(time_end - time_start))

    # find the best model
    best_idx = np.argmax(scores)
    best_params = results[best_idx]

    if csv_filename is not None:
        # create a DataFrame
        results_df = pd.DataFrame(results)
        results_df["accuracy"] = scores
        # save results
        save_path = cv_result_save_path(csv_filename)
        results_df.to_csv(save_path, index=False)

    return best_params


def train_eval_results(
    model, params, X_train, X_test, y_train, y_test, dataset_name, model_name
):
    # for params in param_sets:
    model = model(**params)
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())

    print("Evaluating model {} with params {}".format(model_name, params_str))
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        train_pres = precision_score(y_train, y_pred_train)
        train_recall = recall_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_pres = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        results = {
            "dataset": dataset_name,
            "model": model_name,
            "test_acc": test_acc,
            "test_pres": test_pres,
            "test_recall": test_recall,
            "train_acc": train_acc,
            "train_pres": train_pres,
            "train_recall": train_recall,
            "train_time": training_time,
            "params": params_str,
        }

    except Exception:
        traceback.print_exc()
        print("Continuing...")
        results = {
            "dataset": dataset_name,
            "model": model_name,
            "test_acc": -1,
            "test_pres": -1,
            "test_recall": -1,
            "train_acc": -1,
            "train_pres": -1,
            "train_recall": -1,
            "train_time": 1e6,
            "params": params_str,
        }
    return results


def transform_labels(y):
    y_new = y.copy()
    y_new[y_new == -1] = 0
    return y_new


def evaluate_models(models_dict, datasets, test_size, save_path, downsample):
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
                "[Progress {}/{}]: Dataset {}: Evaluating {})".format(
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
            X_train, X_test, y_train_, y_test_ = train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=True,
                stratify=y,
                random_state=RANDOM_SEED,
            )

            if model_name == "xgboost":
                # transform the labels from -1, 1 to 0, 1
                y_train = transform_labels(y_train_)
                y_test = transform_labels(y_test_)
            else:
                y_train = y_train_
                y_test = y_test_

            if random_search_iters > 0:
                # many params do a random search with CV to pick as suitable one
                params_eval = random_search(
                    model=model_class,
                    model_params=params_dict,
                    X=X_train,
                    y=y_train,
                    max_iters=random_search_iters,
                    csv_filename="cls_cv_{}_{}.csv".format(dataset_name, model_name),
                )
            else:
                params_eval = params_dict

            res = train_eval_results(
                model=model_class,
                params=params_eval,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                dataset_name=dataset_name,
                model_name=model_name,
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
            iter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="cls",
        help="Select the experiment config from ['cls', 'gbmap', 'knn']",
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

    if args.config == "cls":
        config_dict = CLASSIFICATION_FULL
    elif args.config == "gbmap":
        config_dict = CLASSIFICATION_GBMAP
    elif args.config == "knn":
        config_dict = CLASSIFICATION_KNN
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
    datasets = classification_datasets(test=args.test)

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
