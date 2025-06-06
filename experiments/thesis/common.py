from consts import WRITE_DIR, CV_WRITE_DIR, FIGURE_WRITE_DIR
import os
import datetime
from pathlib import Path

from gbmap.data.datasets import (
    auto_mpg,
    california_housing,
    wine_quality,
    wines_classification,
    mnist,
    breast_cancer,
    diabetes,
    abalone,
    german_credit,
    concrete,
    cpu_small,
    higgs,
    airquality,
    eeg_eye_state,
    make_synth,
    superconductor,
    qm9,
)


def script_abs_path():
    # get the script absolute path
    return os.path.dirname(os.path.abspath(__file__))


def safe_write_dir_path(write_dir):
    # path to the results dir
    write_path = os.path.join(script_abs_path(), write_dir)
    # create a dir if it does not exist yet
    Path(write_path).mkdir(exist_ok=True)
    return write_path


def cv_result_save_path(csv_filename):
    # form a path to the cv results dir
    results_dir = safe_write_dir_path(CV_WRITE_DIR)
    # form a path for the csv file
    save_path = os.path.join(results_dir, csv_filename)
    return save_path


def fig_result_save_path(fig_filename):
    # form a path to the cv results dir
    results_dir = safe_write_dir_path(FIGURE_WRITE_DIR)
    # form a path for the csv file
    save_path = os.path.join(results_dir, fig_filename)
    return save_path


def datetime_csv_filename(file_name):
    time_now = datetime.datetime.now()
    date_time_str = time_now.strftime("%Y-%m-%d-%H%M")
    csv_filename = "{}_{}.csv".format(file_name, date_time_str)
    return csv_filename


def result_csv_save_path(csv_filename):
    # path to the results dir
    results_dir = safe_write_dir_path(WRITE_DIR)

    # add datetime to the filename
    datetime_filename = datetime_csv_filename(csv_filename)
    # form a path for the csv file
    save_path = os.path.join(results_dir, datetime_filename)
    return save_path


def regression_datasets(test=False):
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    if test:
        datasets = {
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
        }
    else:
        datasets = {
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
            "california": {"loader": california_housing, "params": {}},
            "wine-red": {"loader": wine_quality, "params": {"color": "red"}},
            "wine-white": {"loader": wine_quality, "params": {"color": "white"}},
            "concrete": {"loader": concrete, "params": {}},
            "cpu-small": {"loader": cpu_small, "params": {}},
            "airquality": {"loader": airquality, "params": {}},
            "superconductor": {"loader": superconductor, "params": {}},
            "qm9-10k": {"loader": qm9, "params": {"size": 10_000}},
            "synthetic-cos-r": {
                "loader": make_synth,
                "params": {"n": 200_000, "p": 200},
            },
        }
    return datasets


def regression_knn_datasets(test=False):
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    if test:
        datasets = {
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
        }
    else:
        datasets = {
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
            "california": {"loader": california_housing, "params": {}},
            "wine-red": {"loader": wine_quality, "params": {"color": "red"}},
            "wine-white": {"loader": wine_quality, "params": {"color": "white"}},
            "concrete": {"loader": concrete, "params": {}},
            "cpu-small": {"loader": cpu_small, "params": {}},
            "airquality": {"loader": airquality, "params": {}},
            "superconductor": {"loader": superconductor, "params": {}},
            "qm9-10k": {"loader": qm9, "params": {"size": 10_000}},
            "synthetic-cos-r": {
                "loader": make_synth,
                "params": {"n": 10_000, "p": 20},
            },
        }
    return datasets


def regression_datasets_mini(test=False):
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    if test:
        datasets = {
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
        }
    else:
        datasets = {
            "superconductor": {"loader": superconductor, "params": {}},
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
            "california": {"loader": california_housing, "params": {}},
            "wine-red": {"loader": wine_quality, "params": {"color": "red"}},
            "wine-white": {"loader": wine_quality, "params": {"color": "white"}},
            "concrete": {"loader": concrete, "params": {}},
            "cpu-small": {"loader": cpu_small, "params": {}},
            "airquality": {"loader": airquality, "params": {}},
            "qm9-10k": {"loader": qm9, "params": {"size": 10_000}},
        }
    return datasets


def features_reg_datasets(test=False):
    if test:
        datasets = {
            "autompg": {"loader": auto_mpg, "params": {}},
            "abalone": {"loader": abalone, "params": {}},
        }
    else:
        datasets = {
            "california": {"loader": california_housing, "params": {}},
            "wine-white": {"loader": wine_quality, "params": {"color": "white"}},
            "concrete": {"loader": concrete, "params": {}},
            "cpu-small": {"loader": cpu_small, "params": {}},
            "superconductor": {"loader": superconductor, "params": {}},
            "qm9-10k": {"loader": qm9, "params": {"size": 10_000}},
            "synthetic-cos-r": {
                "loader": make_synth,
                "params": {"n": 200_000, "p": 200},
            },
        }
    return datasets


def features_cls_datasets(test=False):
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    if test:
        datasets = {
            "diabetes": {
                "loader": diabetes,
                "params": {},
            },
            "german-credit": {
                "loader": german_credit,
                "params": {},
            },
        }
    else:
        datasets = {
            "higgs-10k": {
                "loader": higgs,
                "params": {"size": 10_000},
            },
            "eeg_eye_state": {
                "loader": eeg_eye_state,
                "params": {},
            },
            "synthetic-cos-c": {
                "loader": make_synth,
                "params": {"n": 200_000, "p": 200, "classification": True},
            },
        }
    return datasets


def classification_datasets(test=False):
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    if test:
        datasets = {
            "wines": {
                "loader": wines_classification,
                "params": {},
            },
            "breast-cancer": {
                "loader": breast_cancer,
                "params": {},
            },
        }
    else:
        datasets = {
            "breast-cancer": {
                "loader": breast_cancer,
                "params": {},
            },
            "diabetes": {
                "loader": diabetes,
                "params": {},
            },
            "german-credit": {
                "loader": german_credit,
                "params": {},
            },
            "higgs-10k": {
                "loader": higgs,
                "params": {"size": 10000},
            },
            "eeg_eye_state": {
                "loader": eeg_eye_state,
                "params": {},
            },
            "synthetic-cos-c": {
                "loader": make_synth,
                "params": {"n": 200_000, "p": 200, "classification": True},
            },
        }
    return datasets


def classification_datasets_mini(test=False):
    """Lazy loader dictionary for regression datasets

    Returns:
        Dictionary: Dict with data loaders and loader params
    """
    if test:
        datasets = {
            "wines": {
                "loader": wines_classification,
                "params": {},
            },
            "breast-cancer": {
                "loader": breast_cancer,
                "params": {},
            },
        }
    else:
        datasets = {
            "breast-cancer": {
                "loader": breast_cancer,
                "params": {},
            },
            "diabetes": {
                "loader": diabetes,
                "params": {},
            },
            "german-credit": {
                "loader": german_credit,
                "params": {},
            },
            "higgs-10k": {
                "loader": higgs,
                "params": {"size": 10000},
            },
            "eeg_eye_state": {
                "loader": eeg_eye_state,
                "params": {},
            },
        }
    return datasets
