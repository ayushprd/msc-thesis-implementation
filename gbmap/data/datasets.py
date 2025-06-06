import os
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris, load_breast_cancer
from urllib.request import urlretrieve
from zipfile import ZipFile
from ucimlrepo import fetch_ucirepo
import jax.numpy as jnp
from jax import random
from jax.nn import softplus, sigmoid
from gbmap.embedding.common import random_unit_vector


def script_abs_path():
    # get the script absolute path
    return os.path.dirname(os.path.abspath(__file__))


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def z_scale_xy(x, y=None):
    scaler_x = StandardScaler()
    x = scaler_x.fit_transform(x)
    if y is None:
        # only x was given
        return x

    # scale y
    scaler_y = StandardScaler()
    if len(y.shape) == 1:
        y = scaler_y.fit_transform(y[:, None])
    elif len(y.shape) == 2:
        y = scaler_y.fit_transform(y)
    else:
        raise ValueError("Invalid format for target vector ({})".format(y.shape))
    return x, y.ravel()


def safe_fetch_uci(id, retrys=3):
    try_n = 0
    while try_n < retrys:
        try:
            data = fetch_ucirepo(id=id)
            return data
        except:
            try_n += 1
            print(
                "Could not fetch the UCI dataset id={}, retrying... {}/{}".format(
                    id, try_n, retrys
                )
            )
    raise RuntimeError("Could not connect to the server, or invalid id={}".format(id))


def auto_mpg(return_df=False):
    # read and preprocess auto mpg dataset

    auto_mpg = pd.read_csv(
        os.path.join(script_abs_path(), "datasets/auto-mpg.data"),
        names=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
            "carname",
        ],
        delim_whitespace=True,
        na_values=["?"],
    )

    X0 = auto_mpg[
        [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
        ]
    ]
    y0 = auto_mpg["mpg"]

    # Split and one-hot encode the origin into USA vs Europe vs Japan
    X0 = np.concatenate(
        (
            X0.values[:, :-1].astype(float),
            np.eye(3)[X0["origin"].values.astype(int) - 1],
        ),
        axis=1,
    )
    y0 = y0.values

    # X0 contains the covariates, y0 is the target variable and names are column names.
    mask = ~np.isnan(X0[:, 2])
    X0 = X0[mask]
    y0 = y0[mask]
    auto_mpg = auto_mpg[mask]

    names = list(auto_mpg.columns[1:-2]) + [
        "origin USA",
        "origin Europe",
        "origin Japan",
    ]

    data = pd.DataFrame(
        np.concatenate([X0, y0.reshape((-1, 1))], axis=1), columns=names + ["mpg"]
    )

    # X and y are normalised by `sklearn.preprocessing.StandardScaler`.
    scale_x = StandardScaler()
    scale_y = StandardScaler()
    X = np.concatenate([scale_x.fit_transform(X0[:, :-3]), X0[:, -3:]], axis=1)
    y = scale_y.fit_transform(y0[:, None])
    y = y.flatten()
    # return X, y, data, np.array(auto_mpg["origin"]) - 1

    if return_df:
        return X, y, data
    else:
        return X, y


def toy_ctsne():
    # conditional t-SNE toy dataset

    df = pd.read_csv(
        os.path.join(script_abs_path(), "datasets/synthetic_toy_ctsne.csv")
    )
    # returns tuple with X, df with labels for clusters
    return (
        df[
            ["d_0", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_8", "d_9"]
        ].values,
        df,
    )


def parabola_1d(size=100, seed=1):
    np.random.seed(seed)
    x = np.linspace(-1, 1, size)
    x = x[:, None]
    y = -(x**2) + 1 + np.random.normal(loc=0, scale=0.1, size=(size, 1))

    scaler_x = StandardScaler()
    x = scaler_x.fit_transform(x)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    return x, y.ravel()


def stepdata(size=100, seed=1):
    np.random.seed(seed)
    x = np.linspace(-1, 1, size)
    y = 2 * (x > 0) + np.random.normal(loc=0, scale=0.05, size=size)

    return x[:, None], y


def geckoq():
    df_data = pd.read_csv(
        os.path.join(script_abs_path(), "datasets/geckoq/Updatedframe.csv"),
    )

    real = [
        "NumOfAtoms",
        "NumOfC",
        "NumOfO",
        "NumOfN",
        "NumHBondDonors",
        "NumHBondAcceptors",
        "NumIntHBonds",
        "C=C (non-aromatic)",
        "C=C-C=O in non-aromatic ring",
        "hydroxyl (alkyl)",
        "aldehyde",
        "ketone",
        "carboxylic acid",
        "ester",
        "nitrate",
        "nitro",
        "carbonylperoxynitrate",
        "hydroperoxide",
        "carbonylperoxyacid",
        "nitroester",
        "na-rings",
        "phenol",
    ]

    binary = ["ether (alicyclic)", "peroxide", "a-rings", "nitrophenol"]

    real_scaled = z_scale_xy(x=df_data[real].values)

    x = np.concatenate([real_scaled, df_data[binary].values], axis=1)

    y = np.log10(df_data["pCOSMO_atm"]).values
    y = y - y.mean()

    return x, y


def binarydata(size1=100, size2=100, seed=1):
    np.random.seed(seed)
    x1 = np.random.multivariate_normal(np.array([0, 0]), np.eye(2), size1)
    x2 = np.random.multivariate_normal(np.array([3, 3]), np.eye(2), size2)
    x = np.concatenate((x1, x2))
    y1 = np.ones(size1)
    y2 = -np.ones(size2)
    y = np.concatenate((y1, y2))
    return x, y


def binarydata2(size1=100, size2=100, seed=1, scale=4):
    np.random.seed(seed)
    x1 = np.random.multivariate_normal(np.array([0, 0]), np.eye(2), size1)
    x2 = scale * np.random.multivariate_normal(np.array([0, 0]), np.eye(2), size2)
    x = np.concatenate((x1, x2))
    y1 = np.ones(size1)
    y2 = -np.ones(size2)
    y = np.concatenate((y1, y2))
    return x, y


def sine_1d(size=100, seed=1):
    np.random.seed(seed)
    x = np.linspace(-4, 4, size)
    x = x[:, None]
    y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=(size, 1))

    return x, y.ravel()


def mnist(class1, class2, normalize=True, test_size=None):
    class_list = list(range(10))

    if class1 not in class_list or class2 not in class_list:
        raise ValueError("class1 and class2 must be integer numbers from 0 to 9.")

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    x_mnist = mnist.data
    y_mnist = mnist.target

    class1 = str(class1)
    class2 = str(class2)

    x_mnist_c1 = x_mnist[y_mnist == class1]
    x_mnist_c2 = x_mnist[y_mnist == class2]
    y_mnist_c1 = np.ones(x_mnist_c1.shape[0])
    y_mnist_c2 = -np.ones(x_mnist_c2.shape[0])

    X = np.concatenate((x_mnist_c1, x_mnist_c2), axis=0)
    y = np.concatenate((y_mnist_c1, y_mnist_c2))

    if normalize:
        max_norm = np.max(np.sqrt(np.sum(X**2, axis=1)))
        X = X / max_norm

    if test_size is None:
        return X, y
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        return x_train, x_test, y_train, y_test


def california_housing():
    x, y = fetch_california_housing(return_X_y=True)
    x, y = z_scale_xy(x, y)
    return x, y


def wine_quality(color="red"):
    if not (color == "red" or color == "white"):
        raise ValueError(
            "color parameter should be 'red' or 'white' (but was {})".format(color)
        )
    # source: https://archive.ics.uci.edu/dataset/186/wine+quality
    data = safe_fetch_uci(id=186)
    # data = fetch_ucirepo(id=186)
    data_all = data["data"]["original"]
    df = data_all.loc[data_all["color"] == color]

    features = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    target = "quality"
    x = df[features].values
    y = df[target].values
    x, y = z_scale_xy(x, y)

    return x, y


def drift_dataset2(
    train_size=100,
    test_size=100,
    n_dim=2,
    n_noise_dim=0,
    mu1=None,
    cov1=None,
    mu2=None,
    cov2=None,
    func=None,
    noise_scale=1,
    seed=42,
    rotation=True,
    classification=False,
    gp_random_state=0,
    kernel=1.0 * RBF(length_scale=0.8),
):
    np.random.seed(seed=seed)
    if mu1 is None:
        mu1 = np.zeros(n_dim)
    if mu2 is None:
        mu2 = np.ones(n_dim)
    if cov1 is None:
        cov1 = np.eye(n_dim)
    if cov2 is None:
        cov2 = np.eye(n_dim)

    w = 1 + np.random.normal(size=(n_dim,))
    # w = np.ones(shape=(n_dim, ))
    x_train = np.random.multivariate_normal(mu1, cov1, train_size)
    x_test = np.random.multivariate_normal(mu2, cov2, test_size)

    # if func is None, the default would be GP samples
    if func is None:
        gp = GaussianProcessRegressor(kernel=kernel)
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.squeeze(
            gp.sample_y(x, n_samples=1, random_state=gp_random_state)
            + np.random.normal(loc=0, scale=0.1, size=(train_size + test_size, 1))
        )
        y_train = y[0:train_size]
        y_test = y[train_size:]

    else:
        y_train = func(x_train) @ w.T + np.squeeze(
            np.random.normal(loc=0, scale=0.1, size=(train_size, 1))
        )
        y_test = func(x_test) @ w.T + np.squeeze(
            np.random.normal(loc=0, scale=0.1, size=(test_size, 1))
        )

    if classification:
        # convert to probability
        y_train_prob = 1 / (1 + np.exp(-y_train))
        y_test_prob = 1 / (1 + np.exp(-y_test))

        y_train = []
        y_test = []
        for i in range(y_train_prob.shape[0]):
            p = y_train_prob[i]
            label = np.random.choice([1, -1], size=1, p=[p, 1 - p])
            y_train.append(label)
        for i in range(y_test_prob.shape[0]):
            p = y_test_prob[i]
            label = np.random.choice([1, -1], size=1, p=[p, 1 - p])
            y_test.append(label)
        y_train = np.array(y_train).astype(int)
        y_test = np.array(y_test).astype(int)

    # include noise
    x_train = np.hstack(
        [
            x_train,
            noise_scale
            * np.random.normal(loc=0, scale=1, size=(train_size, n_noise_dim)),
        ]
    )
    x_test = np.hstack(
        [
            x_test,
            noise_scale
            * np.random.normal(loc=0, scale=1, size=(test_size, n_noise_dim)),
        ]
    )

    # rotation
    Q = None
    if rotation:
        random_matrix = np.random.rand(x_train.shape[1], x_train.shape[1])
        Q, R = np.linalg.qr(random_matrix)
        x_train = x_train @ Q
        x_test = x_test @ Q

    return x_train, x_test, y_train, y_test, Q


def toy_dataset(
    train_size=100,
    test_size1=100,
    test_size2=100,
    n_dim=2,
    n_noise_dim=0,
    m1=None,
    m2=None,
    seed=42,
    gp_random_state=0,
    kernel=1.0 * RBF(length_scale=0.8),
):
    np.random.seed(seed=seed)

    mu = np.zeros(n_dim + n_noise_dim)
    mu1 = np.hstack((m1 * np.ones(n_dim), np.zeros(n_noise_dim)))
    mu2 = np.hstack((np.zeros(n_dim), m2 * np.ones(n_noise_dim)))
    cov = np.eye(n_dim + n_noise_dim)

    x_train = np.random.multivariate_normal(mu, cov, train_size)
    x_test1 = np.random.multivariate_normal(mu1, cov, test_size1)
    x_test2 = np.random.multivariate_normal(mu2, cov, test_size2)
    gp = GaussianProcessRegressor(kernel=kernel)
    x = np.concatenate((x_train, x_test1, x_test2), axis=0)
    y = np.squeeze(
        gp.sample_y(x[:, :n_dim], n_samples=1, random_state=gp_random_state)
        + np.random.normal(
            loc=0, scale=0.1, size=(train_size + test_size1 + test_size2, 1)
        )
    )
    y_train = y[0:train_size]
    y_test1 = y[train_size : train_size + test_size1]
    y_test2 = y[train_size + test_size1 :]

    return x_train, x_test1, x_test2, y_train, y_test1, y_test2


def wines_classification():
    # source: https://archive.ics.uci.edu/dataset/186/wine+quality
    # data = fetch_ucirepo(id=186)
    data = safe_fetch_uci(id=186)

    df = data["data"]["original"]
    # df = data_all.loc[data_all["color"] == color]

    features = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    target = "color"
    x = df[features].values
    y = df[target].values
    x = z_scale_xy(x)

    # red == 1, white == -1
    y = 1 * (y == "red")
    y[y == 0] = -1

    return x, y


def iris_binary():
    data = load_iris()
    # targets: ['setosa', 'versicolor', 'virginica']
    y = data["target"]
    # select 'versicolor' and 'virginica'
    mask = y > 0
    x = data["data"][mask, :]
    y = y[mask]
    # scale x
    x = z_scale_xy(x)
    # versicolor == 1, virginica == -1
    y[y == 2] = -1
    return x, y


def breast_cancer():
    # data source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    x, y = load_breast_cancer(return_X_y=True)
    # malignant == -1, benign == 1
    y[y == 0] = -1
    x = z_scale_xy(x)
    return x, y


def diabetes():
    df = pd.read_csv(os.path.join(script_abs_path(), "datasets/diabetes.csv"))
    x = df[
        [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
    ].values
    x = z_scale_xy(x)
    # outcome 0 == -1, outcome 1 == 1
    y = df["Outcome"].values
    y[y == 0] = -1
    return x, y


def abalone():
    # source: https://archive.ics.uci.edu/dataset/1/abalone

    # fetch dataset
    # abalone = fetch_ucirepo(id=1)
    abalone = safe_fetch_uci(id=1)

    # data (as pandas dataframes)
    features = abalone.data.features
    y = abalone.data.targets.values

    sex_one_hot = np.eye(2)[(features["Sex"] == "F").values.astype(int)]
    real_vars = features[
        [
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight",
        ]
    ].values

    real_vars, y = z_scale_xy(real_vars, y)
    x = np.concatenate([sex_one_hot, real_vars], axis=1)
    return x, y


def german_credit():
    # source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    # df = fetch_ucirepo(id=144)
    df = safe_fetch_uci(id=144)

    X = df.data.features
    y = df.data.targets["class"]

    # List of categorical and numerical features
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    preprocessing = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(), categorical_features),
            ("scaler", StandardScaler(), numerical_features),
        ]
    )

    x = preprocessing.fit_transform(X)
    y = np.where(y == 1, 1, -1)
    return x, y


def concrete():
    # source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    df = fetch_openml(data_id=4353, parser="auto")
    data = df.data
    data = data.dropna()
    x = data[
        [
            "Cement (component 1)(kg in a m^3 mixture)",
            "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
            "Fly Ash (component 3)(kg in a m^3 mixture)",
            "Water  (component 4)(kg in a m^3 mixture)",
            "Superplasticizer (component 5)(kg in a m^3 mixture)",
            "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
            "Fine Aggregate (component 7)(kg in a m^3 mixture)",
            "Age (day)",
        ]
    ]
    y = data[["Concrete compressive strength(MPa. megapascals)"]]
    return z_scale_xy(x, y)


def cpu_small():
    # source https://www.openml.org/search?type=data&status=active&id=227
    x, y = fetch_openml(data_id=227, parser="auto", return_X_y=True)
    x = np.array(x)
    y = np.array(y)
    return z_scale_xy(x, y)


def higgs(size=None):
    # source https://www.openml.org/search?type=data&sort=runs&id=23512&status=active
    data = fetch_openml(data_id=23512, parser="auto")
    mask = ~data["data"].isnull().any(axis=1)
    x = data["data"][mask].values
    y = data["target"][mask].values.astype(int)

    x = z_scale_xy(x)
    y = np.where(y == 1, 1, -1)
    if size is None:
        return x, y
    return x[0:size], y[0:size]


def airquality(scale_=True):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    data_dir = script_abs_path()
    filename = os.path.join(data_dir, "datasets/AirQualityUCI.zip")
    csv_name = "AirQualityUCI.csv"

    if not os.path.isfile(filename):
        r = requests.get(url)
        if r.status_code == 200:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, "wb") as f:
                f.write(r.content)

    with ZipFile(filename, "r") as zip:
        with zip.open(csv_name) as f:
            data = pd.read_csv(f, sep=";", decimal=",")

    # Cleaning copied from https://bitbucket.org/edahelsinki/regressionfunction/src/master/python_notebooks/regressionfunction_notebook.ipynb
    data = data.replace(-200, np.nan)
    # impute cases where only 1 hour of data is missing by the mean of its successor and predessor
    for j in range(data.shape[1]):
        for i in range(1, data.shape[0]):
            if (
                (pd.isna(data.iloc[i, j]))
                and not pd.isna(data.iloc[i - 1, j])
                and not pd.isna(data.iloc[i + 1, j])
            ):
                data.iloc[i, j] = (data.iloc[i - 1, j] + data.iloc[i + 1, j]) / 2
    data = data.drop(columns=["NMHC(GT)"])  # Mostly NA.
    data = data.dropna(axis=1, how="all").dropna(axis=0)
    covariates = [
        "PT08.S1(CO)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    target = "CO(GT)"

    X = data[covariates].values
    y = data[target].values

    if scale_:
        X, y = z_scale_xy(X, y)
        return X, y
    else:
        return X, y


def qm9(normalize=True, size=None):
    """Get the QM9 dataset as used in the "Slisemap application" paper: http://arxiv.org/abs/2310.15610"""
    path = os.path.join(script_abs_path(), "datasets", "slisemap_phys.zip")
    if not os.path.exists(path):
        url = "https://www.edahelsinki.fi/papers/SI_slisemap_phys.zip"
        urlretrieve(url, path)
    df = pd.read_feather(ZipFile(path).open("SI/data/qm9_interpretable.feather"))
    df.drop(columns="index", inplace=True)
    X = df.to_numpy(np.float32)
    y = pd.read_feather(ZipFile(path).open("SI/data/qm9_label.feather"))
    y = y["homo"].to_numpy()

    if size is not None:
        heavies_idx = (
            df.sort_values(by="MW", ascending=False).iloc[:size, :].index.values
        )
        X = X[heavies_idx, :]
        y = y[heavies_idx]
    if normalize:
        X, y = z_scale_xy(X, y)
    else:
        max_norm = np.max(np.sqrt(np.sum(X**2, axis=1)))
        X = X / max_norm
        y = y - y.mean()

    return X, y


def eeg_eye_state():
    # source https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=1471
    x, y = fetch_openml(data_id=1471, parser="auto", return_X_y=True)
    x = np.array(x)
    y = np.array(y)
    x = z_scale_xy(x)
    y = np.where(y == "1", 1, -1).astype(int)  # Convert 0 to -1
    return x, y


def madelon():
    # https://www.openml.org/search?type=data&status=active&id=1485
    x, y = fetch_openml(data_id=1485, parser="auto", return_X_y=True)
    y = np.where(y.astype(int) == 1, 1, -1)
    x = z_scale_xy(x=x.values)
    return x, y


def mushroom():
    x, y = fetch_openml(data_id=24, parser="auto", return_X_y=True)
    x = x.drop(columns=["veil-type"])  # non-informative, only one unique value
    preprocessing = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(), list(x.columns)),
        ]
    )
    x = preprocessing.fit_transform(x).toarray()
    y = np.where(y == "p", 1, -1)
    return x, y


def wave_energy():
    # https://www.openml.org/search?type=data&status=active&id=44975
    # https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm
    x, y = fetch_openml(data_id=44975, parser="auto", return_X_y=True)
    variables = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "y1",
        "y2",
        "y3",
        "y4",
        "y5",
        "y6",
        "y7",
        "y8",
        "y9",
        "y10",
        "y11",
        "y12",
        "y13",
        "y14",
        "y15",
        "y16",
    ]
    x, y = z_scale_xy(
        x=x[variables].values,
        y=y.values,
    )
    return x, y


def superconductor(size=None):
    # https://www.openml.org/search?type=data&status=active&id=43174
    # https://archive.ics.uci.edu/dataset/464/superconductivty+data
    x, y = fetch_openml(data_id=43174, parser="auto", return_X_y=True)
    x = x.values
    y = y.values
    if size is not None:
        np.random.seed(42)
        indices = np.random.choice(x.shape[0], size, replace=False)
        x = x[indices, :]
        y = y[indices]
    x, y = z_scale_xy(x, y)
    return x, y


def get_gbdata(size, dims, classification=False):
    WRITE_DIR = "gbdata"
    full_write_path = os.path.join(script_abs_path(), WRITE_DIR)
    create_dir(full_write_path)

    task_suffix = ""
    if classification:
        task_suffix = "_cls"

    file_name_x = "n{}d{}_x.npy".format(size, dims)
    file_name_y = "n{}d{}_y{}.npy".format(size, dims, task_suffix)
    save_pathx = os.path.join(full_write_path, file_name_x)
    save_pathy = os.path.join(full_write_path, file_name_y)

    if not os.path.isfile(save_pathx) or not os.path.isfile(save_pathy):
        print("size={} dims={} not in cache, generating...".format(size, dims))

        x, _, y, _, _ = drift_dataset2(
            train_size=size,
            test_size=0,
            n_dim=dims,
            n_noise_dim=0,
            rotation=False,
            classification=classification,
        )
        np.save(save_pathx, x)
        np.save(save_pathy, y)
        print("Saved x to: {}".format(save_pathx))
        print("Saved y to: {}".format(save_pathy))
    else:
        # load data
        x = np.load(save_pathx)
        y = np.load(save_pathy)

    x = StandardScaler().fit_transform(x)
    if not classification:
        y = StandardScaler().fit_transform(y[:, None])
        y = y.ravel()

    return x, y


def creditcard():
    # source https://www.openml.org/search?type=data&status=active&id=1597
    x, y = fetch_openml(data_id=1597, parser="auto", return_X_y=True)
    x = np.array(x)
    y = np.array(y)
    x = z_scale_xy(x)
    y = np.where(y == "1", 1, -1).astype(int)  # Convert 0 to -1
    df = pd.DataFrame(x)
    df["y"] = y
    df_class_1 = df[df["y"] == 1]
    df_class_minus_1 = df[df["y"] == -1]
    df_class_minus_1_sample = df_class_minus_1.sample(10000, random_state=42)
    df_sampled = pd.concat([df_class_1, df_class_minus_1_sample])
    df_sampled = df_sampled.sample(frac=1, random_state=123).reset_index(drop=True)
    y_sampled = df_sampled["y"].values
    x_sampled = df_sampled.drop(columns=["y"]).values
    return x_sampled, y_sampled


def scale_data(n_size=100, n_dims=2, random_seed=0, classification=False):
    np.random.seed(random_seed)
    mu1 = np.zeros(n_dims)
    cov1 = np.eye(n_dims)
    x = np.random.multivariate_normal(mu1, cov1, n_size)
    w = np.random.multivariate_normal(np.zeros(1), np.eye(1), n_dims)
    w = w / w.sum()
    y = (np.sin(x) @ w).ravel() + np.random.normal(0, 0.05, n_size)
    y = y / y.std()

    if classification:
        # sigmoid
        prob = 1 / (1 + np.exp(-y))
        y = np.where(prob < 0.5, 1, -1)

    return x, y


# Make synthetic data as described in the manuscript
def make_synth(
    n,
    p,
    ntrain=None,
    prel=None,
    nonlin=lambda x: 5.0 * jnp.cos(x),
    get_drift=False,
    classification=False,
    seed=42,
):
    key = random.PRNGKey(seed)
    if ntrain is None:
        ntrain = n
    if prel is None:
        prel = p
    keyX, key_irr, keyu, keyU = random.split(key, 4)
    X = random.normal(keyX, (n, prel))
    u = random_unit_vector(prel, keyu)
    if get_drift:
        drift = X @ u
    else:
        drift = None
    y = nonlin(X) @ u

    if prel < p:
        X = jnp.hstack((X, random.normal(key_irr, (n, p - prel))))
    X = X @ random.orthogonal(keyU, p)

    y = y - jnp.average(y[:ntrain])

    if classification:
        y = jnp.select(
            [random.uniform(key, y.shape) <= sigmoid(y)], [1.0], default=-1.0
        )

    if drift:
        return np.array(X), np.array(y), drift
    else:
        return np.array(X), np.array(y)


def make_synth_drift(
    n,
    p,
    key,
    ntrain=None,
    prel=None,
    nonlin=lambda x: 5.0 * jnp.cos(x),
):
    if ntrain is None:
        ntrain = n
    if prel is None:
        prel = p
    keyX, keyX2, key_irr, key_irr2, keyu, keyU = random.split(key, 6)
    X = random.normal(keyX, (n, prel))
    X2 = random.normal(
        keyX2, (n, prel)
    )  # we use X2 to get another training data set only (for F1 score computation)
    u = random_unit_vector(prel, keyu)

    drift = X @ u
    drift2 = X2 @ u

    y = nonlin(X) @ u
    y2 = nonlin(X2) @ u

    if prel < p:
        X = jnp.hstack((X, random.normal(key_irr, (n, p - prel))))
        X2 = jnp.hstack((X2, random.normal(key_irr2, (n, p - prel))))
    X = X @ random.orthogonal(keyU, p)
    X2 = X2 @ random.orthogonal(keyU, p)

    y = y - jnp.average(y[:ntrain])
    y2 = y2 - jnp.average(
        y[:ntrain]
    )  # we use y[:,ntrain] (not y2[:train]) for consistency

    return (
        X,
        y,
        drift,
        X2,
        y2,
        drift2,
    )
