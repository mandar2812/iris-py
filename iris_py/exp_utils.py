"""Methods for training IRIS classifiers.

@date: 2022-07-24
@authors: Mandar Chandorkar
"""

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import logging
import shutil
import yaml

from sklearn import datasets
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, Kernel
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

log = logging.getLogger(__name__)


_KERNELS: Dict[str, Kernel] = {"RBF": RBF, "Matern": Matern}


@dataclass(frozen=True)
class IRISData:
    training: Tuple[pd.DataFrame, pd.DataFrame]
    test: Tuple[pd.DataFrame, pd.DataFrame]
    n_features: int
    n_classes: int


def create_experiment(top_dir: Path, prefix: str = "Iris") -> Path:
    """Create a new experiment directory with proper structure.

    Args:
        top_dir (Path): Create the new experiment under this dir.
        prefix (str): Experiment name starts with the provided prefix.
    """
    now = datetime.now()
    new_exp_dir = top_dir / f"{prefix}_{now:%Y-%m-%d-%H-%M}"
    model_dir = new_exp_dir / "model"
    results_dir = new_exp_dir / "results"
    plots_dir = results_dir / "plots"
    data_dir = new_exp_dir / "data"
    for dir in [new_exp_dir, model_dir, results_dir, data_dir, plots_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    return new_exp_dir


def create_iris_split(test_fraction: float, seed: int = 42) -> IRISData:
    """Load and randomly split the iris dataset."""
    assert test_fraction < 1.0, "Test data fraction must be less than 1.0."
    iris_dataset: Tuple[pd.DataFrame, pd.DataFrame] = datasets.load_iris(
        return_X_y=True, as_frame=True
    )
    x, y = iris_dataset

    x_test: pd.DataFrame = x.sample(frac=test_fraction, random_state=seed)
    y_test: pd.DataFrame = y.loc[x_test.index]

    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return IRISData(
        training=(x_train, y_train),
        test=(x_test, y_test),
        n_features=x.shape[1],
        n_classes=y.unique().shape[0],
    )


def save_data_splits(data_dir: Path, dataset: IRISData):
    """Serialise a split data set to disk."""
    log.info(f"Saving data splits.")
    dataset.training[0].to_json(
        data_dir / "training_features.json", orient="records", indent=4
    )
    dataset.training[1].to_json(
        data_dir / "training_labels.json", orient="records", indent=4
    )

    dataset.test[0].to_json(data_dir / "test_features.json", orient="records", indent=4)
    dataset.test[1].to_json(data_dir / "test_labels.json", orient="records", indent=4)


def run_new(exp_config_path: Path):
    """Experiment runner.

    Run a new IRIS training experiment.
    The method accepts keyword arguments which are read from YAML config files.
    """
    log.info(f"Loading experiment config from {exp_config_path}")
    assert exp_config_path.is_file(), f"{exp_config_path} does not exist."
    with open(exp_config_path, "r") as config_file:
        exp_config: Dict[str, Any] = yaml.safe_load(config_file)

    exp_top_dir: Path = exp_config.get("exp_top_dir", Path.home())
    # Create a new experiment directory
    exp_dir = create_experiment(exp_top_dir)

    log.info(f"Created experiment {exp_dir}")

    # Copy the experiment config.
    shutil.copyfile(exp_config_path, exp_dir / "config.yaml")

    test_frac: float = float(exp_config["data_prep"]["test_data_frac"])
    dataset: IRISData = create_iris_split(test_fraction=test_frac)
    save_data_splits(exp_dir / "data", dataset)

    kernel_kwargs: Dict[str, Any] = exp_config["model"]["kernel"].get("kwargs", {})

    kernel = 1.0 * _KERNELS[exp_config["model"]["kernel"]["name"]](
        [1.0] * dataset.n_features, **kernel_kwargs
    )

    x_train, y_train = dataset.training
    x_test, y_test = dataset.test

    model = GaussianProcessClassifier(kernel=kernel)
    model.fit(x_train, y_train)

    # Make predictions on test set.
    y_test_pred = model.predict(x_test)
    confusion_mat: np.ndarray = confusion_matrix(y_test, y_test_pred)
    np.save(exp_dir / "results" / "confusion_mat.npy", confusion_mat)
    print(confusion_mat)

    # Make charts from evaluation metrics.
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    cm_disp.plot()
    plt.savefig(exp_dir / "results" / "plots" / "confusion_matrix.png")
    plt.close()
