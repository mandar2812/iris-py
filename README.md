# IRIS Classification Python

This package is meant to be an outline for creating a python based ML experimentation system. Using the iris dataset as a simple example, it has the following features:

* Project structure based on [PyPA recommendations](https://python-packaging.readthedocs.io/en/latest/minimal.html).
* Pip installable for development purposes as well as a command line tool.
* Exprement initiation, processing and storage of important artifacts.

## Installation

There are two ways to install this package. Note that ssh access to github must be configured for (1) to work. Usage of a fresh python virtual environment is recommended `python -m venv /my/new/venv`.

1. As a command line utility.

```shell
pip install iris_py@git+ssh://git@github.com/mandar2812/iris-py.git#egg=iris_py
```

2. In development mode, (zsh users must use quotes i.e. `".[dev]"`).

```shell
git clone git@github.com:mandar2812/iris-py.git
cd iris-py
pip install -e .[dev]
```

## Usage

Run a new experiment with

```shell
python -m iris_py new-exp
```

The script creates a new experiment directory (like `Iris-{DATETIME}`) and stores the training results and configuration files inside it. Each experiment can be customised using a YAML config file which can be passed as a command line argument, if none is provided `data/default_config.yaml` is used.