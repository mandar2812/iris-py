"""Command line entry point - IRIS Py

@date: 2022-07-24
@authors: Mandar Chandorkar
"""

import argparse
from typing import Any, Callable, Dict
import pkg_resources
from pathlib import Path
from iris_py.exp_utils import run_new

_default_exp_config: Path = Path(
    pkg_resources.resource_filename(__name__, "data/default_config.yaml")
)

parser_entry = argparse.ArgumentParser(description="IRIS Model Training Utility")
subparsers = parser_entry.add_subparsers(dest="subcommand")

parser_new_exp = subparsers.add_parser(
    "new-exp", aliases=["new"], help="New training experiment."
)

parser_new_exp.add_argument(
    "--config", dest="config", type=Path, default=_default_exp_config
)
parser_new_exp.set_defaults(process_func=run_new)

args: Dict[str, Any] = vars(parser_entry.parse_args())


process_func: Callable[[Dict[str, Any]], None] = args.pop("process_func")
process_func(args["config"])
