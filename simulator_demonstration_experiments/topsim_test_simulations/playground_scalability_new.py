# Copyright (C) 2024 RW Bunney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import time
import argparse

import logging
import simpy
from pathlib import Path
from datetime import date

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)
parser = argparse.ArgumentParser(Path(__file__).name, )
parser.add_argument('method', help="batch, static, all", type=str)
parser.add_argument('-f', '--file', help="Use a specific file", type=str)
parser.add_argument('-d', '--dir', help="Use a directory of config", type=str)
parser.add_argument('--slurm', help='Use slurm', action="store_true")

# RUN_PATH = Path.cwd()
run_path = Path(__file__).parent
logging.info("Script is running in %s, saving output there...", run_path)
experiment_path = None
args = parser.parse_args()
if args.dir and args.file:
    LOGGER.warning("Passed both directory and file options; they are incompatible")
    parser.print_help
    sys.exit()
elif args.dir and args.slurm:
    LOGGER.warning("Passing directory and slurm options is incompatible; pass specific file.")
    parser.print_help
    sys.exit()
elif args.dir:
    experiment_path = Path(args.dir)
elif args.file:
    LOGGER.debug("Running experiment with file...")
    experiment_path = Path(args.file)

cfg_paths = []
if experiment_path.is_dir():
    cfg_paths = [( experiment_path / p) for p in os.listdir(experiment_path) if ".json" in p]

    for p in cfg_paths:
        if not p.exists():
            LOGGER.warning(f"Exiting simulation, simulation config %s does not exist", p)
            exit()
else:
    cfg_paths.append(experiment_path)

from topsim.utils.experiment import Experiment

# TODO Fix up all the combinations here with slurm and directories/paths
LOGGER.debug("Using method: %s", args.method)
if args.method == 'batch':
    e = Experiment(cfg_paths, [("batch", "batch")], output=experiment_path.parent / "results",
                   sched_args={"ignore_ingest": False, "use_workflow_dop": True},
                   slurm=args.slurm)

elif args.method == 'static':
    e = Experiment(cfg_paths, [("static", "dynamic_plan")], output=experiment_path.parent / "results",
                   sched_args={"ignore_ingest": False, "use_workflow_dop": True},
                   slurm=args.slurm)
else:
    e = Experiment(
        cfg_paths,
        [("batch", "batch"), ("static", 'dynamic_plan')],
        output=experiment_path.parent / "results",
        sched_args={"ignore_ingest": False, "use_workflow_dop": True} # This is what we want to use by 'defualt'
    )

e.run()

