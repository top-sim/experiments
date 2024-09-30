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

import logging
import simpy
from pathlib import Path
from datetime import date

logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)

# RUN_PATH = Path.cwd()
run_path = Path(__file__).parent
logging.info("Script is running in %s, saving output there...", run_path)

# cfg_path = Path("/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/results_with_metadata/low/prototype/skaworkflows_2024-05-12_12:29:18")

# BASE_PATH = Path("/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/scalability/low_maximal")
BASE_PATH = Path(sys.argv[1])

cfg_paths = [(BASE_PATH / p) for p in os.listdir(BASE_PATH) if ".json" in p]

# if not cfg_path.exists():
#     LOGGER.warning(f"Exiting simulation, simulation config does not exist")
#     exit()

for p in cfg_paths:
    if not p.exists():
        LOGGER.warning(f"Exiting simulation, simulation config %s does not exist", p)
        exit()

from topsim.utils.experiment import Experiment

alg = sys.argv[2]
if alg == 'batch':
    e = Experiment(cfg_paths, [("batch", "batch")], output=BASE_PATH / "results")
elif alg == 'static':
    e = Experiment(cfg_paths, [("static", "dynamic_plan")], output=BASE_PATH / "results")
else:
    e = Experiment(cfg_paths, [("batch", "batch"), ("static", 'dynamic_plan')], output=BASE_PATH / "results")

e.run()

# from shadow.algorithms.heuristic import heft, fcfs
# from shadow.models.workflow import Workflow
# from shadow.models.environment import Environment
# from skaworkflows.config_generator import config_to_shadow
#
# shadow_config = config_to_shadow(Path("/home/rwb/Dropbox/University/PhD/experiment_data"
#                                    "/chapter4/scalability/low_maximal/prototype/skaworkflows_2024-06-02_14-00-32.json"))
# env = Environment(shadow_config, dictionary=True)
# workflow = Workflow("/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/scalability/low_maximal/prototype/workflows/-6190016813116619223_2024-06-02_14:00:32")
# workflow.add_environment(env)
# print(fcfs(workflow, 0).makespan)
# print(heft(workflow, 1).makespan)
