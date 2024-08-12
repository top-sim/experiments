# Copyright (C) 2024 RW Bunney
import json
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

BASE_PATH = Path(
    "/home/rwb/Dropbox/University/PhD/experiment_data/chapter5/interdependency/subset_schedule_comparison")
# skaworkflows_2024-06-25_20-42-08_9.json
cfg_paths = sorted([(BASE_PATH / p) for p in os.listdir(BASE_PATH) if ".json" in p])

for p in cfg_paths:
    if not p.exists():
        LOGGER.warning(f"Exiting simulation, simulation config %s does not exist", p)
        exit()


from topsim.utils.experiment import Experiment

# for p in cfg_paths:
#     if 'tmp.' in p.name:
#         p.rename((p. parent / str(p.name.split('tmp.')[1])))

# i = 0
# while i < 32:
#     cfg_paths[i].rename(cfg_paths[i].parent / str("tmp." + cfg_paths[i].name))
#     i += 1
from memory_profiler import profile

@profile
def runexp():
    # e = Experiment(cfg_paths, [("batch", "batch") ], output=run_path)
    # e = Experiment(cfg_paths, [("batch", "batch"), ("static", "dynamic_plan")],
    #                output=run_path)
    e = Experiment(cfg_paths, [("static", "dynamic_plan") ], output=run_path)
    e.run()

runexp()
"""
Investigate: 
    /home/rwb/Dropbox/University/PhD/experiment_data/chapter5/interdependency/low_maximal/skaworkflows_2024-06-25_20-58-16_9.json
    
Looks like this is braking the simulator as a result of recent buffer changes.
"""
