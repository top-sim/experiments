# Copyright (C) 22/2/22 RW Bunney

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
RUN_PATH = Path(f'chapter4/topsim_initial_experiment/')
#
# cfg_path = Path(
#     '/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/topsim_initial_experiment/low/prototype/low_sdp_config.json')

# cfg_path_scatter = Path('/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/topsim_initial_experiment/low/prototype/no_data_low_sdp_config_prototype_n896_896channels.json')
cfg_path_scatter = Path('/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/topsim_initial_experiment/low/scatter/no_data_low_sdp_config_scatter_n896_896channels.json')

if not cfg_path_scatter.exists():
    LOGGER.info(f"Exiting simulation, simulation config does not exist")
    exit()

from topsim.utils.experiment import Experiment

e = Experiment([cfg_path_scatter], [("batch", "batch")])
e.run()
