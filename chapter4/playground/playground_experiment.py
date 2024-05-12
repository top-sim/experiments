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

cfg_path = Path("/home/rwb/Dropbox/University/PhD/experiment_data/chapter4/results_with_metadata/low/prototype/skaworkflows_2024-05-12_12:29:18")

if not cfg_path.exists():
    LOGGER.warning(f"Exiting simulation, simulation config does not exist")
    exit()

from topsim.utils.experiment import Experiment

e = Experiment([cfg_path], [("batch", "batch")], output=run_path)
e.run()
