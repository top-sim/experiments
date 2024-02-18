# Copyright (C) 6/5/22 RW Bunney

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

"""
Some baseline scheduling comparisons for workflows

Maximal channels will be 896 for LOW, 786 for MID for both the completely
parallel workflow, which is representative of the parametric scheduling model.

4 experiments exist here, with data generated for each accordingly.
"""

import logging

from pathlib import Path
import sys
sys.path.append('/home/rwb/github/skaworkflows')

import skaworkflows.config_generator as config_generator

logging.basicConfig(level='INFO')
LOGGER = logging.getLogger(__name__)

# Imaging HPSOs

hpso_path_telescope = (
    "chapter3/skaworkflows_scalability_tests/maximal_low_imaging_scalability_telescope.json"
)

hpso_path_channels = (
    "chapter3/skaworkflows_scalability_tests/maximal_low_imaging_scalability_channels.json"
)

BASE_DIR = Path(f"/home/rwb/Dropbox/University/PhD/experiment_data/chapter3/scalability")
# PROTOTYPE_WORKFLOW_PATHS = {"DPrepA": "prototype", "DPrepB": "prototype"}
PROTOTYPE_WORKFLOW_PATHS = {"ICAL": "prototype", "DPrepA": "prototype",
                            "DPrepB": "prototype", "DPrepC": "prototype",
                            "DPrepD": "prototype"}

# SCATTER_WORKFLOW_PATHS = {"ICAL": "scatter", "DPrepA": "scatter",
#                           "DPrepB": "scatter", "DPrepC": "scatter",
#                           "DPrepD": "scatter"}

LOW_CONFIG = Path("low_sdp_config")
# MID_CONFIG = Path("mid_sdp_config")

# TODO we are going to use the file-based config generation for this 
# TODO-TODO what did I mean by file-based config generation???

# base_graph_paths = PROTOTYPE_WORKFLOW_PATHS
low_path_str = BASE_DIR / 'low'
# mid_path_str = BASE_DIR / 'mid'
# Generate configuration with prototype SKA Workflow
# SKA LOW
import time
start = time.time()


config_generator.create_config('low', 'parametric', nodes=896,
    hpso_path=Path(hpso_path_telescope), output_dir=Path(low_path_str)/'prototype',
    cfg_name=f"{LOW_CONFIG}_prototype_telescope.json", base_graph_paths=PROTOTYPE_WORKFLOW_PATHS,
    timestep="seconds", data=False)

# config_generator.create_config('low', 'parametric', nodes=896,
#     hpso_path=Path(hpso_path_telescope), output_dir=Path(low_path_str)/'prototype',
#     cfg_name=f"{LOW_CONFIG}_prototype_channels.json", base_graph_paths=PROTOTYPE_WORKFLOW_PATHS,
#     timestep="seconds", data=False)

finish = time.time()

LOGGER.info(f"Total generation time was: {(finish-start)/60} minutes")
