# Copyright (C) 11/4/22 RW Bunney

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
Create a single hpso config file + workflow for each hpso in SKA-LOW,
for the multiple baselines for which we have done costing.
"""

import pandas as pd
import json
import logging

import skaworkflows.common as common
import skaworkflows.workflow.hpso_to_observation as hto

from pathlib import Path
from skaworkflows.hpconfig.utils.classes import ARCHITECTURE
from skaworkflows.hpconfig.specs.sdp import SDP_PAR_MODEL
from skaworkflows.config_generator import create_config, config_to_shadow

BASELINES = ['long', 'mid', 'short']
HPSOS = ['hpso01', 'hpso02a', 'hpso02b', 'hpso04a', 'hpso05a']

run_dir = Path.cwd()
base_dir = Path(f"config_generation/hpso_baslines")
for baseline in BASELINES:
    for hpso in HPSOS:
        hpso_path = f"single_{hpso}.json"
        LOGGER.info("Starting config generation...")
        output_path = Path(f"baseline/hpso")
        if not output_path.exists():
            output_path.mkdir(parents=True,exist_ok=True)
    component_sizing = Path(common.COMPONENT_SIZING_LOW)
    total_sizing = Path(common.TOTAL_SIZING_LOW)
    SD
    timestep = "seconds"
    data = False

    cfg_path = create_config(
        common.MAX_TEL_DEMAND,
        hpso_path,
        output_path,
        component_sizing,
        total_sizing,
        sdp,
        timestep,
        data=data
    )
