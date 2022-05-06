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
Cross-compare the sum of compute in our generated workflows with the
parametric model values
"""

import json
from pathlib import Path

import skaworkflows.workflow.workflow_analysis as wa

from parametric_model_baselines.generate_data import (
    LOW_HPSO_PATH, MID_HPSO_PATH
)

# low_base

with LOW_HPSO_PATH.open('r') as f:
    hpso_data = json.load(f)

low_base_dir = Path("parametric_model_baselines/low_base")
low_base_workflows_dir = low_base_dir / "workflows"
config = low_base_dir / "low_sdp_config.json"

for file in low_base_workflows_dir:
    continue

MID_HPSO_PATH = Path("parametric_model_baselines/maximal_mid_imaging.json")
