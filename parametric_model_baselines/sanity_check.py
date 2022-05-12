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

from parametric_model_baselines.generate_data import (
    LOW_TOTAL_SIZING, MID_TOTAL_SIZING
)

# low_base

with LOW_HPSO_PATH.open('r') as f:
    hpso_dict = json.load(f)

low_base_dir = Path("parametric_model_baselines/low_base")
low_base_workflows_dir = low_base_dir / "workflows"
config = low_base_dir / "low_sdp_config.json"

hpso_list = hpso_dict['items']
print(f" HPSO | Expected | Actual")
for file in low_base_workflows_dir.iterdir():
    componenents = file.name.split("_")
    hpso_id = componenents[0]
    hpso_data = hpso_list[0]
    for observation in hpso_list:
        if hpso_id == observation["hpso"]:
            hpso_data = observation
            break
    expected = wa.calculate_expected_flops(
        hpso_id,
        hpso_data["workflows"],
        hpso_data["duration"],
        LOW_TOTAL_SIZING,
        hpso_data["baseline"]
    )

    actual = wa.calculate_total_flops(file)

    print(f"{hpso_id}| {expected} | {actual}")

low_par_dir = Path("parametric_model_baselines/low_parallel")
low_par_workflows_dir = low_par_dir / "workflows"
config = low_base_dir / "low_sdp_config.json"

hpso_list = hpso_dict['items']
print(f" HPSO | Expected | Actual")
for file in low_par_workflows_dir.iterdir():
    componenents = file.name.split("_")
    hpso_id = componenents[0]
    hpso_data = hpso_list[0]
    for observation in hpso_list:
        if hpso_id == observation["hpso"]:
            hpso_data = observation
            break
    expected = wa.calculate_expected_flops(
        hpso_id,
        hpso_data["workflows"],
        hpso_data["duration"],
        LOW_TOTAL_SIZING,
        hpso_data["baseline"]
    )

    actual = wa.calculate_total_flops(file)

    print(f"{hpso_id}| {expected} | {actual}")

with MID_HPSO_PATH.open('r') as f:
    hpso_dict = json.load(f)

mid_base_dir = Path("parametric_model_baselines/mid_base")
mid_base_workflows_dir = mid_base_dir / "workflows"
config = mid_base_dir / "mid_sdp_config.json"

hpso_list = hpso_dict['items']

for file in mid_base_workflows_dir.iterdir():
    componenents = file.name.split("_")
    hpso_id = componenents[0]
    hpso_data = hpso_list[0]
    for observation in hpso_list:
        if hpso_id == observation["hpso"]:
            hpso_data = observation
            break
    expected = wa.calculate_expected_flops(
        hpso_id,
        hpso_data["workflows"],
        hpso_data["duration"],
        MID_TOTAL_SIZING,
        hpso_data["baseline"]
    )

    actual = wa.calculate_total_flops(file)

    print(f"{hpso_id}| {expected} | {actual}")


mid_par_dir = Path("parametric_model_baselines/mid_par")
mid_par_workflows_dir = mid_par_dir / "workflows"
hpso_list = hpso_dict['items']
for file in mid_base_workflows_dir.iterdir():
    componenents = file.name.split("_")
    hpso_id = componenents[0]
    hpso_data = hpso_list[0]
    for observation in hpso_list:
        if hpso_id == observation["hpso"]:
            hpso_data = observation
            break
    expected = wa.calculate_expected_flops(
        hpso_id,
        hpso_data["workflows"],
        hpso_data["duration"],
        MID_TOTAL_SIZING,
        hpso_data["baseline"]
    )

    actual = wa.calculate_total_flops(file)

    print(f"{hpso_id}| {expected} | {actual}")
