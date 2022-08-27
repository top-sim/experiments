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


from chapter5.parametric_model_baselines.generate_data import (
    LOW_HPSO_PATHS, MID_HPSO_PATHS
)

from chapter5.parametric_model_baselines.generate_data import (
    LOW_TOTAL_SIZING, MID_TOTAL_SIZING
)

# low_base

# with LOW_HPSO_PATH.open('r') as f:
#     hpso_dict = json.load(f)
# for path in LOW_HPSO_PATHS
BASE_DIR = Path(f"chapter5/parametric_model_baselines")

config_iterations = [896, 512]
channel_iterations = [896, 512]
data_iterations = [False, True]
graph_iterations = ['prototype', 'scatter']
for data in data_iterations:
    for config in config_iterations:
        for channel in channel_iterations:
            if config == 512 and channel == 896:
                # We are not interested in this experiment
                    continue
            low_hpso_str = next(
                x for x in LOW_HPSO_PATHS if f'{channel}' in x
            )
            mid_hpso_str = next(
                x for x in MID_HPSO_PATHS if f'{channel}' in x
            )
            with Path(low_hpso_str).open() as f:
                hpso_low_dict = json.load(f)
            with Path(mid_hpso_str).open() as f:
                hpso_mid_dict = json.load(f)
            for graph in graph_iterations:
                low_path_str = (f'{BASE_DIR}/low_{graph}/c{config}/n{channel}')
                mid_path_str = (f'{BASE_DIR}/mid_{graph}/c{config}/n{channel}')

                hpso_list = hpso_low_dict['items']
                print(f" HPSO | Expected | Actual")
                for file in (Path(low_path_str) / 'workflows').iterdir():
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
                    print(f"String {low_path_str}")
                    print(f"{hpso_id}| {expected} | {actual}")
                hpso_list = hpso_mid_dict['items']
                for file in (Path(mid_path_str)/ 'workflows').iterdir():
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


