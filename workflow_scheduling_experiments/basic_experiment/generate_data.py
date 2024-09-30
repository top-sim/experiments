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
import time
from pathlib import Path
import sys

sys.path.append("/home/rwb/github/skaworkflows")

import skaworkflows.config_generator as config_generator


logging.basicConfig(level="INFO")
LOGGER = logging.getLogger(__name__)


# Either "prototype" or "scatter"
graph_type = sys.argv[1]

# HPSOs observation plans
HPSO_PLANS = [
    # "chapter3/observation_plans/maximal_low_imaging_896channels.json",
    # "chapter3/observation_plans/maximal_low_imaging_896channels_256Antennas.json",
    # "chapter3/observation_plans/maximal_low_imaging_896channels_128Antennas.json",
    # "chapter3/observation_plans/maximal_low_imaging_512channels.json",
    "chapter3/observation_plans/maximal_mid_imaging_786channels.json",
]


BASE_DIR = Path("/home/rwb/Dropbox/University/PhD/experiment_data/")
# {graph_type}_WORKFLOW_PATHS = {"DPrepA": graph_type, "DPrepB": graph_type}

WORKFLOW_TYPE_MAP = {
    "ICAL": graph_type,
    "DPrepA": graph_type,
    "DPrepB": graph_type,
    "DPrepC": graph_type,
    "DPrepD": graph_type,
}

SCATTER_WORKFLOW_TYPE_MAP = {
    "ICAL": "scatter",
    "DPrepA": "scatter",
    "DPrepB": "scatter",
    "DPrepC": "scatter",
    "DPrepD": "scatter",
}

LOW_CONFIG = Path("low_sdp_config")
MID_CONFIG = Path("mid_sdp_config")

low_path = BASE_DIR / "chapter3/results_with_metadata" / "low_maximal"
mid_path_str = BASE_DIR / "chapter3/results_with_metadata" / "mid_maximal"
start = time.time()

telescope = "mid"
infrastructure_style = "parametric"
nodes = 786
timestamp = "seconds"
data = [False, True]
data_distribution = ["standard"] #, "edges"]
overwrite = True

for hpso in HPSO_PLANS:
    for d in data:
        for dist in data_distribution:
            config_generator.create_config(
                telescope=telescope,
                infrastructure=infrastructure_style,
                nodes=nodes,
                hpso_path=Path(hpso),
                output_dir=Path(low_path / graph_type),
                base_graph_paths=WORKFLOW_TYPE_MAP,
                data=d,
                data_distribution=dist,
            )

# config_generator.create_config(
#     'low', 'parametric', nodes=896, hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
#                                cfg_name=f"{LOW_CONFIG}_{graph_type}_n896_896channels.json", base_graph_paths=WORKFLOW_PATHS,
#                                timestep="seconds", data=False, overwrite=True)
#
# config_generator.create_config('low', 'parametric', nodes=896,
#     hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
#     cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels.json"), base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=False, overwrite=True, data_distribution='edges')
#
# config_generator.create_config('low', 'parametric', nodes=896,
#     hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
#     cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels.json"), base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=True, overwrite=True, data_distribution='edges')
#
#
# config_generator.create_config('low', parametric', nodes=896,
#     hpso_path=Path(hpso_path_896_256), output_dir=Path(low_path_str)/graph_type,
#     cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels_256Antennas.json"), base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=True, overwrite=True, data_distribution='edges')
#
# config_generator.create_config('low', 'parametric', nodes=896,
#     hpso_path=Path(hpso_path_896_128), output_dir=Path(low_path_str)/graph_type,
#     cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels_128Antennas.json"), base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=True, overwrite=True, data_distribution='edges')


# #
# # config_generator.create_config('low', 'parametric', nodes=896,
# #     hpso_path=Path(hpso_path_512), output_dir=Path(low_path_str)/'{graph_type}',
# #     cfg_name=f"{LOW_CONFIG}_{graph_type}_n{896}_512channels.json", b0ase_graph_paths={graph_type}_WORKFLOW_PATHS,
# #     timestep="seconds", data=False)
# #
# config_generator.create_config('low', 'parametric', nodes=896,
#     hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
#     cfg_name=f"{LOW_CONFIG}_{graph_type}_n{896}_896channels.json", base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=True)
#
# config_generator.create_config('low', 'parametric', nodes=896,
#     hpso_path=Path(hpso_path_512), output_dir=Path(low_path_str)/'{graph_type}',
#     cfg_name=f"{LOW_CONFIG}_{graph_type}_n{896}_512channels.json", base_graph_paths={graph_type}_WORKFLOW_PATHS,
#     timestep="seconds", data=True)
#
# config_generator.create_config('low', 'parametric', 896,
#     hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/'scatter',
#     cfg_name=f"{LOW_CONFIG}_scatter_n{896}_896channels.json", base_graph_paths=SCATTER_WORKFLOW_PATHS,
#     timestep="seconds", data=False)
#
# config_generator.create_config('low', 'parametric', 896,
#     hpso_path=Path(hpso_path_512), output_dir=Path(low_path_str)/'scatter',
#     cfg_name=f"{LOW_CONFIG}_scatter_n{896}_512channels.json", base_graph_paths=SCATTER_WORKFLOW_PATHS,
#     timestep="seconds", data=False)
#
# # SKA MID

# config_generator.create_config('mid', 'parametric', nodes=786,
#     hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str)/graph_type,
#     cfg_name=f"{MID_CONFIG}_{graph_type}_n786_786channels.json", base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=False)
# #
# config_generator.create_config('mid', 'parametric', nodes=786,
#     hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str)/graph_type,
#     cfg_name=f"{MID_CONFIG}_{graph_type}_n786_786channels.json", base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=False, overwrite=True, data_distribution="edges")
# #
# config_generator.create_config('mid', 'parametric', nodes=786,
#     hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str)/graph_type,
#     cfg_name=f"{MID_CONFIG}_{graph_type}_n786_786channels.json", base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=True, overwrite=True, data_distribution="edges")
# #
#
# config_generator.create_config('mid', 'parametric', nodes=786,
#     hpso_path=Path(hpso_path_mid_512), output_dir=Path(mid_path_str)/graph_type,
#     cfg_name=f"{MID_CONFIG}_{graph_type}_n{896}_896channels.json", base_graph_paths=WORKFLOW_PATHS,
#     timestep="seconds", data=False)
#
# config_generator.create_config('mid', 'parametric', nodes=786,
#                                hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str) / graph_type,
#                                cfg_name=f"{MID_CONFIG}_{graph_type}_n786_896channels.json", base_graph_paths=WORKFLOW_PATHS,
#                                timestep="seconds", data=True)
#
# config_generator.create_config('mid', 'parametric', nodes=786,
#     hpso_path=Path(hpso_path_mid_512), output_dir=Path(mid_path_str)/'{graph_type}',
#     cfg_name=f"{MID_CONFIG}_{graph_type}_n{896}_512channels.json", base_graph_paths={graph_type}_WORKFLOW_PATHS,
#     timestep="seconds", data=True)
#
# config_generator.create_config('mid', 'parametric', 786,
#     hpso_path=Path(hpso_path_mid_896), output_dir=Path(mid_path_str)/'scatter',
#     cfg_name=f"{MID_CONFIG}_scatter_n{896}_896channels.json", base_graph_paths=SCATTER_WORKFLOW_PATHS,
#     timestep="seconds", data=False)
#
# config_generator.create_config('mid', 'parametric', 786,
#     hpso_path=Path(hpso_path_mid_512), output_dir=Path(mid_path_str)/'scatter',
#     cfg_name=f"{MID_CONFIG}_scatter_n{896}_512channels.json", base_graph_paths=SCATTER_WORKFLOW_PATHS,
#     timestep="seconds", data=False)
# #
finish = time.time()

LOGGER.info(f"Total generation time was: {(finish-start)/60} minutes")
