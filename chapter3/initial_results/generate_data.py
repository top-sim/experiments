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
import skaworkflows.config_generator as config_generator

sys.path.append('/home/rwb/github/skaworkflows')

logging.basicConfig(level='INFO')
LOGGER = logging.getLogger(__name__)

# Imaging HPSOs

graph_type = sys.argv[1]

hpso_path_896 = (
    "chapter3/observation_plans/maximal_low_imaging_896channels.json"
)

hpso_path_512 = (
    "chapter3/observation_plans/maximal_low_imaging_512channels.json"
)

hpso_path_mid_786 = (
    "chapter3/observation_plans/maximal_mid_imaging_786channels.json"
)

hpso_path_mid_512 = (
    "chapter3/observation_plans/maximal_mid_imaging_512channels.json"
)

hpso_maximal_single = (
    "chapter3/observation_plans/maximal_imaging_single_896channels.json"
)


BASE_DIR = Path("/home/rwb/Dropbox/University/PhD/experiment_data/")
# {graph_type}_WORKFLOW_PATHS = {"DPrepA": graph_type, "DPrepB": graph_type}
WORKFLOW_PATHS = {"ICAL": graph_type, "DPrepA": graph_type,
                            "DPrepB": graph_type, "DPrepC": graph_type,
                            "DPrepD": graph_type}

SCATTER_WORKFLOW_PATHS = {"ICAL": "scatter", "DPrepA": "scatter",
                          "DPrepB": "scatter", "DPrepC": "scatter",
                          "DPrepD": "scatter"}

LOW_CONFIG = Path("low_sdp_config")
MID_CONFIG = Path("mid_sdp_config")

# TODO we are going to use the file-based config generation for this

# base_graph_paths = {graph_type}_WORKFLOW_PATHS
low_path_str = BASE_DIR / "chapter3/initial_results" / 'low_maximal'
mid_path_str = BASE_DIR / "chapter3/initial_results" / 'mid_maximal'
# Generate configuration with {graph_type} SKA Workflow
# SKA LOW
start = time.time()

config_generator.create_config('low', 'parametric', nodes=896,
    hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
    cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels.json"), base_graph_paths=WORKFLOW_PATHS,
    timestep="seconds", data=False, overwrite=True)

config_generator.create_config('low', 'parametric', nodes=896,
    hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
    cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels.json"), base_graph_paths=WORKFLOW_PATHS,
    timestep="seconds", data=False, overwrite=True, data_distribution='edges')

config_generator.create_config('low', 'parametric', nodes=896,
    hpso_path=Path(hpso_path_896), output_dir=Path(low_path_str)/graph_type,
    cfg_name=(f"{LOW_CONFIG}_{graph_type}_n896_896channels.json"), base_graph_paths=WORKFLOW_PATHS,
    timestep="seconds", data=True, overwrite=True, data_distribution='edges')


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

config_generator.create_config('mid', 'parametric', nodes=786,
    hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str)/graph_type,
    cfg_name=f"{MID_CONFIG}_{graph_type}_n786_786channels.json", base_graph_paths=WORKFLOW_PATHS,
    timestep="seconds", data=False)
#
config_generator.create_config('mid', 'parametric', nodes=786,
    hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str)/graph_type,
    cfg_name=f"{MID_CONFIG}_{graph_type}_n786_786channels.json", base_graph_paths=WORKFLOW_PATHS,
    timestep="seconds", data=False, overwrite=True, data_distribution="edges")
#
config_generator.create_config('mid', 'parametric', nodes=786,
    hpso_path=Path(hpso_path_mid_786), output_dir=Path(mid_path_str)/graph_type,
    cfg_name=f"{MID_CONFIG}_{graph_type}_n786_786channels.json", base_graph_paths=WORKFLOW_PATHS,
    timestep="seconds", data=True, overwrite=True, data_distribution="edges")
#
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