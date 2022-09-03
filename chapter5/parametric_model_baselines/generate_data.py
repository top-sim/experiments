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

import skaworkflows.config_generator as config_generator

logging.basicConfig(level='INFO')
LOGGER = logging.getLogger(__name__)

# Imaging HPSOs
LOW_HPSO_PATHS = [
    "chapter5/parametric_model_baselines/maximal_low_imaging_512channels.json",
    "chapter5/parametric_model_baselines/maximal_low_imaging_896channels.json"]
MID_HPSO_PATHS = [
    "chapter5/parametric_model_baselines/maximal_mid_imaging_512channels.json",
    "chapter5/parametric_model_baselines/maximal_mid_imaging_786channels.json"]


BASE_DIR = Path(f"chapter5/parametric_model_baselines")

PROTOTYPE_WORKFLOW_PATHS = {"ICAL": "prototype", "DPrepA": "prototype",
                            "DPrepB": "prototype", "DPrepC": "prototype",
                            "DPrepD": "prototype"}

SCATTER_WORKFLOW_PATHS = {"ICAL": "scatter", "DPrepA": "scatter",
                          "DPrepB": "scatter", "DPrepC": "scatter",
                          "DPrepD": "scatter"}

LOW_CONFIG = Path("low_sdp_config")
MID_CONFIG = Path("mid_sdp_config")

# Experiments
data_iterations = [False, True]
graph_iterations = ['prototype', 'scatter']


def generate_low(config_iterations, channel_iterations):
    LOGGER.info("config,channel,graph,data")
    for data in data_iterations:
        for config in config_iterations:
            for channel in channel_iterations:
                if config == 512 and channel == 896:
                    continue
                low_hpso_str = next(
                    x for x in LOW_HPSO_PATHS if f'{channel}' in x)
                for graph in graph_iterations:
                    if graph == 'prototype':
                        base_graph_paths = PROTOTYPE_WORKFLOW_PATHS
                    else:
                        base_graph_paths = SCATTER_WORKFLOW_PATHS
                    low_path_str = (
                        f'{BASE_DIR}/low_{graph}/c{config}/n{channel}')
                    # Generate configuration with prototype SKA Workflow
                    LOGGER.info("%s,s,%s,%s,%s,%s,%s", low_hpso_str, config,
                                channel, graph, data, low_path_str)
                    config_generator.create_config(
                        'low', 'parametric', config,
                        hpso_path=Path(low_hpso_str),
                        output_dir=Path(low_path_str),
                        cfg_name=f'{LOW_CONFIG}_n{config}',
                        base_graph_paths=base_graph_paths,
                        timestep='seconds',
                        data=data
                    )


def generate_mid(config_iterations, channel_iterations):
    LOGGER.info("config,channel,graph,data")
    for data in data_iterations:
        for config in config_iterations:
            for channel in channel_iterations:
                if config == 512 and channel == 786:
                    continue
                mid_hpso_str = next(
                    x for x in MID_HPSO_PATHS if f'{channel}' in x)
                for graph in graph_iterations:
                    if graph == 'prototype':
                        base_graph_paths = PROTOTYPE_WORKFLOW_PATHS
                    else:
                        base_graph_paths = SCATTER_WORKFLOW_PATHS
                    mid_path_str = (
                        f'{BASE_DIR}/mid_{graph}/c{config}/n{channel}')
                    config_generator.create_config(
                       'mid', 'parametric', config,
                       hpso_path=Path(mid_hpso_str),
                       output_dir=Path(mid_path_str),
                       cfg_name=f'{MID_CONFIG}_n{config}',
                       base_graph_paths=base_graph_paths,
                       timestep='seconds', data=data
                    )


if __name__ == '__main__':
    low_config_iterations = [896, 512]
    low_channel_iterations = [896, 512]
    generate_low(low_config_iterations, low_channel_iterations)
    mid_config_iterations = [786, 512]
    mid_channel_iterations = [786, 512]
    generate_mid(mid_config_iterations, mid_channel_iterations)
