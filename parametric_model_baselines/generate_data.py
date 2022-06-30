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
"""


from pathlib import Path

import skaworkflows.config_generator as con_gen
from skaworkflows.hpconfig.specs.sdp import (
    SDP_PAR_MODEL_LOW,
    SDP_PAR_MODEL_MID
)


# Imaging HPSOs
LOW_HPSO_PATH = Path("parametric_model_baselines/maximal_low_imaging.json")
MID_HPSO_PATH = Path("parametric_model_baselines/maximal_mid_imaging.json")

# Maximum telescope stations
LOW_MAX = 512
MID_MAX = 197

# Parametric model sizing
LOW_TOTAL_SIZING = Path(
    f"../skaworkflows/skaworkflows/data/pandas_sizing/"
    f"total_compute_SKA1_Low.csv"
)

LOW_COMPONENT_SIZING = Path(
    f"../skaworkflows/skaworkflows/data/pandas_sizing/"
    f"component_compute_SKA1_Low.csv"
)

MID_TOTAL_SIZING = Path(
    f"../skaworkflows/skaworkflows/data/pandas_sizing/"
    f"total_compute_SKA1_Mid.csv"
)

MID_COMPONENT_SIZING = Path(
    f"../skaworkflows/skaworkflows/data/pandas_sizing/"
    f"component_compute_SKA1_Mid.csv"
)

SKA_LOW_SDP = SDP_PAR_MODEL_LOW()
SKA_LOW_SDP.set_nodes(512)
SKA_MID_SDP = SDP_PAR_MODEL_MID()

# Output directories
LOW_OUTPUT_DIR = Path(f"parametric_model_baselines/low_base")
MID_OUTPUT_DIR = Path(f"parametric_model_baselines/mid_base")

LOW_OUTPUT_DIR_PAR = Path(f"parametric_model_baselines/low_parallel")
MID_OUTPUT_DIR_PAR = Path(f"parametric_model_baselines/mid_parallel")


BASE_GRAPH = Path("../skaworkflows/skaworkflows/data/hpsos/dprepa.graph")
PARALLEL_BASE_GRAPH = Path(
    "../skaworkflows/skaworkflows/data/hpsos/dprepa_parallel.graph"
)

WORKFLOW_PATHS = {
    "ICAL": BASE_GRAPH,
    "DPrepA": BASE_GRAPH,
    "DPrepB": BASE_GRAPH,
    "DPrepC": BASE_GRAPH,
    "DPrepD": BASE_GRAPH
}

LOW_CONFIG = "low_sdp_config.json"
MID_CONFIG = "mid_sdp_config.json"

# Generate configuration with prototype SKA Workflow
con_gen.create_config(
    telescope_max=LOW_MAX,
    hpso_path=LOW_HPSO_PATH,
    output_dir=LOW_OUTPUT_DIR,
    cfg_name=LOW_CONFIG,
    component=LOW_COMPONENT_SIZING,
    system=LOW_TOTAL_SIZING,
    cluster=SKA_LOW_SDP,
    base_graph_paths=WORKFLOW_PATHS,
    timestep='seconds',
    data=False,
)

con_gen.create_config(
    telescope_max=MID_MAX,
    hpso_path=MID_HPSO_PATH,
    output_dir=MID_OUTPUT_DIR,
    cfg_name=MID_CONFIG,
    component=MID_COMPONENT_SIZING,
    system=MID_TOTAL_SIZING,
    cluster=SKA_MID_SDP,
    base_graph_paths=WORKFLOW_PATHS,
    timestep='seconds',
    data=False,
)

PARALLEL_WORKFLOW_PATHS = {
    "ICAL": PARALLEL_BASE_GRAPH,
    "DPrepA": PARALLEL_BASE_GRAPH,
    "DPrepB": PARALLEL_BASE_GRAPH,
    "DPrepC": PARALLEL_BASE_GRAPH,
    "DPrepD": PARALLEL_BASE_GRAPH,
}


# Generate configuration with SDP equivalent base graph (entirely parallel)
con_gen.create_config(
    telescope_max=LOW_MAX,
    hpso_path=LOW_HPSO_PATH,
    output_dir=LOW_OUTPUT_DIR_PAR,
    cfg_name=LOW_CONFIG,
    component=LOW_COMPONENT_SIZING,
    system=LOW_TOTAL_SIZING,
    cluster=SKA_LOW_SDP,
    base_graph_paths=PARALLEL_WORKFLOW_PATHS,
    timestep='seconds',
    data=False,
)

con_gen.create_config(
    telescope_max=MID_MAX,
    hpso_path=MID_HPSO_PATH,
    output_dir=MID_OUTPUT_DIR_PAR,
    cfg_name=MID_CONFIG,
    component=MID_COMPONENT_SIZING,
    system=MID_TOTAL_SIZING,
    cluster=SKA_MID_SDP,
    base_graph_paths=PARALLEL_WORKFLOW_PATHS,
    timestep='seconds',
    data=False,
)
