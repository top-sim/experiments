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

"""
Some baseline scheduling comparisons for workflows

Maximal channels will be 896 for LOW, 786 for MID for both the completely
parallel workflow, which is representative of the parametric scheduling model.

4 experiments exist here, with data generated for each accordingly.
"""

import logging
import os
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
# HPSO_PLANS = [
#     "chapter4/scalability/simulationA.json",
# ]

HPSO_PLANS = os.listdir("chapter4/scalability/simfiles/mid")

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

low_path = BASE_DIR / "chapter4/scalability" / "low_maximal"
mid_path_str = BASE_DIR / "chapter4/scalability" / "mid_maximal"
start = time.time()

telescope = "mid" # TODO update this so it's not necessary in the config generation call
infrastructure_style = "parametric"
nodes = 896 # TODO update this so it's not necessary in the config generation call
timestep = 5
data = [False]
data_distribution = ["standard"] # ["edges"] #, "standard"]
overwrite = False

for hpso in HPSO_PLANS:
    for d in data:
        for dist in data_distribution:
            config_generator.create_config(
                telescope=telescope,
                infrastructure=infrastructure_style,
                nodes=nodes,
                hpso_path=Path(__file__).parent / 'simfiles/mid' / Path(hpso),
                output_dir=Path(low_path / graph_type),
                base_graph_paths=WORKFLOW_TYPE_MAP,
                timestep=timestep,
                data=d,
                data_distribution=dist,
            )

finish = time.time()

LOGGER.info(f"Total generation time was: {(finish-start)/60} minutes")
