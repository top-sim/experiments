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
    "chapter5/simulationBase.json",
    "chapter5/simulationAltered.json"
]


BASE_DIR = Path("/home/rwb/Dropbox/University/PhD/experiment_data/")

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

low_path = BASE_DIR / "chapter5/" / "low"
mid_path_str = BASE_DIR / "chapter5/" / "mid"
start = time.time()

telescope = "low" # TODO update this so it's not necessary in the config generation call
infrastructure_style = "parametric"
nodes = 256  # TODO update this so it's not necessary in the config generation call
timestamp = 5
data = [True]
data_distribution = ["edges"]
overwrite = False

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
                timestep=5,
                data=d,
                data_distribution=dist,
            )

finish = time.time()

LOGGER.info(f"Total generation time was: {(finish-start)/60} minutes")
