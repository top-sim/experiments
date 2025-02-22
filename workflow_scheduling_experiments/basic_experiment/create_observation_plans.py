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

import random
from pathlib import Path
import logging
import argparse

import numpy as np

VERBOSE = False
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

from skaworkflows.config_generator import create_config

LOW_OBSERVATIONS= {'hpso01': {"duration": 18000,
                  'workflows': ["ICAL", "DPrepA", "DPrepB",  "DPrepC", "DPrepD"]},
       'hpso02a': {"duration": 18000,
                   'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"]},
       'hpso02b': {"duration": 18000,
                   'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"]}}

# TODO These defaults really should be stored in SKAWorkflows and referenced exclusively
# there until the end of time. Bonus points for wrapping the SDP parametric model

MID_OBSERVATIONS= {
    'hpso13': {'duration': 28800,
               'baseline': 35000,
               'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso15': {'duration': 15840,
               'baseline': 15000,
               'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso22': {'duration': 28800,
               'baseline': 150000,
               'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso32': {'duration': 7920,
               'baseline': 20000,
               'workflows': ["ICAL", "DPrepB"]}
}

channel_multiplier = 128


def maximal_low_obs_plan():
    """
    Generate a plan for a single observation

    This is purely to validate and demonstrate the scheduling heuristic performance.

    Returns
    -------
    plans: list of json-compliant dictionaries, representing observation plans
    """
    nodes = 896
    max_demand = 512
    max_channels = 256 # Max for low
    max_baseline = 65000
    params = []
    LOGGER.info("Preparing maximal output for LOW telescope\n"
                "Nodes: %i\n Stations/Antennas:%i\nChannels: %s\n Baseline: %ikm",
                nodes, max_demand, max_channels, max_baseline)
    observation = {
        "nodes": nodes,
        "infrastructure": "parametric",
        "telescope": "low",
        "items": [
            {
                "count": 1,
                "hpso": "hpso01",
                "duration": LOW_OBSERVATIONS['hpso01']['duration'],
                "workflows": LOW_OBSERVATIONS['hpso01']['workflows'],
                "demand": max_demand,  # demand * 1,
                "channels": max_demand * channel_multiplier,
                "coarse_channels": nodes,  # parallel channels,
                "baseline": max_baseline,
                "telescope": "low"
            },
            {
                "count": 1,
                "hpso": "hpso02a",
                "duration": LOW_OBSERVATIONS['hpso02a']['duration'],
                "workflows": LOW_OBSERVATIONS['hpso02a']['workflows'],
                "demand": max_demand,  # demand * 1,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,  # parallel channels,
                "baseline": max_baseline,
                "telescope": "low"
            },
            {
                "count": 1,
                "hpso": "hpso02a",
                "duration": LOW_OBSERVATIONS['hpso02b']['duration'],
                "workflows": LOW_OBSERVATIONS['hpso02b']['workflows'],
                "demand": max_demand,  # demand * 1,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,  # parallel channels,
                "baseline": max_baseline,
                "telescope": "low"
            }
        ]
    }
    params.append(observation)

    return params


def maximal_mid_obs_plan():
    """
    Currently, this is a placeholder method to generate one of a couple different
    observation plans.

    Expect this method to be a) renamed in the future and b) improved upon

    'hpso13': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso15': {'duration': 15840, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso32': {'duration': 7920, 'workflows': ["ICAL", "DPrepB"]}


    Returns
    -------

    """

    nodes = 796
    max_demand = 197
    max_channels = 512 # Max for mid
    params = []
    LOGGER.info("Preparing maximal output for LOW telescope\n"
                "Nodes: %i\n Stations/Antennas:%i\nChannels: %s\n Baseline: Various",
                nodes, max_demand, max_channels)
    observation = {
        "nodes": nodes,
        "infrastructure": "parametric",
        "telescope": "mid",
        "items": [
            {
                "count": 1,
                "hpso": "hpso13",
                "duration": MID_OBSERVATIONS['hpso13']['duration'],
                "workflows": MID_OBSERVATIONS['hpso13']['workflows'],
                "demand": max_demand,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso13']['baseline'],
                "telescope": "mid"
            },
            {
                "count": 1,
                "hpso": 'hpso15',
                "duration": MID_OBSERVATIONS['hpso15']['duration'],
                "workflows": MID_OBSERVATIONS['hpso15']['workflows'],
                "demand": max_demand,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso15']['baseline'],
                "telescope": "mid"
            },
            {
                "count": 1,
                "hpso": 'hpso22',
                "duration": MID_OBSERVATIONS['hpso22']['duration'],
                "demand": max_demand
                ,
                "workflows": MID_OBSERVATIONS['hpso22']['workflows'],
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso22']['baseline'],
                "telescope": "mid"
            },
            {
                "count": 1,
                "hpso": 'hpso32',
                "duration": MID_OBSERVATIONS['hpso32']['duration'],
                "demand": max_demand,
                "workflows": MID_OBSERVATIONS['hpso32']['workflows'],
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso32']['baseline'],
                "telescope": "mid"
            }
        ]
    }

    params.append(observation)
    return params

def minimal_low_obs_plan():
    """
    Used for debugging and comparisons
    Returns
    -------

    """
    nodes = 896
    max_demand = 64
    max_channels = 64 # Max for low
    max_baseline = 65000 # currently don't have data generated for lower baseline
    params = []
    LOGGER.info("Preparing maximal output for LOW telescope\n"
                "Nodes: %i\n Stations/Antennas:%i\nChannels: %s\n Baseline: %ikm",
                nodes, max_demand, max_channels, max_baseline)
    observation = {
        "nodes": nodes,
        "infrastructure": "parametric",
        "telescope": "low",
        "items": [
            {
                "count": 1,
                "hpso": "hpso01",
                "duration": LOW_OBSERVATIONS['hpso01']['duration'],
                "workflows": LOW_OBSERVATIONS['hpso01']['workflows'],
                "demand": max_demand,  # demand * 1,
                "channels": max_demand * channel_multiplier,
                "coarse_channels": max_channels,  # parallel channels,
                "baseline": max_baseline,
                "telescope": "low"
            },
            {
                "count": 1,
                "hpso": "hpso02a",
                "duration": LOW_OBSERVATIONS['hpso02a']['duration'],
                "workflows": LOW_OBSERVATIONS['hpso02a']['workflows'],
                "demand": max_demand,  # demand * 1,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": max_channels,  # parallel channels,
                "baseline": 65000.0,
                "telescope": "low"
            },
            {
                "count": 1,
                "hpso": "hpso02b",
                "duration": LOW_OBSERVATIONS['hpso02b']['duration'],
                "workflows": LOW_OBSERVATIONS['hpso02b']['workflows'],
                "demand": max_demand,  # demand * 1,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": max_channels,  # parallel channels,
                "baseline": 65000.0,
                "telescope": "low"
            }
        ]
    }
    params.append(observation)

    return params

def minimal_mid_obs_plan():
    """
    Currently, this is a placeholder method to generate one of a couple different
    observation plans.

    Expect this method to be a) renamed in the future and b) improved upon

    'hpso13': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso15': {'duration': 15840, 'workflows': ["ICAL", "DPrepA", "DPrepB", "DPrepC"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso22': {'duration': 28800, 'workflows': ["ICAL", "DPrepA", "DPrepB"]},
    'hpso32': {'duration': 7920, 'workflows': ["ICAL", "DPrepB"]}


    Returns
    -------

    """

    nodes = 796
    max_demand = 512
    max_channels = 512 # Max for mid
    params = []
    LOGGER.info("Preparing maximal output for LOW telescope\n"
                "Nodes: %i\n Stations/Antennas:%i\nChannels: %s\n Baseline: Various",
                nodes, max_demand, max_channels)
    observation = {
        "nodes": nodes,
        "infrastructure": "parametric",
        "telescope": "mid",
        "items": [
            {
                "count": 1,
                "hpso": "hpso13",
                "duration": MID_OBSERVATIONS['hpso13']['duration'],
                "workflows": MID_OBSERVATIONS['hpso13']['workflows'],
                "demand": max_demand,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso013']['baseline'],
                "telescope": "mid"
            },
            {
                "count": 1,
                "hpso": 'hpso15',
                "duration": MID_OBSERVATIONS['hpso15']['duration'],
                "workflows": MID_OBSERVATIONS['hpso15']['workflows'],
                "demand": max_demand,
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso15']['baseline'],
                "telescope": "mid"
            },
            {
                "count": 1,
                "hpso": 'hpso22',
                "duration": MID_OBSERVATIONS['hpso22']['duration'],
                "demand": max_demand
                ,
                "workflows": MID_OBSERVATIONS['hpso22']['workflows'],
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpos22']['baseline'],
                "telescope": "mid"
            },
            {
                "count": 1,
                "hpso": 'hpso32',
                "duration": MID_OBSERVATIONS['hpso32']['duration'],
                "demand": max_demand,
                "workflows": MID_OBSERVATIONS['hpso32']['workflows'],
                "channels": max_channels * channel_multiplier,
                "coarse_channels": nodes,
                "baseline": MID_OBSERVATIONS['hpso32']['baseline'],
                "telescope": "mid"
            }
        ]
    }

    params.append(observation)
    return params


if __name__ == '__main__':

    parser = argparse.ArgumentParser(Path(__file__).name,)
    parser.add_argument('path')
    parser.add_argument('telescope', help="Choose from 'low', 'mid', or 'all")
    parser.add_argument('graph_type', help="prototype, parallel")
    parser.add_argument('scale', help="maximal, minimal")

    args = parser.parse_args()

    WORKFLOW_TYPE_MAP = {
        "ICAL": args.graph_type,
        "DPrepA": args.graph_type,
        "DPrepB": args.graph_type,
        "DPrepC": args.graph_type,
        "DPrepD": args.graph_type,
    }

    all_params = []
    if args.scale == 'minimal':
        all_params.append(minimal_low_obs_plan())
        all_params.append(minimal_mid_obs_plan())
    elif args.telescope == 'low':
        logging.info("Creating plan for LOW telescope.")
        all_params.append(maximal_low_obs_plan())
    elif args.telescope == 'mid':
        logging.info("Creating plan for MID telescope.")
        all_params.append(maximal_mid_obs_plan())
    else:
        logging.info("Creating plans for all telescopes")
        all_params.append(maximal_low_obs_plan())
        all_params.append(maximal_mid_obs_plan())

    low_path = Path(args.path) / args.telescope

    print("Creating config")
    for ap in all_params:
        for plan in ap:
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=False,
                data_distribution='standard',
                multiple_plans=False)
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=True,
                data_distribution='standard',
                multiple_plans=False)
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=True,
                data_distribution='edges',
                multiple_plans=False)
