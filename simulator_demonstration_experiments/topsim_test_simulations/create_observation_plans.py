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
This script creates debug/testing simulations for things like identifying things like:

- How long a single, maximal case simulation takes
- How to distrubute multiple simulations across SLURM Clusters
- How to divide experiments

"""

import os
import json
import random
import sys
from pathlib import Path
import logging

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

#
# These are the 'coarse-grained' channel values.
SKA_channels = [64, 128, 256]
# SKA_low_channels = [64, 128, 256, ]

# 32 is an arbitrary minimum; 512 hard maximum
SKA_Low_antenna = [64, 128, 256, 512]
# 64 + N, where 64 is the base number of dishes
SKA_Mid_antenna = [64, 102, 140, 197]

channel_multiplier = 128
def permute_low_observation_plans_minimal():
    """
    Using the lowest SKA LOW configurations as a starting point, modify the observation
    plan
    based on the parameters, in order to generate a configuration that targets a particular
    telescope demand.

    Returns
    -------

    """
    hpso_idx = ['hpso01', 'hpso02a', 'hpso02b']  # {'hpso01': 0, 'hpso2': 1, 'hpso3': 2}
    hpso_demand = {'hpso01': 64, 'hpso02a': 64, 'hpso02b': 64}
    hpso_observations = {'hpso01': 1, 'hpso02a': 2, 'hpso02b': 2}
    default_observation_ratios = [1, 2, 2]
    n = 25
    max = 512
    print(len(hpso_demand))
    random.seed(1)
    final_set = {}
    for j in [64]:
        for i in range(0, len(hpso_demand)):
            idx = random.randint(0, len(hpso_demand) - 1)
            hpso_demand[hpso_idx[idx]] = j
            number_obs = np.array(default_observation_ratios) * n
            demand_ratio = sum(np.array(list(hpso_demand.values())) * number_obs) / (
                    sum(number_obs) * max)
            tmp = {}
            for hpso, demand in hpso_demand.items():
                tmp[hpso] = {'demand': demand,
                              'num_obs': number_obs[hpso_idx.index(hpso)]}
            final_set[demand_ratio] = tmp

    if VERBOSE:
        for comb, num in final_set.items():
            print(comb, num)

    return final_set


def simple_test_obs_plan_low():
    """
    Generate a plan where each HPSO is the only observation in the plan.

    This is purely to validate and demonstrate the scheduling heuristic performance.

    Returns
    -------
    plans: list of json-compliant dictionaries, representing observation plans
    """
    nodes = 896
    max_demand = 64
    max_channels = 64 # Max for low
    max_baseline = 65000 # currently don't have data generated for lower baseline
    params = []
    LOGGER.info("Preparing maximal output for LOW telescope\n"
                "Nodes: %i\n Stations/Antennas:%i\nChannels: %s\n Baseline: %ikm",
                nodes, max_demand, max_channels, max_baseline)

    for hpso in LOW_OBSERVATIONS:
        observation = {
            "nodes": nodes,
            "infrastructure": "parametric",
            "telescope": "low",
            "items": [
                {
                    "count": 1,
                    "hpso": hpso,
                    "duration": LOW_OBSERVATIONS[hpso]['duration'],
                    "workflows": LOW_OBSERVATIONS[hpso]['workflows'],
                    "demand": max_demand,  # demand * 1,
                    "channels": max_demand * channel_multiplier,
                    "coarse_channels": max_channels,  # parallel channels,
                    "baseline": max_baseline,
                    "telescope": "low"
                }
            ]
        }
        params.append(observation)

    return params


def standard_mid_obs_plan(verbose=False):
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

    count = 0
    params = []
    for channels in SKA_channels:
        for i in range(len(SKA_Mid_antenna) - 1):
            demand = SKA_Mid_antenna[i]
            alt_demand = SKA_Mid_antenna[i + 1]
            observation = {
                "nodes": 512,
                "infrastructure": "parametric",
                "telescope": "mid",
                "items": [
                    {
                        "count": 1,
                        "hpso": "hpso13",
                        "duration": MID_OBSERVATIONS['hpso13']['duration'],
                        "workflows": MID_OBSERVATIONS['hpso13']['workflows'],
                        "demand": demand,
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 2,
                        "hpso": 'hpso15',
                        "duration": MID_OBSERVATIONS['hpso15']['duration'],
                        "workflows": MID_OBSERVATIONS['hpso15']['workflows'],
                        "demand": alt_demand,
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 2,
                        "hpso": 'hpso22',
                        "duration": MID_OBSERVATIONS['hpso22']['duration'],
                        "demand": demand,
                        "workflows": MID_OBSERVATIONS['hpso22']['workflows'],
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": 4,
                        "hpso": 'hpso32',
                        "duration": MID_OBSERVATIONS['hpso32']['duration'],
                        "demand": demand,
                        "workflows": MID_OBSERVATIONS['hpso32']['workflows'],
                        "channels": channels * channel_multiplier,
                        "coarse_channels": channels,
                        "baseline": 65000.0,
                        "telescope": "low"
                    }
                ]
            }

            params.append(observation)
            # path_name = dir_name / 'low' / f"low_{count}_.json"
            # with path_name.open('w') as fp:
            #     json.dump(observation, fp, indent=2)
    if verbose:
        print(json.dumps(params, indent=2))
    return params


def test_time_for_month_observation():
    """
    Create a configuration for one month of observations (~120 observations at 6 hours)

    This is so we can determine how long this takes.

    Returns
    -------

    """
    nodes = 896
    LOW_MAX = 512

    count = 0
    params = []

    permutations = permute_low_observation_plans_minimal()
    channels_demand = 128

    for demand, hpso_numbers in permutations.items():
        observation = {
            "nodes": 896,
            "infrastructure": "parametric",
            "telescope": "low",
            "items": []
        }
        for hpso, items in hpso_numbers.items():
            observation['items'].append(
                {
                    "count": items['num_obs'],
                    "hpso": hpso,
                    "duration": LOW_OBSERVATIONS[hpso]['duration'],
                    "workflows": LOW_OBSERVATIONS[hpso]['workflows'],
                    "demand": items['demand'],
                    "channels": channels_demand * channel_multiplier,
                    "coarse_channels": items['demand'],
                    "baseline": 65000.0,
                    "telescope": "low"
                },
            )
        params.append(observation)

    return params


def create_demand_ratio_permutations():
    x = [64, 64, 64]
    n = 3
    max = 512
    print(len(x))
    random.seed(0)
    for j in [64, 128, 256, 512]:
        for i in range(0, len(x)):
            idx = random.randint(0, len(x) - 1)
            x[idx] = j
            print(x)
            number_obs = np.array([1, 2, 2]) * n
            print(number_obs, sum(number_obs))
            print(sum(np.array(x) * number_obs) / (sum(number_obs) * max))



def varied_low_obs_plan():
    params = []
    SKA_channels = [0]

    for channels in SKA_channels:
        for i in range(len(SKA_Low_antenna)):
            demand = SKA_Low_antenna[i]
            channels_demand = 128
            alt_demand = min(demand * 2, SKA_Low_antenna[-1])
            # alt_channels = min(channels*2, SKA_channels[-1])

            observation = {
                "nodes": 896,
                "infrastructure": "parametric",
                "telescope": "low",
                "items": [
                    {
                        "count": random.randint(3, 5),
                        "hpso": "hpso01",
                        "duration": LOW_OBSERVATIONS['hpso01']['duration'],
                        "workflows": LOW_OBSERVATIONS['hpso01']['workflows'],
                        "demand": alt_demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": random.randint(6, 10),
                        "hpso": 'hpso02a',
                        "duration": LOW_OBSERVATIONS['hpso02a']['duration'],
                        "workflows": LOW_OBSERVATIONS['hpso02a']['workflows'],
                        "demand": demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    },
                    {
                        "count": random.randint(6, 10),
                        "hpso": 'hpso02b',
                        "duration": LOW_OBSERVATIONS['hpso02b']['duration'],
                        "demand": demand,
                        "workflows": LOW_OBSERVATIONS['hpso02b']['workflows'],
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low"
                    }
                ]
            }
            params.append(observation)
            # path_name = dir_name / 'low' / f"low_{count}_.json"
            # with path_name.open('w') as fp:
            #     json.dump(observation, fp, indent=2)
    return params



import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(Path(__file__).name,)
    parser.add_argument('path')
    parser.add_argument('telescope', help="Choose from 'low' or 'mid'")
    parser.add_argument('graph_type', help="prototype, parallel")
    parser.add_argument('test', help="Choose from 'month', 'minimal'")

    # parser.add_argument() # TODO num_observation_repeats, seed
    args = parser.parse_args()

    WORKFLOW_TYPE_MAP = {
        "ICAL": args.graph_type,
        "DPrepA": args.graph_type,
        "DPrepB": args.graph_type,
        "DPrepC": args.graph_type,
        "DPrepD": args.graph_type,
    }

    all_params = []
    if args.telescope == 'low':
        if args.test == 'month':
            all_params.append(test_time_for_month_observation())
        else:
            logging.warning("Other options are not configured yet, exiting script...")
            sys.exit(1)
        # all_params.append(varied_low_obs_plan())
    elif args.telescope == 'mid':
        all_params.append(standard_mid_obs_plan())
    else:
        logging.info("Alternative not implemented yet, exiting script...")
        sys.exit()
        # all_params.append(simple_single_obs_plan())

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
                multiple_plans=True)
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=True,
                data_distribution='edges',
                multiple_plans=True)
