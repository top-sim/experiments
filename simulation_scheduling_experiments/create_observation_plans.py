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

import json
import random
import sys
from pathlib import Path
import logging

import numpy as np

from skaworkflows.config_generator import create_config

VERBOSE = True
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    # My complete laziness
    # https://java2blog.com/object-of-type-int64-is-not-json-serializable/

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int64):
            return int(o)
        return super(NpEncoder, self).default(o)


LOW_OBSERVATIONS = {
    "hpso01": {
        "duration": 18000,
        "baseline": 65000,
        "workflows": ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"],
        "ratio": 1,
    },
    "hpso02a": {
        "duration": 18000,
        "baseline": 65000,
        "workflows": ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"],
        "ratio": 1,
    },
    "hpso02b": {
        "duration": 18000,
        "baseline": 65000,
        "workflows": ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"],
        "ratio": 1,
    },
    "hpso04a": {
        "duration": 2400,
        "baseline": 65000,
        "workflows": ["Pulsar"],
        "ratio": 5,
    },
    "hpso05a": {
        "duration": 2400,
        "baseline": 65000,
        "workflows": ["Pulsar"],
        "ratio": 2,
    }
}

# TODO These defaults really should be stored in SKAWorkflows and referenced exclusively
# there until the end of time. Bonus points for wrapping the SDP parametric model

MID_OBSERVATIONS = {
    "hpso13": {
        "duration": 28800,
        "baseline": 35000,
        "workflows": ["ICAL", "DPrepA", "DPrepB", "DPrepC"],
        "ratio": 1,
    },
    "hpso15": {
        "duration": 15840,
        "baseline": 15000,
        "workflows": ["ICAL", "DPrepA", "DPrepB", "DPrepC"],
        "ratio": 2,
    },
    "hpso22": {
        "duration": 28800,
        "baseline": 150000,
        "workflows": ["ICAL", "DPrepA", "DPrepB"],
        "ratio": 2,
    },
    "hpso32": {
        "duration": 7920,
        "baseline": 20000,
        "workflows": ["ICAL", "DPrepB"],
        "ratio": 4,
    },
}

LOW_RATIOS = [1, 2, 2]

MID_RATIOS = [1, 2, 2, 4]

MAX_ANTENNA = 512
#
# These are the 'coarse-grained' channel values.
SKA_channels = [64, 128, 256]

# 32 is an arbitrary minimum; 512 hard maximum
SKA_Low_antenna = [64, 128, 256, 512]
# 64 + N, where 64 is the base number of dishes
SKA_Mid_antenna = [64, 102, 140, 197]

channel_multiplier = 128


def total(d):
    total = {}
    for k, v in d.items():
        total[k] = sum([x for y, x in v.items()])
    return total


def values_to_nparray(value_map, key):
    return np.fromiter((y[key] for x, y in value_map.items()), int)


def spread_observations_across_demand(number, demand_pool):
    count = 0
    obs = []
    while number > 0:
        nobs = random.randint(0, number)
        if count < len(demand_pool) - 1:
            obs.append(nobs)
            number -= nobs
        else:
            obs.append(number)
            number = 0
        count += 1
    length = len(obs)
    demand_length = len(demand_pool)
    if length < demand_length:
        obs.extend(0 for x in range(demand_length - length))

    return obs  # Enumerate over this with the demand dictionary


def calc_demand_ratio(hpso_demand):
    total_obs = sum([sum(x.values()) for x in hpso_demand.values()])
    total_demand = total_obs * MAX_ANTENNA
    cumulative_demand = 0
    for hpso, items in hpso_demand.items():
        for antenna, num in items.items():
            cumulative_demand += antenna * num

    return cumulative_demand / total_demand


def permute_low_observation_plans(n=1):
    hpso_idx = ["hpso01", "hpso02a", "hpso02b"]  # {'hpso01': 0, 'hpso2': 1, 'hpso3': 2}
    max_largest_demand = 2
    random.seed(100)

    final_set = {}
    for g in range(100):
        hpso_demand = {"hpso01": {}, "hpso02a": {}, "hpso02b": {}, "hpso04a": {}, "hpso05a": {}}
        for i, antenna in enumerate(SKA_Low_antenna):
            for hpso in hpso_demand:
                for j in SKA_Low_antenna[0:i + 1]:
                    hpso_demand[hpso].update({j: 0})
            # DEMAND POOL slowly gets bigger
            number_obs = values_to_nparray(LOW_OBSERVATIONS, "ratio") * n
            ## NEW CODE
            prev_hpso = None
            for j, items in enumerate(hpso_demand.items()):
                hpso, demand = items
                obs = spread_observations_across_demand(number_obs[j],
                                                        hpso_demand[hpso])
                prev_d = []
                # Allocate demand across antenna options
                for i, d in enumerate(demand):
                    if d == 512:
                        tmp = obs[i]
                        leftover = tmp - max_largest_demand
                        if leftover > 0:
                            demand[d] = max_largest_demand
                            intermediate_obs = {p: 0 for p in prev_d}
                            int_obs = spread_observations_across_demand(
                                leftover, intermediate_obs)
                            for x, key in enumerate(intermediate_obs):
                                demand[key] += int_obs[x]
                        else:
                            demand[d] = obs[i]
                    else:
                        demand[d] = obs[i]
                    if d > 64:
                        prev_d.append(d)

            tmp = {}
            demand_ratio = np.round(calc_demand_ratio(hpso_demand), 2)
            if demand_ratio in final_set:
                continue
            for hpso, demand in hpso_demand.items():
                tmp[hpso] = []
                for antenna, obs in demand.items():
                    tmp[hpso].append({
                        "demand": antenna,
                        "num_obs": obs
                    })
            final_set[demand_ratio] = tmp

    return final_set


def simple_single_obs_plan():
    """
    Generate a plan for a single observation

    This is purely to validate and demonstrate the scheduling heuristic performance.

    Returns
    -------
    plans: list of json-compliant dictionaries, representing observation plans
    """
    params = []
    observation = {
        "nodes": 896,
        "infrastructure": "parametric",
        "telescope": "low",
        "items": [
            {
                "count": 1,
                "hpso": "hpso01",
                "duration": LOW_OBSERVATIONS["hpso01"]["duration"],
                "workflows": LOW_OBSERVATIONS["hpso01"]["workflows"],
                "demand": 512,  # demand * 1,
                "channels": 256 * channel_multiplier,  # channels * channel_multiplier,
                # Consider putting a hard limit on the max channels if
                # generating workflows for a 896 syste?
                "coarse_channels": 896,  # parallel channels,
                "baseline": 65000.0,
                "telescope": "low",
            }
        ],
    }
    params.append(observation)

    return params


def values_to_nparray(value_map, key):
    return np.fromiter((y[key] for x, y in value_map.items()), int)


def calc_n_for_given_time_in_seconds(time: int, durations: np.array, ratios: np.array):
    """
    Determine the 'ratio' multiplier for a given set of HPSO ratios and durations,
    such that n * ratios gives a total observation plan of at least 'time' length.
    """
    total = 0
    n = 0
    while total < time:
        total += sum(durations * (ratios))
        n += 1
    return n


def create_week_plan(telescope: str):
    """
    Create a week's worth of observations
    """
    duration = 7 * 24 * 3600
    if telescope == "low":
        n = calc_n_for_given_time_in_seconds(
            duration,
            values_to_nparray(LOW_OBSERVATIONS, "duration"),
            values_to_nparray(LOW_OBSERVATIONS, "ratio"),
        )
        LOGGER.info("Creating %d iterations of observations")
        return standard_low_obs_plan(permute_low_observation_plans(n))
    if telescope == "mid":
        n = calc_n_for_given_time_in_seconds(
            duration,
            values_to_nparray(MID_OBSERVATIONS, "duration"),
            values_to_nparray(MID_OBSERVATIONS, "ratio"),
        )
        LOGGER.info("Creating %d iterations of observations")
        return standard_mid_obs_plan(permute_mid_observation_plan(n))


def permute_mid_observation_plan(n=1):
    """
    Create combinations of demand
    """
    hpso_idx = [
        "hpso13",
        "hpso15",
        "hpso22",
        "hpso32",
    ]

    final_set = {}
    max_largest_demand = 2
    random.seed(100)
    # default_observation_ratios = [1, 2, 2, 4]
    # max_antenna = 197
    # final_set = {}
    # for j in [64, 102, 140, 197]:  # TODO put these in a global/common SKAworkflows file
    #     for i in range(0, len(hpso_demand)):
    #         idx = random.randint(0, len(hpso_demand) - 1)
    #         hpso_demand[hpso_idx[idx]] = j
    #         number_obs = np.array(default_observation_ratios) * n
    #         demand_ratio = sum(np.array(list(hpso_demand.values())) * number_obs) / (
    #                 sum(number_obs) * max_antenna
    #         )
    #         if demand_ratio > 0.8:
    #             continue
    #         tmp = {}
    #         for hpso, demand in hpso_demand.items():
    #             tmp[hpso] = {
    #                 "demand": demand,
    #                 "num_obs": number_obs[hpso_idx.index(hpso)],
    #             }
    #         final_set[demand_ratio] = tmp
    # final_set = {}

    for g in range(100):
        hpso_demand = {
            "hpso13": {},
            "hpso15": {},
            "hpso22": {},
            "hpso32": {},
        }
        for i, antenna in enumerate(SKA_Mid_antenna):
            for hpso in hpso_demand:
                for j in SKA_Mid_antenna[0:i + 1]:
                    hpso_demand[hpso].update({j: 0})
            # DEMAND POOL slowly gets bigger
            number_obs = values_to_nparray(MID_OBSERVATIONS, "ratio") * n
            ## NEW CODE
            prev_hpso = None
            for j, items in enumerate(hpso_demand.items()):
                hpso, demand = items
                obs = spread_observations_across_demand(number_obs[j],
                                                        hpso_demand[hpso])
                prev_d = []
                # Allocate demand across antenna options
                for i, d in enumerate(demand):
                    if d == 512:
                        tmp = obs[i]
                        leftover = tmp - max_largest_demand
                        if leftover > 0:
                            demand[d] = max_largest_demand
                            # TODO consider experimenting with this by just using
                            # smallest
                            intermediate_obs = {p: 0 for p in prev_d}
                            int_obs = spread_observations_across_demand(
                                leftover, intermediate_obs)
                            for x, key in enumerate(intermediate_obs):
                                demand[key] += int_obs[x]
                        else:
                            demand[d] = obs[i]
                    else:
                        demand[d] = obs[i]
                    if d > 64:
                        prev_d.append(d)

            tmp = {}
            demand_ratio = np.round(calc_demand_ratio(hpso_demand), 2)
            if demand_ratio in final_set:
                continue
            for hpso, demand in hpso_demand.items():
                tmp[hpso] = []
                for antenna, obs in demand.items():
                    tmp[hpso].append({
                        "demand": antenna,
                        "num_obs": obs
                    })
            final_set[demand_ratio] = tmp


def standard_mid_obs_plan(num_obs_repeats: dict):
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
    params = []
    # permutations = permute_mid_observation_plan()
    channels_demand = 128

    for demand, hpso_numbers in num_obs_repeats.items():
        observation = {
            "nodes": 796,
            "infrastructure": "parametric",
            "telescope": "mid",
            "items": [],
        }
        for hpso, items in hpso_numbers.items():
            observation["items"].append(
                {
                    "count": items["num_obs"],
                    "hpso": hpso,
                    "duration": MID_OBSERVATIONS[hpso]["duration"],
                    "workflows": MID_OBSERVATIONS[hpso]["workflows"],
                    "demand": items["demand"],
                    "channels": channels_demand * channel_multiplier,
                    "coarse_channels": items["demand"],
                    "baseline": MID_OBSERVATIONS[hpso]["baseline"],
                    "telescope": "mid",
                },
            )
        params.append(observation)

    if VERBOSE:
        print(json.dumps(params, indent=2, cls=NpEncoder))

    return params


def standard_low_obs_plan(
        num_obs_repeats: dict,
):
    """
    Currently, this is a placeholder method to generate one of a couple different
    observation plans.

    Expect this method to be a) renamed in the future and b) improved upon

    Parameters
    ----------

    Returns
    -------

    """
    params = {}

    # permutations = permute_low_observation_plans()
    # demand = SKA_Low_antenna[i]
    channels_demand = 128
    # alt_demand = min(demand * 2, SKA_Low_antenna[-1])  # was -1

    for demand, hpso_numbers in num_obs_repeats.items():
        observation = {
            "nodes": 896,
            "infrastructure": "parametric",
            "telescope": "low",
            "items": [],
        }
        for hpso, items in hpso_numbers.items():
            for el in items:
                observation["items"].append(
                    {
                        "count": el["num_obs"],
                        "hpso": hpso,
                        "duration": LOW_OBSERVATIONS[hpso]["duration"],
                        "workflows": LOW_OBSERVATIONS[hpso]["workflows"],
                        "demand": el["demand"],
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": el["demand"],
                        "baseline": LOW_OBSERVATIONS[hpso]["baseline"],
                        "telescope": "low",
                    },
                )
        params[demand] = observation

    if VERBOSE:
        print(json.dumps(params, indent=2, cls=NpEncoder, sort_keys=True))

    return params


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
                        "duration": LOW_OBSERVATIONS["hpso01"]["duration"],
                        "workflows": LOW_OBSERVATIONS["hpso01"]["workflows"],
                        "demand": alt_demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low",
                    },
                    {
                        "count": random.randint(6, 10),
                        "hpso": "hpso02a",
                        "duration": LOW_OBSERVATIONS["hpso02a"]["duration"],
                        "workflows": LOW_OBSERVATIONS["hpso02a"]["workflows"],
                        "demand": demand,
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low",
                    },
                    {
                        "count": random.randint(6, 10),
                        "hpso": "hpso02b",
                        "duration": LOW_OBSERVATIONS["hpso02b"]["duration"],
                        "demand": demand,
                        "workflows": LOW_OBSERVATIONS["hpso02b"]["workflows"],
                        "channels": channels_demand * channel_multiplier,
                        "coarse_channels": demand,
                        "baseline": 65000.0,
                        "telescope": "low",
                    },
                ],
            }
            params.append(observation)
            # path_name = dir_name / 'low' / f"low_{count}_.json"
            # with path_name.open('w') as fp:
            #     json.dump(observation, fp, indent=2)
    return params


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        Path(__file__).name,
    )
    parser.add_argument("path")
    parser.add_argument("telescope", help="Choose from 'low' or 'mid'")
    parser.add_argument("graph_type", help="prototype, parallel")
    parser.add_argument("--test", default=False, action="store_true")

    # parser.add_argument() # TODO num_observation_repeats, seed
    args = parser.parse_args()

    WORKFLOW_TYPE_MAP = {
        "ICAL": args.graph_type,
        "DPrepA": args.graph_type,
        "DPrepB": args.graph_type,
        "DPrepC": args.graph_type,
        "DPrepD": args.graph_type,
        "Pulsar": "pulsar",
    }

    random.seed(2)
    if args.test:
        VERBOSE = True
        random.seed(0)
        n = calc_n_for_given_time_in_seconds(
            7 * 24 * 3600,
            values_to_nparray(LOW_OBSERVATIONS, "duration"),
            values_to_nparray(LOW_OBSERVATIONS, "ratio"),
        )
        params = standard_low_obs_plan(permute_low_observation_plans(n))
        json.dumps(params, indent=2, cls=NpEncoder)
        # LOGGER.debug(params)
        # create_week_plan(1)
        # params = standard_mid_obs_plan(permute_mid_observation_plan(n))
        # LOGGER.debug(params)
        sys.exit(0)

    all_params = []
    all_params.append(create_week_plan(args.telescope))

    low_path = Path(args.path) / args.telescope

    print("Creating config")
    # sys.exit()
    for ap in all_params:
        sorted_keys = sorted(ap.keys())
        for demand in sorted_keys:
            print(f"Creating plan with demand: {demand}")
            plan = ap[demand]
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=False,
                data_distribution="standard",
                multiple_plans=False,
            )
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=True,
                data_distribution="standard",
                multiple_plans=False,
            )
            create_config(
                plan,
                low_path,
                WORKFLOW_TYPE_MAP,
                timestep=5,
                data=True,
                data_distribution="edges",
                multiple_plans=False,
            )
