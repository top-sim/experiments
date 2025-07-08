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
from skaworkflows.observation.observation import HPSOParameter, ObservationPlan

from skaworkflows.config_generator import create_config
from skaworkflows import common
from skaworkflows.observation.parameters import load_observation_defaults

VERBOSE = False
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

LOW_OBSERVATION_DEFAULTS = load_observation_defaults("SKALow")

MID_OBSERVATIONS_DEFAULTS = load_observation_defaults("SKAMid")


def values_to_nparray(value_map, key):
    """
    Take the key and get all values from the map
    :param value_map:
    :param key:
    :return:
    """
    return np.fromiter((y[key] for x, y in value_map.items()), int)


def spread_observations_across_demand(number, demand_pool, seed=None):
    """
    Given the number of observations and a 'demand pool' of resources (e.g. [64, 128]), spread
    the number of observations across that pool of resources.

    The outcome should be a list that maps a certain number of observations
    to each resource amount, such that all numbers match the total number of observations
    required for that HPSO in a given plan (based on the ratio).

    :param number:
    :param demand_pool:
    :return: observations for each resource amount
    """

    if seed is not None:
        random.seed(seed)

    fraction = demand_pool.get('ratio', {})
    baselines = list(demand_pool.get('baseline', {}).keys())

    if not fraction or not baselines:
        raise ValueError("Both 'ratio' and 'baseline' must be provided and non-empty.")

    station_types = list(fraction.keys())
    station_counts = {}
    allocated = 0

    # Step 1: Allocate number of stations per ratio
    for i, s in enumerate(station_types):
        if i == len(station_types) - 1:
            count = number - allocated
        else:
            count = int(round(number * fraction[s]))
            allocated += count
        station_counts[s] = count

    # Step 2: Track unique (station, baseline) combinations manually
    grouped = {}
    for station, count in station_counts.items():
        for _ in range(count):
            baseline = random.choice(baselines)
            key = (station, baseline)
            if key in grouped:
                grouped[key] += 1
            else:
                grouped[key] = 1

    # Step 3: Format the output
    result = []
    for (station, baseline), count in grouped.items():
        result.append({
            'stations': station,
            'baseline': baseline,
            'num': count
        })

    return result


def calc_demand_ratio(hpso_demand, telescope):
    # TODO re-calculate this using the new approach
    total_obs = sum([sum(x.values()) for x in hpso_demand.values()])
    total_demand = total_obs * telescope.max_stations
    cumulative_demand = 0
    for hpso, items in hpso_demand.items():
        for antenna, num in items.items():
            cumulative_demand += antenna * num

    return cumulative_demand / total_demand


def permute_low_observation_plans(n=1):
    max_largest_demand = 2
    random.seed(100)
    telescope = common.SKALow()
    final_set = []
    final_d = {}
    ratios = [
        [
            {64: 1},
            {64: 0.75, 128: 0.25},
            {64: 0.5, 128: 0.25, 256: 0.25},
            {64: 0.5, 128: 0.25, 256: 0.2, 512: 0.05}
        ],
        [
            {64: 0.9, 128: 0.1},
            {64: 0.5, 128: 0.5},
            {64: 0.5, 128: 0.35, 256: 0.15},
            {64: 0.5, 128: 0.35, 256: 0.10, 512: 0.05}
        ],
        [
            {64: 0.8, 128: 0.2},
            {64: 0.4, 128: 0.6},
            {64: 0.4, 128: 0.4, 256: 0.2},
            {64: 0.5, 128: 0.4, 256: 0.05, 512: 0.05}
        ],
    ]
    hpso_demand = {key: {'stations':{}, 'baseline':{}} for key in LOW_OBSERVATION_DEFAULTS["hpsos"]}
    for r in ratios:
        for x in range(5):
            for i, baseline in enumerate(telescope.baselines):
                _ratio = r[min(i, len(telescope.stations)-1)]
                pname = f"ratios_{x}" + ''.join(f"_{k}-{v}" for k, v in _ratio.items())
                LOGGER.info("Ratio: %s", pname)
                for hpso in hpso_demand:
                    for j in telescope.stations[0:i+1]:
                        hpso_demand[hpso]['stations'].update({j: 0})
                    for j in telescope.baselines[0:i+1]:
                        hpso_demand[hpso]['baseline'].update({j:0})
                    hpso_demand[hpso]['ratio'] = _ratio
                    # DEMAND POOL slowly gets bigger
                number_obs = values_to_nparray(LOW_OBSERVATION_DEFAULTS["hpsos"], "observing_ratio") * n
                observations = {}
                for j, items in enumerate(hpso_demand.items()):
                    hpso, demand = items
                    obs = spread_observations_across_demand(number_obs[j],
                                                            hpso_demand[hpso])
                    observations[hpso] = obs
                final_d[pname] = observations
                # final_set.append(observations)
    LOGGER.info("Final set #: %d", len(final_d))
    return final_d


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
    # One day
    duration = 7 * 24 * 3600
    if telescope == "low":
        n = calc_n_for_given_time_in_seconds(
            duration,
            values_to_nparray(LOW_OBSERVATION_DEFAULTS["hpsos"], "duration"),
            values_to_nparray(LOW_OBSERVATION_DEFAULTS["hpsos"], "observing_ratio"),
        )
        LOGGER.info("Creating %d iterations of observations")
        return standard_low_obs_plan(permute_low_observation_plans(n))
    elif telescope == "mid":
        n = calc_n_for_given_time_in_seconds(
            duration,
            values_to_nparray(MID_OBSERVATIONS_DEFAULTS["hpsos"], "duration"),
            values_to_nparray(MID_OBSERVATIONS_DEFAULTS["hpsos"], "observing_ratio"),
        )
        LOGGER.info("Creating %d iterations of observations")
        return standard_mid_obs_plan(permute_mid_observation_plan(n))
    else:
        return None


def permute_mid_observation_plan(n=1):
    """
    Create combinations of demand
    """

    final_set = {}
    max_largest_demand = 2
    telescope = common.SKAMid()
    random.seed(100)

    for g in range(100):
        hpso_demand = {key["hpso"]: {} for key in MID_OBSERVATIONS_DEFAULTS["hpsos"]}
        for i, antenna in enumerate(telescope.stations):
            for hpso in hpso_demand:
                for j in telescope.stations[0:i + 1]:
                    hpso_demand[hpso].update({j: 0})
            # DEMAND POOL slowly gets bigger
            number_obs = values_to_nparray(MID_OBSERVATIONS_DEFAULTS["hpsos"], "ratio") * n
            ## NEW CODE
            prev_hpso = None
            for j, items in enumerate(hpso_demand.items()):
                hpso, demand = items
                obs = spread_observations_across_demand(number_obs[j],
                                                        hpso_demand[hpso])
                prev_d = []
                # Allocate demand across antenna options
                for i, d in enumerate(demand):
                    if d == telescope.max_stations:
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
            demand_ratio = np.round(calc_demand_ratio(hpso_demand, telescope), 2)
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
    'hpso32': {'duration': 7920, 'workflows': ["ICAL", "DPrepB"]}


    Returns
    -------

    """
    params = []
    # permutations = permute_mid_observation_plan()
    channels_demand = 128
    telescope = common.SKAMid
    for demand, hpso_numbers in num_obs_repeats.items():
        plan = telescope.initialise_plan()
        for hpso, items in hpso_numbers.items():
            for el in items:
                plan.add_observation(HPSOParameter(
                    count=el["num_obs"],
                    hpso=hpso,
                    duration=MID_OBSERVATIONS_DEFAULTS["hpsos"][hpso]["duration"],
                    workflows=MID_OBSERVATIONS_DEFAULTS["hpsos"][hpso]["workflows"],
                    demand=el["demand"],
                    channels=channels_demand * plan.telescope.channel_multiplier,
                    workflow_parallelism=el["demand"],
                    baseline=MID_OBSERVATIONS_DEFAULTS["hpsos"][hpso]["baseline"],
                    telescope=str(plan.telescope))
                )
        params.append(plan)

    if VERBOSE:
        print(json.dumps(params, indent=2, cls=common.NpEncoder))

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

    channels_demand = 128
    for name, combination in num_obs_repeats.items():
        plan = ObservationPlan("low")
        LOGGER.info("Generating plan for: %s", name)
        for hpso, items in combination.items():
            for el in items:
                plan.add_observation(HPSOParameter(
                    count=el["num"],
                    hpso=hpso,
                    duration=LOW_OBSERVATION_DEFAULTS["hpsos"][hpso]["duration"],
                    workflows=LOW_OBSERVATION_DEFAULTS["hpsos"][hpso]["workflows"],
                    demand=el["stations"],
                    channels=channels_demand * plan.telescope.channels_multiplier,
                    workflow_parallelism=el["stations"],
                    baseline=el['baseline'],
                    telescope=str(plan.telescope))
                )
        params[name] = plan.to_json()

    if VERBOSE:
        print(json.dumps(params, indent=2, cls=common.NpEncoder, sort_keys=True))

    LOGGER.info("Plans created: %s", params.keys())
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
            values_to_nparray(LOW_OBSERVATION_DEFAULTS, "duration"),
            values_to_nparray(LOW_OBSERVATION_DEFAULTS, "ratio"),
        )
        params = standard_low_obs_plan(permute_low_observation_plans(n))
        json.dumps(params, indent=2, cls=common.NpEncoder)

        sys.exit(0)

    # all_params = []
    all_params = create_week_plan(args.telescope)

    low_path = Path(args.path) / args.telescope

    print("Creating config")
    # sys.exit()
    print(f"Total plans: {len(all_params)}")
    # for ap in all_params:
        # sorted_keys = sorted(ap)
    for name, plan in all_params.items():
        print(f"Creating plan with demand: {name}")
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
