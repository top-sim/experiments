# copyright (c) 2024 rw bunney

# this program is free software: you can redistribute it and/or modify
# it under the terms of the gnu general public license as published by
# the free software foundation, either version 3 of the license, or
# (at your option) any later version.

# this program is distributed in the hope that it will be useful,
# but without any warranty; without even the implied warranty of
# merchantability or fitness for a particular purpose.  see the
# gnu general public license for more details.

# you should have received a copy of the gnu general public license
# along with this program.  if not, see <https://www.gnu.org/licenses/>.

import json
import random
import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from build.lib.skaworkflows.common import SKALow

from skaworkflows.observation.observation import HPSOParameter, ObservationPlan
from skaworkflows.config_generator import create_config
from skaworkflows import common
from skaworkflows.observation.parameters import load_observation_defaults

verbose = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

low_observation_defaults = load_observation_defaults("skalow")

mid_observations_defaults = load_observation_defaults("skamid")


def values_to_nparray(value_map, key):
    """
    take the key and get all values from the map
    :param value_map:
    :param key:
    :return:
    """
    return np.fromiter((y[key] for x, y in value_map.items()), int)

def create_baseline_sample(num_observations, alpha, baseline_limit):
    values = [b for b in SKALow.baselines if b <=baseline_limit]

    def compute_weights(values, alpha):
        normalized = [v / max(values) for v in values]
        weights = [x ** alpha for x in normalized]
        total = sum(weights)
        return [w / total for w in weights]


    weights = list(reversed(compute_weights(values, alpha)))
    return random.choices(list(values), weights=list(weights), k=num_observations)
    # experiments.append({
    #         'experiment': i + 1,
    #         'alpha': round(alpha, 2),
    #         'weights': weights,
    #         'sample': sample
    #     })
    #
    # for exp in experiments:
    #     print(f"Experiment {exp['experiment']} (alpha={exp['alpha']}):")
    #     print(f"  Weights: {[round(w, 3) for w in exp['weights']]}")
    #     print(f"  Sample: {exp['sample']}")
    #     print()

def spread_observations_across_demand(number_obs, demand_pool, alpha, baseline_limit, seed=None):
    """
    given the number of observations and a 'demand pool' of resources (e.g. [64, 128]), spread
    the number of observations across that pool of resources.

    the outcome should be a list that maps a certain number of observations
    to each resource amount, such that all numbers match the total number of observations
    required for that hpso in a given plan (based on the ratio).

    :param number_obs:
    :param demand_pool:
    :return: observations for each resource amount
    """

    if seed is not None:
        random.seed(seed)

    # bl_weights = create_baseline_sample(number_obs)

    fraction = demand_pool.get('ratio', {})
    # baselines = list(demand_pool.get('baseline', {}).keys())

    if not fraction: #or not baselines:
        raise ValueError("both 'ratio' and 'baseline' must be provided and non-empty.")

    station_types = list(fraction.keys())
    station_counts = {}
    allocated = 0

    # step 1: allocate number of stations per ratio
    for i, s in enumerate(station_types):
        if i == len(station_types) - 1:
            count = number_obs - allocated
        else:
            count = int(round(number_obs * fraction[s]))
            allocated += count
        station_counts[s] = count

    # step 2: track unique (station, baseline) combinations manually
    grouped = {}
    for station, num_obs in station_counts.items():
        sample = create_baseline_sample(num_obs, alpha, baseline_limit)
        for baseline in sample:
            key = (station, baseline)
            if key in grouped:
                grouped[key] += 1
            else:
                grouped[key] = 1

    # step 3: format the output
    result = []
    for (station, baseline), count in grouped.items():
        result.append({
            'stations': station,
            'baseline': baseline,
            'num': count,
            'alpha': alpha,
        })

    return result


def calc_demand_ratio(hpso_demand, telescope):
    # todo re-calculate this using the new approach
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
            {64: 0.5, 128: 0.30, 256: 0.20},
            {64: 0.5, 128: 0.20, 256: 0.20, 512: 0.10}
        ],
        [
            {64: 0.8, 128: 0.2},
            {64: 0.4, 128: 0.6},
            {64: 0.4, 128: 0.2, 256: 0.2},
            {64: 0.5, 128: 0.2, 256: 0.2, 512: 0.2}
        ],
    ]

    # baseline_permutations = [telescope.baselines[:i+1] for i, e in enumerate(telescope.baselines)]
    num_baseline_permutations = 5
    baseline_permutation_alphas = []
    for i in range(num_baseline_permutations):
        baseline_permutation_alphas.append(1 - (1 - ((num_baseline_permutations - i) / num_baseline_permutations)))
    hpso_demand = {key: {'stations':{}, 'baseline':{}} for key in low_observation_defaults["hpsos"]}
    for i, alpha in enumerate(baseline_permutation_alphas):
        for r in ratios:
            for x, _ in enumerate(telescope.stations):
                    _ratio = r[min(x, len(telescope.stations)-1)]
                    pname = f"ratios-{i}" + ''.join(f"_{k}-{v}" for k, v in _ratio.items()) + f"_alpha-{alpha:0.2f}"
                    logger.info("ratio: %s", pname)
                    for hpso in hpso_demand:
                        for j in telescope.stations[0:i+1]:
                            if j > 256 and hpso in ['hpso04a', 'hpso05a']:
                                continue
                            hpso_demand[hpso]['stations'].update({j: 0})
                        # for j in alpha:
                        #     if j > 32 and hpso in ['hpso04a', 'hpso05a']:
                        #         continue
                        #     hpso_demand[hpso]['baseline'].update({j:0})
                        hpso_demand[hpso]['ratio'] = _ratio
                        # demand pool slowly gets bigger
                    number_obs = values_to_nparray(low_observation_defaults["hpsos"], "observing_ratio") * n
                    observations = {}
                    for j, items in enumerate(hpso_demand.items()):
                        hpso, demand = items
                        baseline_limit = 24 if hpso in ['hpso04a', 'hpso05a'] else 65
                        obs = spread_observations_across_demand(number_obs[j],
                                                                hpso_demand[hpso], alpha, baseline_limit)
                        observations[hpso] = obs
                    # observations['alpha'] = alpha
                    final_d[pname] = observations
                    # final_set.append(observations)
    dfs = []
    for r in ratios:
        df = pd.DataFrame(r)
        dfs.append(df.replace(np.nan, "", regex=True))
    df_ratios = pd.concat(dfs)
    df_ratios.to_csv('hpso_ratios.csv')
    logger.info("final set #: %d", len(final_d))
    return final_d


def calc_n_for_given_time_in_seconds(time: int, durations: np.array, ratios: np.array):
    """
    determine the 'ratio' multiplier for a given set of hpso ratios and durations,
    such that n * ratios gives a total observation plan of at least 'time' length.
    """
    total = 0
    n = 0
    while total < time:
        total += sum(durations * (ratios))
        n += 1
    return n


def generate_permutations_table(permutations: dict, index: int):
    key = list(permutations.keys())[index]
    d = permutations[key]
    df_hpso_ratios = {}
    for hpso, el in d.items():
        df_hpso_ratios[hpso] = pd.DataFrame(el)
    for hpso, df in df_hpso_ratios.items():
        df = df.sort_values(by=['stations', 'baseline'])
        df['hpso'] = hpso
        df_hpso_ratios[hpso] = df
    df = pd.concat(list(df_hpso_ratios.values()))
    pivot = df.pivot_table(
        index=['baseline', 'hpso'],
        columns='stations',
        values='num',
        aggfunc='sum',  # sum in case of duplicates, or use 'first' if always unique
        fill_value=""  # keep empty if no entry
    )
    pivot_station = df.pivot_table(
        index=['stations', 'hpso'],
        columns='baseline',
        values='num',
        aggfunc='sum',  # sum in case of duplicates, or use 'first' if always unique
        fill_value=""  # keep empty if no entry
    )
    # This is Ryan breaking his #1 rule of not encoding core information about a dataset in its file name.
    pivot.to_csv(f"hpso_permutation_{key}_pivot.csv")
    pivot_station.to_csv(f"hpso_permutation_{key}_stations.csv")

def create_week_plan(telescope: str):
    """
    create a week's worth of observations
    """
    # one day
    duration = 7 * 24 * 3600
    if telescope == "low":
        n = calc_n_for_given_time_in_seconds(
            duration,
            values_to_nparray(low_observation_defaults["hpsos"], "duration"),
            values_to_nparray(low_observation_defaults["hpsos"], "observing_ratio"),
        )
        logger.info("creating %d iterations of observations")
        permutations = permute_low_observation_plans(n)
        generate_permutations_table(permutations, 1)
        generate_permutations_table(permutations, 5)
        generate_permutations_table(permutations, 10)
        generate_permutations_table(permutations, 54)
        generate_permutations_table(permutations, 59)
        return standard_low_obs_plan(permutations)
    elif telescope == "mid":
        n = calc_n_for_given_time_in_seconds(
            duration,
            values_to_nparray(mid_observations_defaults["hpsos"], "duration"),
            values_to_nparray(mid_observations_defaults["hpsos"], "observing_ratio"),
        )
        logger.info("creating %d iterations of observations")
        return standard_mid_obs_plan(permute_mid_observation_plan(n))
    else:
        return None


def permute_mid_observation_plan(n=1):
    """
    create combinations of demand
    """

    final_set = {}
    max_largest_demand = 2
    telescope = common.skamid()
    random.seed(100)

    for g in range(100):
        hpso_demand = {key["hpso"]: {} for key in mid_observations_defaults["hpsos"]}
        for i, antenna in enumerate(telescope.stations):
            for hpso in hpso_demand:
                for j in telescope.stations[0:i + 1]:
                    hpso_demand[hpso].update({j: 0})
            # demand pool slowly gets bigger
            number_obs = values_to_nparray(mid_observations_defaults["hpsos"], "ratio") * n
            ## new code
            prev_hpso = None
            for j, items in enumerate(hpso_demand.items()):
                hpso, demand = items
                obs = spread_observations_across_demand(number_obs[j],
                                                        hpso_demand[hpso])
                prev_d = []
                # allocate demand across antenna options
                for i, d in enumerate(demand):
                    if d == telescope.max_stations:
                        tmp = obs[i]
                        leftover = tmp - max_largest_demand
                        if leftover > 0:
                            demand[d] = max_largest_demand
                            # todo consider experimenting with this by just using
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
    currently, this is a placeholder method to generate one of a couple different
    observation plans.

    expect this method to be a) renamed in the future and b) improved upon

    'hpso13': {'duration': 28800, 'workflows': ["ical", "dprepa", "dprepb", "dprepc"]},
    'hpso15': {'duration': 15840, 'workflows': ["ical", "dprepa", "dprepb", "dprepc"]},
    'hpso22': {'duration': 28800, 'workflows': ["ical", "dprepa", "dprepb"]},
    'hpso32': {'duration': 7920, 'workflows': ["ical", "dprepb"]}


    returns
    -------

    """
    params = []
    # permutations = permute_mid_observation_plan()
    channels_demand = 128
    telescope = common.skamid
    for demand, hpso_numbers in num_obs_repeats.items():
        plan = telescope.initialise_plan()
        for hpso, items in hpso_numbers.items():
            for el in items:
                plan.add_observation(HPSOParameter(
                    count=el["num_obs"],
                    hpso=hpso,
                    duration=mid_observations_defaults["hpsos"][hpso]["duration"],
                    workflows=mid_observations_defaults["hpsos"][hpso]["workflows"],
                    demand=el["demand"],
                    channels=channels_demand * plan.telescope.channel_multiplier,
                    workflow_parallelism=el["demand"],
                    baseline=mid_observations_defaults["hpsos"][hpso]["baseline"],
                    telescope=str(plan.telescope))
                )
        params.append(plan)

    if verbose:
        print(json.dumps(params, indent=2, cls=common.npencoder))

    return params


def standard_low_obs_plan(
        num_obs_repeats: dict,
):
    """
    currently, this is a placeholder method to generate one of a couple different
    observation plans.

    expect this method to be a) renamed in the future and b) improved upon

    parameters
    ----------

    returns
    -------

    """
    params = {}

    channels_demand = 128
    for name, combination in num_obs_repeats.items():
        plan = ObservationPlan("low")
        logger.info("generating plan for: %s", name)
        for hpso, items in combination.items():
            for el in items:
                plan.add_observation(HPSOParameter(
                    count=el["num"],
                    hpso=hpso,
                    duration=low_observation_defaults["hpsos"][hpso]["duration"],
                    workflows=low_observation_defaults["hpsos"][hpso]["workflows"],
                    demand=el["stations"],
                    channels=channels_demand * plan.telescope.channels_multiplier,
                    workflow_parallelism=el["stations"],
                    baseline=el['baseline'],
                    telescope=str(plan.telescope))
                )
        params[name] = plan.to_json()

    if verbose:
        print(json.dumps(params, indent=2, cls=common.npencoder, sort_keys=True))

    logger.info("plans created: %s", params.keys())
    return params


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        Path(__file__).name,
    )
    parser.add_argument("path")
    parser.add_argument("telescope", help="choose from 'low' or 'mid'")
    parser.add_argument("graph_type", help="prototype, parallel")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--tables", default=False, action="store_true", help='Generate tables and do not run config generation')

    # parser.add_argument() # todo num_observation_repeats, seed
    args = parser.parse_args()

    workflow_type_map = {
        "ical": args.graph_type,
        "dprepa": args.graph_type,
        "dprepb": args.graph_type,
        "dprepc": args.graph_type,
        "dprepd": args.graph_type,
        "pulsar": "pulsar",
    }

    random.seed(2)
    if args.test:
        verbose = True
        random.seed(0)
        n = calc_n_for_given_time_in_seconds(
            7 * 24 * 3600,
            values_to_nparray(low_observation_defaults, "duration"),
            values_to_nparray(low_observation_defaults, "ratio"),
        )
        params = standard_low_obs_plan(permute_low_observation_plans(n))
        json.dumps(params, indent=2, cls=common.npencoder)

        sys.exit(0)

    all_params = create_week_plan(args.telescope)
    if args.tables:
        sys.exit(0)

    low_path = Path(args.path) / args.telescope

    print("creating config")
    # sys.exit()
    print(f"total plans: {len(all_params)}")
    # for ap in all_params:
        # sorted_keys = sorted(ap)
    # for multiplier in [1, 2, 5]:
    for name, plan in all_params.items():
        print(f"creating plan with demand: {name}")
        create_config(
            plan,
            low_path,
            workflow_type_map,
            timestep=5,
            data=False,
            data_distribution="standard",
            multiple_plans=False,
        )
        create_config(
            plan,
            low_path,
            workflow_type_map,
            timestep=5,
            data=True,
            data_distribution="standard",
            multiple_plans=False,
        )
        create_config(
            plan,
            low_path,
            workflow_type_map,
            timestep=5,
            data=True,
            data_distribution="edges",
            multiple_plans=False,
        )
