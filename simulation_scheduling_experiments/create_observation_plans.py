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
import logging

import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path

from skaworkflows.common import SKALow

from skaworkflows.observation.observation import HPSOParameter, ObservationPlan
from skaworkflows.config_generator import create_config
from skaworkflows import common
from skaworkflows.observation.parameters import load_observation_defaults

verbose = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

low_observation_defaults = load_observation_defaults("skalow")

mid_observations_defaults = load_observation_defaults("skamid")

RATIOS = [
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
        {64: 0.25, 128: 0.25, 256: 0.25, 512: 0.25}
    ],
]

import itertools
# Based on heatmap of pairs comparing PFLOPs of observations
all_pairs = list(itertools.product(SKALow.baselines, SKALow.stations))
SKALOW_LARGE_PAIRS = [(65, 256), (32.5, 512), (65, 512)]
SKALOW_MED_PAIRS = [(65, 64), (65, 128),(32.5, 128), (32.5, 256), (16.25, 512), (16.25, 256), (8.125, 512)]
SKALOW_SMALL_PAIRS = list(set(all_pairs) - set(SKALOW_MED_PAIRS) - set(SKALOW_LARGE_PAIRS))

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

def spread_observations_across_demand(number_obs, demand_pool, pairs, baseline_limit, seed=None):
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
        acceptable_pairs = [(y,x) for (x, y) in pairs if y == station]
        # acceptable_baselines = list({x for (x, y) in acceptable_stations})
        # sample = create_baseline_sample(num_obs, alpha, baseline_limit)
        sample = random.choices(acceptable_pairs, k=num_obs)
        for key in sample:
            # key = (station, baseline)
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
            'alpha': pairs,
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

def make_lattice_tagged(N: int, step: int = 1, maximum_large: float = 1.0) -> pd.DataFrame:
    """
    Create ternary-style sequence of experiments with set-maximum value for the
    number of "large" observations.
    """
    max_large = int(N * maximum_large)
    rows = []
    for s in range(0, N + 1, step):
        for m in range(0, N - s + 1, step):
            l = N - s - m
            status = "valid" if l <= max_large else "excluded"
            rows.append({"small": s, "medium": m, "large": l, "status": status})
    return pd.DataFrame(rows)

def ternary_coordinates(df, N):
    """
    Convert (small, medium, large) counts to 2D coordinates for a ternary plot.
    """
    s = df["small"].to_numpy()
    m = df["medium"].to_numpy()
    l = df["large"].to_numpy()
    x = 0.5 * (2*m + l) / N
    y = (np.sqrt(3)/2) * l / N
    return x, y

def permute_low_observation_plans(n=1):
    """
    Create our experimental observing plans

    :param n: This is the multiple of our time-on-sky ratios that are stored in low_observation_defaults

    :return:
    """
    random.seed(100)
    final_set = []

    observation_amounts = {}
    total_obs = 0
    for hpso, d in low_observation_defaults['hpsos'].items():
        tmp = d['observing_ratio']*n
        observation_amounts[hpso] = tmp
        total_obs+=tmp
    lattice = make_lattice_tagged(N=total_obs, step=5, maximum_large=0.25)
    experiments = lattice[lattice['status']=='valid']
    excluded = ["hpso04a", "hpso05a"]
    final_d = {}
    for i, row in enumerate(experiments.iterrows()):
        row_id, e = row
        pname = f"experiment_" + ''.join(f"small-{e['small']}_medium-{e['medium']}_large-{e['large']}")
        logger.info("Experiment params: %s", pname)
        number_obs = {k: v for k, v in observation_amounts.items()}
        rng = np.random.default_rng()
        observations_options = []
        observations_options.extend(rng.choice(SKALOW_LARGE_PAIRS, e['large']).tolist())
        observations_options.extend(rng.choice(SKALOW_MED_PAIRS, e['medium']).tolist())

        observation_pairs = {}
        exhausted = False
        hpsos = [h for h in list(number_obs.keys()) if h not in excluded]
        while not exhausted:
            for t in hpsos:
                if t not in observation_pairs:
                    observation_pairs[t] = []
                if not observations_options:
                    exhausted = True
                    break
                rint = rng.integers(low=0, high=len(observations_options), size=1)[0]
                if observations_options:
                    if len(observation_pairs[t]) < number_obs[t]:
                        observation_pairs[t].append(tuple(observations_options.pop(rint)))
                        # baseline, stations =   observations_options.pop(rint)
                        # observation_pairs[t].append({'baseline':baseline, 'stations':stations})
                    else:
                        del number_obs[t]
            # Either we're out of observation options or we've removed all the non-excluded HPSOs from our dictionary
            if not observations_options or len(number_obs) < len(hpsos):
                exhausted = True

        observations_options.extend(rng.choice(SKALOW_SMALL_PAIRS, e['small']).tolist())
        while observations_options:
            for t in  list(number_obs.keys()):
                if t not in observation_pairs:
                    observation_pairs[t] = []
                rint = rng.integers(low=0, high=len(observations_options), size=1)[0]
                if observations_options:
                    if len(observation_pairs[t]) < number_obs[t]:
                        observation_pairs[t].append(tuple(observations_options.pop(rint)))
                        # baseline, stations =   observations_options.pop(rint)
                        # observation_pairs[t].append({'baseline':baseline, 'stations':stations})
                if not observations_options:
                    break


        final_d[pname] = observation_pairs
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
    counter = dict(Counter(df_hpso_ratios))
    for pair, count in counter.items():
        baseline, stations = pair

    for hpso, el in d.items():
        df_hpso_ratios[hpso] = pd.DataFrame(el, columns=['baseline', 'stations'])
    for hpso, df in df_hpso_ratios.items():
        df = df.sort_values(by=['stations', 'baseline'])
        df['hpso'] = hpso
        df['num'] = 1
        df_hpso_ratios[hpso] = df
    df = pd.concat(list(df_hpso_ratios.values()))
    df = df.groupby(["baseline", "stations", "hpso"])["num"].count().reset_index()
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
        # generate_permutations_table(permutations, 5)
        # generate_permutations_table(permutations, 10)
        # generate_permutations_table(permutations, 35)
        generate_permutations_table(permutations, 84)
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
    from collections import Counter
    channels_demand = 128
    for name, combination in num_obs_repeats.items():
        plan = ObservationPlan("low")
        logger.info("generating plan for: %s", name)
        for hpso, items in combination.items():
            counter = dict(Counter(items))
            for pair, count in counter.items():
                baseline, stations = pair
                plan.add_observation(HPSOParameter(
                    count=count,
                    hpso=hpso,
                    duration=low_observation_defaults["hpsos"][hpso]["duration"],
                    workflows=low_observation_defaults["hpsos"][hpso]["workflows"],
                    demand=stations,
                    channels=channels_demand * plan.telescope.channels_multiplier,
                    workflow_parallelism=stations,
                    baseline=baseline,
                    telescope=str(plan.telescope))
                )
        params[name] = plan.to_json()

    if verbose:
        print(json.dumps(params, indent=2, cls=common.npencoder, sort_keys=True))

    logger.info("Plans created:")
    for key in params:
        logger.info("\t %s", key)
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
        "ICAL": args.graph_type,
        "DPrepA": args.graph_type,
        "DPrepB": args.graph_type,
        "DPrepC": args.graph_type,
        "DPrepD": args.graph_type,
        "Pulsar": "pulsar",
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
