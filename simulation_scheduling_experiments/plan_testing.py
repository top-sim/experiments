import random
from symtable import Symbol

import numpy as np

# total = 0
expected = 0.5
percentage = {'hpso01': 0.25, 'hpso02a': 0.375, 'hpso02b': 0.375}
# TODO
# Approach this an alternate way by increasing the demand incrementally/randomly from a
# selection to incrementally achieve the expected value.
# This will probably require some finangling

demand = {'hpso01': [256], 'hpso02a': [256], 'hpso02b': [256]}
max = 512
min_obs = 6
current_count = {'hpso01': 0, 'hpso02a': 0, 'hpso02b': 0}

# while total < expected * min_obs and sum(current_count.values()) < min_obs:
#     for pick in percentage:
#         num_obs = sum(current_count.values())
#         if num_obs == 0:
#             current_count[pick] += 1
#             pick_demand = demand[pick]
#             total += (random.choice(pick_demand) / max)
#             # print(total)
#         else:
#             # print("world")
#             if current_count[pick] / min_obs < percentage[pick]:
#                 current_count[pick] += 1
#                 pick_demand = demand[pick]
#                 total += (random.choice(pick_demand) / max)
#                 # print(total)
# This is wrong - 512 for all of these should return 1.0

# hpso_idx = {'hpso01': 0, 'hpso2':1, 'hpso3':2}
# x = [64,64,64] #64,64,64]
# n=25
# max = 512
# print(len(x))
# random.seed(5)
# final_set = {}
# for j in [64, 128, 256, 512]:
#     for i in range(0,len(x)):
#         idx = random.randint(0, len(x)-1)
#         x[idx] = j
#         # print(x)
#         number_obs = np.array([1,2,2])*n
#         # number_obs = np.array([random.randint(1,3), random.randint(2,5),random.randint(2,5)])*n
#         # print(sum(number_obs))
#         demand_ratio = sum(np.array(x)*number_obs)/(sum(number_obs)*max)
#
#         final_set[demand_ratio] = ([a for a in x], number_obs)

# for demand, comb in final_set.items():
#     print(demand, comb)
# print(len(final_set) * 4 * 2 * 2)
SKA_Low_antenna = [64, 128, 256, 512]

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
        "ratio": 2,
    },
    "hpso02b": {
        "duration": 18000,
        "baseline": 65000,
        "workflows": ["ICAL", "DPrepA", "DPrepB", "DPrepC", "DPrepD"],
        "ratio": 2,
    },
}

MAX_ANTENNA = 512

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

hpso_idx = ["hpso01", "hpso02a", "hpso02b"]  # {'hpso01': 0, 'hpso2': 1, 'hpso3': 2}
n = 16
max_largest_demand = 2
random.seed(25)
for g in range(10):
    hpso_demand = {"hpso01": {}, "hpso02a": {}, "hpso02b": {}}
    final_set = {}
    for i, antenna in enumerate(SKA_Low_antenna):
        for hpso in hpso_demand:
            for j in SKA_Low_antenna[0:i+1]:
                hpso_demand[hpso].update({j: 0})
    # DEMAND POOL slowly gets bigger
        number_obs = values_to_nparray(LOW_OBSERVATIONS, "ratio") * n
        ## NEW CODE
        prev_hpso = None
        for j, items in enumerate(hpso_demand.items()):
            hpso, demand = items
            obs = spread_observations_across_demand(number_obs[j],
                                                        hpso_demand[hpso])
            prev_d = 0
            for i, d in enumerate(demand):
                if demand == 512:
                    tmp = obs[i]
                    leftover = max_largest_demand - tmp
                    if leftover > 0:
                        demand[d] = max_largest_demand
                        demand[prev_d] = leftover
                    else:
                        demand[d] = obs[i]
                else:
                    demand[d] = obs[i]
                prev_d = d

        tmp = {}
        demand_ratio = calc_demand_ratio(hpso_demand)

        for hpso, demand in hpso_demand.items():
            tmp[hpso] = []
            for antenna, obs in demand.items():
                tmp[hpso].append({
                    "demand": antenna,
                    "num_obs": obs
                })
        final_set[demand_ratio] = tmp

    print(final_set.keys())

import json

# json.dumps(final_set, indent=2)
