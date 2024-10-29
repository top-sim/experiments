import random
from symtable import Symbol

import numpy as np

total = 0
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

while total < expected * min_obs and sum(current_count.values()) < min_obs:
    for pick in percentage:
        num_obs = sum(current_count.values())
        if num_obs == 0:
            current_count[pick] += 1
            pick_demand = demand[pick]
            total += (random.choice(pick_demand) / max)
            # print(total)
        else:
            # print("world")
            if current_count[pick] / min_obs < percentage[pick]:
                current_count[pick] += 1
                pick_demand = demand[pick]
                total += (random.choice(pick_demand) / max)
                # print(total)
# This is wrong - 512 for all of these should return 1.0

hpso_idx = {'hpso01': 0, 'hpso2':1, 'hpso3':2}
x = [64,64,64] #64,64,64]
n=25
max = 512
print(len(x))
random.seed(5)
final_set = {}
for j in [64, 128, 256, 512]:
    for i in range(0,len(x)):
        idx = random.randint(0, len(x)-1)
        x[idx] = j
        # print(x)
        number_obs = np.array([1,2,2])*n
        # number_obs = np.array([random.randint(1,3), random.randint(2,5),random.randint(2,5)])*n
        # print(sum(number_obs))
        demand_ratio = sum(np.array(x)*number_obs)/(sum(number_obs)*max)

        final_set[demand_ratio] = ([a for a in x], number_obs)

for demand, comb in final_set.items():
    print(demand, comb)
print(len(final_set) * 4 * 2 * 2)


# for hpso, index in hpso_idx.items():
#     print(f"{hpso} has {number_obs[index]} observations demanding {x[index]} arrays ")
