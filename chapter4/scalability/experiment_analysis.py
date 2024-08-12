import numpy as np
import pandas as pd

resfile = "results_f2024-05-26.h5"
store = pd.HDFStore(resfile)
keysplit = []
for k in store.keys():
    keysplit.append(k.split('/'))
store.close()
print(keysplit)
dataset_types = ['sim', 'summary', 'tasks']
simulations = {f"{e[1]}/{e[2]}": {d: None for d in dataset_types} for e in keysplit}
for simulation, dtype in simulations.items():
    for dst in dataset_types:
        simulations[simulation][dst] = pd.read_hdf(resfile, key=f"{simulation}/{dst}")
for simulation, dtype in simulations.items():
    df = simulations[simulation]['summary']
    print(np.round(np.array(df['time']) * 5 / 3600, 2))
