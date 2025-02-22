import pandas as pd
d = pd.read_hdf("results_f2024-05-12.h5", key="Sun240512203704/skaworkflows_2024-05-12_12:29:18/summary")
print(d)

