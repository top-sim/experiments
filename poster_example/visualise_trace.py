# Copyright (C) 10/9/20 RW Bunney

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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("darkgrid")
# tracefile = 'simulations/heft_sim/output/sim.trace'
heft_pickle = 'poster_20cluster.trace-heft.pkl'
fcfs_pickle = 'poster_20cluster.trace-fcfs.pkl'
# pheft_pickle = 'simulations/heft_sim/output/heft_sim.trace-pheft.pkl'
# with open(tracefile, 'r') as infile:
# 	trace = json.load(infile)

df_heft = pd.read_pickle(heft_pickle)
df_fcfs = pd.read_pickle(fcfs_pickle)
# df_pheft = pd.read_pickle(pheft_pickle)

fig, axs = plt.subplots(nrows=2)

sns.lineplot(
    data=df_heft, x=df_heft.index, y=df_heft.running_tasks, ax=axs[0],
    label='HEFT'
)
axs[0].set(ylabel='No. Running Tasks in SDP')
sns.lineplot(
    data=df_fcfs, x=df_fcfs.index, y=df_fcfs.running_tasks,
    ax=axs[0], label='FCFS'
)

sns.lineplot(
    data=df_heft, x=df_heft.index, y="available_resources",
    ax=axs[1], label='HEFT'
)

sns.lineplot(
    data=df_fcfs, x=df_fcfs.index, y=df_fcfs.available_resources,
    ax=axs[1], label='FCFS'
)
axs[1].set(xlabel='No. Available Machines in Cluster', ylabel='Sim Runtime')
axs[0].legend()
axs[1].legend()
# sns.lineplot(
#     data=df_heft, x=df_heft.index, y="ingest_resources", ax=axs[0, 1]
# )
# sns.lineplot(
#     data=df_heft, x=df_heft.index, y="hotbuffer_current_capacity", ax=axs[1, 0]
# )
# sns.lineplot(
#     data=df_heft, x=df_heft.index, y="coldbuffer_current_capacity", ax=axs[1, 1]
# )

plt.savefig('comparison.svg',format='svg')

# for timestamp in trace:
# 	# print('Time @ {}'.format(timestamp['timestamp']))
#
# 	print('\tcluster_state:')
# 	for element in timestamp['cluster_state']:
# 		for m in timestamp['cluster_state']['machines']:
# 			print('\t\t{}'.format(m))
# 	print('\ttelescope_state:')
# 	for element in timestamp['telescope_state']:
# 		print('\t\t{}: {}'.format(
# 			element, timestamp['telescope_state'][element]
# 		))
# 	print('\tscheduler_state:')
# 	print('\t\t{}'.format(timestamp['scheduler_state']))
# 	print('\t{}'.format(timestamp['buffer_state']))
# 	for element in timestamp['buffer_state']:
# 		print('\t\t{}'.format(timestamp['buffer_state'][element]))
#
