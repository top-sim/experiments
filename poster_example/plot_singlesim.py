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
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

import datetime


def plot_comparison_tasks_resources(pickle1, pickle2):
    """
    Parameters
    ----------
    pickle1 : pandas Pickle
        First pickle to compare
    pickle2 : second Pickle to compare

    return None
    """

    sns.set_style("darkgrid")

    df_heft = pd.read_pickle(pickle1)
    df_fcfs = pd.read_pickle(pickle2)

    fig, axs = plt.subplots(nrows=2)

    sns.lineplot(
        data=df_heft, x=df_heft.index, y=df_heft.running_tasks, ax=axs[0],
        label='HEFT'
    )
    axs[0].set(ylabel='No. Running Tasks')
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
    axs[1].set(ylabel='No. Available Machines', xlabel='Sim Runtime')
    axs[0].legend()
    axs[1].legend()

    current_time = f'{time.time()}'.split('.')[0]
    plt.savefig(f'heft-fcfs_comparison_singlesim_{current_time}.svg',format='svg')


if __name__ == '__main__':

    plot_comparison_tasks_resources('poster_20cluster.trace-fcfs.pkl',
                                    'poster_20cluster.trace-fcfs.pkl')

