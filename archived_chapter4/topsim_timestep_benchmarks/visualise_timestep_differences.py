# Copyright (C) 2022 RW Bunney

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
from matplotlib.ticker import AutoMinorLocator


def create_standard_axis(ax, minor=True):
    """
    Create an axis and figure with standard grid structure and ticks

    Saves us having to reference/create an arbitrary

    Returns
    -------
    axis
    """

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_axisbelow(True)
    ax.grid(axis='both', which='major', color='grey', zorder=0)
    if minor:
        ax.grid(axis='both', which='minor', color='lightgrey',
                linestyle='dotted')
    ax.tick_params(
        right=True, top=True, which='both', direction='in'
    )

    return ax
