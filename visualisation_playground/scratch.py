# Copyright (C) 7/9/21 RW Bunney

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

# Show what is exucting at specific timesteps
sample = tasks_df[tasks_df['config'].str.contains(ex_wf_config)]
sample[
    (sample['aft'] > 1478) & (sample['aft'] < 1490)][['tasks', 'ast','aft']
]

"""
The above commands return; from this we can show that the final tasks of the 
'emu1' workflow are finishing, which is why tasks-use is low. 

                             tasks   ast   aft
13558               emu2_272_c0_13  1469  1479
13559               emu2_272_c1_13  1469  1479
13560               emu2_272_c2_13  1469  1479
13561               emu2_272_c3_13  1469  1479
13562              emu1_211_c74_18  1456  1479
13563              emu1_211_c75_18  1456  1479
13564              emu1_211_c76_18  1456  1479
13565              emu1_211_c77_18  1456  1479
13566              emu1_211_c78_18  1456  1479
13567              emu1_211_c79_18  1456  1479
13568               emu2_272_c4_13  1470  1480
13569               emu2_272_c5_13  1471  1481
13570               emu2_272_c6_13  1471  1481
13571               emu2_272_c7_13  1471  1481
13572    wallaby_423_channel_split  1480  1482
13573               emu2_272_c8_13  1473  1483
13574               emu2_272_c9_13  1473  1483
13575              emu2_272_c10_13  1473  1483
13576              emu2_272_c11_13  1473  1483
13577              emu2_272_c12_13  1473  1483
13578              emu2_272_c13_13  1473  1483
13579              emu2_272_c14_13  1473  1483
13580               dingo_150_c5_3  1462  1485
13581   wallaby_423_channel_split0  1483  1485
13582   wallaby_423_channel_split1  1483  1485
13583   wallaby_423_channel_split2  1483  1485
13584   wallaby_423_channel_split3  1483  1485
13585   wallaby_423_channel_split4  1483  1485
13586   wallaby_423_channel_split5  1483  1485
13587   wallaby_423_channel_split6  1483  1485
13588   wallaby_423_channel_split7  1483  1485
13589   wallaby_423_channel_split8  1483  1485
13590   wallaby_423_channel_split9  1483  1485
13591  wallaby_423_channel_split10  1483  1485
13592  wallaby_423_channel_split11  1483  1485
13593  wallaby_423_channel_split12  1483  1485
13594  wallaby_423_channel_split13  1483  1485
13595  wallaby_423_channel_split14  1483  1485
13596  wallaby_423_channel_split15  1483  1485
13597  wallaby_423_channel_split16  1483  1485
13598  wallaby_423_channel_split17  1483  1485
13599  wallaby_423_channel_split18  1483  1485
13600  wallaby_423_channel_split19  1483  1485
13601  wallaby_423_channel_split20  1486  1488
13602  wallaby_423_channel_split21  1486  1488
13603  wallaby_423_channel_split22  1486  1488
13604  wallaby_423_channel_split23  1486  1488
13605  wallaby_423_channel_split24  1486  1488
13606  wallaby_423_channel_split25  1486  1488
13607  wallaby_423_channel_split26  1486  1488
13608  wallaby_423_channel_split27  1486  1488
13609  wallaby_423_channel_split28  1486  1488
13610  wallaby_423_channel_split29  1486  1488
13611  wallaby_423_channel_split30  1486  1488
13612  wallaby_423_channel_split31  1486  1488
13613  wallaby_423_channel_split32  1486  1488
13614  wallaby_423_channel_split33  1486  1488
13615  wallaby_423_channel_split34  1486  1488
13616  wallaby_423_channel_split35  1486  1488
13617  wallaby_423_channel_split36  1486  1488
13618  wallaby_423_channel_split37  1486  1488
13619  wallaby_423_channel_split38  1486  1488
13620  wallaby_423_channel_split39  1486  1488
13621              emu2_272_c15_13  1479  1489
13622              emu2_272_c16_13  1479  1489
13623              emu2_272_c17_13  1479  1489
13624              emu2_272_c18_13  1479  1489
"""