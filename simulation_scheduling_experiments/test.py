from tqdm import tqdm
import time

for i in range(0, 10):
    for j in tqdm(range(10), position=i, ncols=0):
        x = i * j
        for k in tqdm(range(5), position=10 + i, ncols=0):
            y = j * i * k
            time.sleep(1)
