import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

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

# Parameters
N = 90
step = 5
max_large_frac = 0.25
max_large = int(N * max_large_frac)

lattice_tagged = make_lattice_tagged(N=N, step=step, maximum_large=max_large_frac)
x, y = ternary_coordinates(lattice_tagged, N)


print(len(lattice_tagged[lattice_tagged['status']=='valid']))
valid_lattice = lattice_tagged[lattice_tagged['status']=='valid']
print(valid_lattice)

ra = 5
small = [(x, ra-x) for x in range(0, ra+1)]
medium = [(x, x-ra) for x in range(2*ra+1, 2*ra+5)]
large = [(x, x-ra) for x in range(10*ra+1, 10*ra+5)]

lattice60 = valid_lattice.iloc[83]
print(lattice60)

np.random.seed(1)
types = {"A": 45, "B": 30, "C": 5, "D":5, "E": 5}
# for experiment in lattice60:
rng = np.random.default_rng()
observations_options=[]
excluded = ["A","B"]
observations_options.extend(rng.choice(large, lattice60['large']).tolist())
print(observations_options)
for t, count in types.items():
   if not t in excluded and observations_options:
       print(f"Processing {t}")
       print(rng.choice(observations_options, count, replace=False).tolist())

observations_options.extend(rng.choice(medium, lattice60['medium']).tolist())
for t, count in types.items():
   if not t in excluded and observations_options:
       print(f"Processing {t}")
       print(rng.choice(observations_options, count, replace=False).tolist())

observations_options.extend(rng.choice(small, lattice60['small']).tolist())
for t, count in types.items():
    if t in excluded and observations_options:
        print(f"Processing {t}")
        print(t, rng.choice(observations_options, count, replace=False).tolist())


""" for t, count in types.items():
   print(t, rng.choice(observations, count, replace=False).tolist())
 """

types = ["A", "B", "C", "D", "E"]

# Plot
fig, ax = plt.subplots(figsize=(7,7))

valid = lattice_tagged["status"] == "valid"
excluded = lattice_tagged["status"] == "excluded"
ax.scatter(x[excluded], y[excluded], c="lightgrey", s=60, label="excluded")
ax.scatter(x[valid], y[valid], c="blue", s=60, label="valid")

# Triangle border
triangle = np.array([[0,0], [1,0], [0.5,np.sqrt(3)/2], [0,0]])
ax.plot(triangle[:,0], triangle[:,1], 'k-', lw=1.5)

# Constraint line: large = max_large
s_vals = np.arange(0, N - max_large + 1)
m_vals = N - max_large - s_vals
l_vals = np.full_like(s_vals, max_large)
cx, cy = ternary_coordinates(pd.DataFrame({"small": s_vals, "medium": m_vals, "large": l_vals}), N)
ax.plot(cx, cy, 'r-', lw=2, label=f"large ≤ {max_large_frac:.0%}")

# Corner labels
ax.text(-0.05, -0.05, "Small", fontsize=12)
ax.text(1.02, -0.05, "Medium", fontsize=12)
ax.text(0.5, np.sqrt(3)/2 + 0.03, "Large", fontsize=12, ha='center')

# Side midpoints
side_midpoints = pd.DataFrame({
    "small": [N/2, N/2, 0],
    "medium": [N/2, 0, N/2],
    "large": [0, N/2, N/2],
    "label": ["Small–Medium", "Small–Large", "Medium–Large"]
})
mx, my = ternary_coordinates(side_midpoints, N)
ax.scatter(mx, my, c="green", s=80, zorder=5)
for i, txt in enumerate(side_midpoints["label"]):
    ax.text(mx[i], my[i]+0.03, txt, fontsize=10, ha='center', color="green")

# Internal midpoints for each component (50% of N)
# internal_midpoints = pd.DataFrame({
#     "small": [N/2, 0, N/4],
#     "medium": [0, N/2, N/4],
#     "large": [N/2, N/2, N/2],
#     "label": ["Small=50%", "Medium=50%", "Large=50%"]
# })
# ix, iy = ternary_coordinates(internal_midpoints, N)
# ax.scatter(ix, iy, c="orange", s=80, zorder=5)
# for i, txt in enumerate(internal_midpoints["label"]):
#     ax.text(ix[i], iy[i]+0.03, txt, fontsize=10, ha='center', color="orange")

ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f"Ternary lattice (N={N}, step={step})\nvalid vs excluded by large ≤ {max_large_frac:.0%}")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
