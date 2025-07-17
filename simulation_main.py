from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import non_homogenous_poisson as nhp
from simulation_code import build_complete_digraph, simulation

travel_time = np.array(
    [[0, 11, 11, 6, 6, 13, 6, 10, 8, 7],
    [7, 0, 6, 4, 8, 9, 5, 6, 3, 4],
    [11, 7, 0, 5, 10, 12, 7, 9, 4, 4],
    [5, 5, 5, 0, 6, 9, 3, 6, 3, 1],
    [4, 11, 11, 5, 0, 7, 4, 10, 8, 6],
    [11, 11, 13, 10, 8, 0, 8, 7, 10, 10],
    [5, 8, 9, 3, 4, 5, 0, 8, 6, 4],
    [8, 6, 9, 5, 9, 7, 6, 0, 6, 6],
    [6, 5, 3, 2, 7, 9, 4, 5, 0, 1],
    [5, 5, 5, 1, 6, 9, 3, 6, 2, 0]]
)

def build_dest_probs(num_hubs: int = 10) -> Dict[int, List[float]]:
    probs: Dict[int, List[float]] = {}
    for h in range(num_hubs):
        vec = np.full(num_hubs, 0.10)
        vec[h] = 0.0                  
        vec[(h + 1) % num_hubs] = 0.20
        probs[h] = vec.tolist()
    return probs

def build_distributions(
    hourly_usage: np.ndarray[np.ndarray[int]],
    T: int,
    max_rate: int, 
    num_hubs: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:

    poisson: Dict[int, np.ndarray] = {}
    timestamps: Dict[int, np.ndarray] = {}
    for hub in range(num_hubs):
        rate_function = nhp.rate_from_hourly_profile(hourly_usage[hub], max_rate)
        distributions = nhp.nonhomogenous_poisson(T, max_rate, rate_function)
        poisson[hub] = nhp.bin_events_by_hour(distributions, T)
        timestamps[hub] = distributions
    return poisson, timestamps
    
def run_simulation(
    hourly_usage_arrays: np.ndarray[np.ndarray[int]],
    probs: Dict[int, np.ndarray[float]],
    *,
    T: float = 24,
    max_rate: int = 5,
    num_hubs: int = 10,
    seed: int = 42
) -> Tuple[np.ndarray[int], np.ndarray[int], Dict[int, List[float]]]:
    rng = np.random.default_rng(seed)

    system_distribution = build_distributions(hourly_usage_arrays, T, max_rate, num_hubs)
    G = build_complete_digraph(travel_time)
    res = simulation(G, system_distribution[0], probs, T = T, rng = rng)
    return res[0], res[1], system_distribution[1]


def plot_lost_sales(
    no_bike: np.ndarray,
    no_parking: np.ndarray,
    *,
    title: str = "Lost-sale events per hour",
    savefile: str | None = None
    ) -> None:

    hours = np.arange(24)
    bar_w = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(hours - bar_w/2, no_bike,   width=bar_w,
           label="No-bike",   edgecolor="black")
    ax.bar(hours + bar_w/2, no_parking, width=bar_w,
           label="No-parking", edgecolor="black")

    ax.set_xticks(hours)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Event count")
    ax.set_title(title)
    ax.legend()

    if savefile:
        plt.savefig(savefile, dpi=150)
        print("Figure saved →", savefile)
    plt.show()

if __name__ == "__main__":
    num_hubs = 10

    max_rates = np.asarray([5, 10, 15, 20, 25, 30, 35, 40], dtype = int)

    base_usage_5 = np.array([0, 2, 0, 0, 0, 3, 3, 3, 2, 2, 2, 1, 1, 0, 3, 1, 1, 1, 5, 5, 0, 1, 1, 0], dtype = int)
    base_usage_10 = np.array([0, 4, 0, 0, 0, 6, 6, 6, 4, 4, 4, 2, 2, 0, 6, 2, 2, 2, 10, 10, 0, 2, 2, 0], dtype = int)
    base_usage_15 = np.array([0, 6, 0, 0, 0, 9, 9, 9, 6, 6, 6, 3, 3, 0, 9, 3, 3, 3, 15, 15, 0, 3, 3, 0], dtype = int)
    base_usage_20 = np.array([0, 8, 0, 0, 0, 12, 12, 12, 8, 8, 8, 4, 4, 0, 12, 4, 4, 4, 20, 20, 0, 4, 4, 0], dtype = int)
    base_usage_25 = np.array([0, 10, 0, 0, 0, 15, 15, 15, 10, 10, 10, 5, 5, 0, 15, 5, 5, 5, 25, 25, 0, 5, 5, 0], dtype = int)
    base_usage_30 = np.array([0, 12, 0, 0, 0, 18, 18, 18, 12, 12, 12, 6, 6, 0, 18, 6, 6, 6, 30, 30, 0, 6, 6, 0], dtype = int)
    base_usage_35 = np.array([0, 14, 0, 0, 0, 21, 21, 21, 14, 14, 14, 7, 7, 0, 21, 7, 7, 7, 35, 35, 0, 7, 7, 0], dtype = int)
    base_usage_40 = np.array([0, 16, 0, 0, 0, 24, 24, 24, 16, 16, 16, 8, 8, 0, 24, 8, 8, 8, 40, 40, 0, 8, 8, 0], dtype = int)

    hourly_usage_arrays_5 = [base_usage_5.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_10 = [base_usage_10.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_15 = [base_usage_15.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_20 = [base_usage_20.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_25 = [base_usage_25.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_30 = [base_usage_30.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_35 = [base_usage_35.copy() for _ in range(num_hubs)]
    hourly_usage_arrays_40 = [base_usage_40.copy() for _ in range(num_hubs)]

    dest_probs = build_dest_probs(num_hubs)

    res_5 = run_simulation(hourly_usage_arrays_5, dest_probs, T = 24, max_rate = 5)
    res_10 = run_simulation(hourly_usage_arrays_10, dest_probs, T = 24, max_rate = 10)
    res_15 = run_simulation(hourly_usage_arrays_15, dest_probs, T = 24, max_rate = 15)
    res_20 = run_simulation(hourly_usage_arrays_20, dest_probs, T = 24, max_rate = 20)
    res_25 = run_simulation(hourly_usage_arrays_10, dest_probs, T = 24, max_rate = 25)
    res_30 = run_simulation(hourly_usage_arrays_10, dest_probs, T = 24, max_rate = 30)
    res_35 = run_simulation(hourly_usage_arrays_10, dest_probs, T = 24, max_rate = 35)
    res_40 = run_simulation(hourly_usage_arrays_10, dest_probs, T = 24, max_rate = 40)

    no_bike_events = np.asarray([res_5[0], res_10[0], res_15[0], res_20[0], res_25[0], res_30[0], res_35[0], res_40[0]])
    no_parking_events = np.asarray([res_5[1], res_10[1], res_15[1], res_20[1], res_25[1], res_30[1], res_35[1], res_40[1]]) 

    #plot_lost_sales(res_5[0], res_5[1], title = "Lost-sale events per hour", savefile = "results.png")
    print(f"max rate = 5")
    print(f"no bike events: {res_5[0]}")
    print(f"no parking events: {res_5[1]}")
    print(f"timestamps: {res_5[2]}")

    print("max rate = 10")
    print(f"no bike events: {res_10[0]}")
    print(f"no parking events: {res_10[1]}")
    print(f"timestamps: {res_10[2]}")

    print("max rate = 15")
    print(f"no bike events: {res_15[0]}")
    print(f"no parking events: {res_15[1]}")
    print(f"timestamps: {res_15[2]}")

    print("max rate = 20")
    print(f"no bike events: {res_20[0]}")
    print(f"no parking events: {res_20[1]}")
    print(f"timestamps: {res_20[2]}")

    print("max rate = 25")
    print(f"no bike events: {res_25[0]}")
    print(f"no parking events: {res_25[1]}")
    print(f"timestamps: {res_25[2]}")

    print("max rate = 30")
    print(f"no bike events: {res_30[0]}")
    print(f"no parking events: {res_30[1]}")
    print(f"timestamps: {res_30[2]}")

    print("max rate = 35")
    print(f"no bike events: {res_35[0]}")
    print(f"no parking events: {res_35[1]}")
    print(f"timestamps: {res_35[2]}")

    print("max rate = 40")
    print(f"no bike events: {res_40[0]}")
    print(f"no parking events: {res_40[1]}")
    print(f"timestamps: {res_40[2]}")

    fig, axes = plt.subplots(2, 2, figsize = (10, 8))
    (ax11, ax12), (ax21, ax22) = axes

    ax11.scatter(max_rates, no_bike_events, s=60)
    ax11.set_xlabel("Max hourly rate")
    ax11.set_ylabel("No-bike events")
    ax11.set_title("Plot 1")

    ax12.scatter(max_rates, no_parking_events, s=60, marker="x")
    ax12.set_xlabel("Max hourly rate")
    ax12.set_ylabel("No-parking events")
    ax12.set_title("Plot-2")

