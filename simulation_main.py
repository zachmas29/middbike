from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
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
    T: float,
    max_rate: int, 
    num_hubs: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:

    poisson: Dict[int, np.ndarray] = {}
    timestamps: Dict[int, np.ndarray] = {}
    for hub in range(num_hubs):
        rate_function = nhp.rate_from_hourly_profile(hourly_usage[hub], max_rate)
        distributions = nhp.nonhomogenous_poisson(T, max_rate, rate_function)
        poisson[hub] = nhp.bin_events_by_hour(distributions)
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
) -> None:
    rng = np.random.default_rng(seed)

    system_distribution = build_distributions(hourly_usage_arrays, T, max_rate, num_hubs)
    G = build_complete_digraph(travel_time)
    res = simulation(G, system_distribution[0], probs, rng = rng)
    print(f"no bike events: {res[0]}")
    print(f"no parking events: {res[1]}")
    print(f"timestamps: {system_distribution[1]}")

if __name__ == "__main__":
    num_hubs = 10

    base_usage = np.array([0, 2, 0, 0, 0, 3, 3, 3, 2, 2, 2, 1,
        1, 0, 3, 1, 1, 1, 6, 7, 0, 1, 1, 0], dtype = int)
    hourly_usage_arrays = [base_usage.copy() for _ in range(num_hubs)]

    dest_probs = build_dest_probs(num_hubs)
    run_simulation(hourly_usage_arrays, dest_probs)


    
