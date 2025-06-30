from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from non_homogenous_poisson import (
    rate_from_hourly_profile,
    nonhomogenous_poisson,
    bin_events_by_hour
)
from simulation_oop import build_complete_digraph, simulation

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


def build_distributions(hourly_usage: List[float],
    T: int,
    max_rate: int, num_hubs):
    poisson = Dict[int, np.array]
    for i in range(num_hubs):
        rate_function = rate_from_hourly_profile(hourly_usage, max_rate)
        distributions = nonhomogenous_poisson(hourly_usage, T, max_rate, rate_function)
        poisson[i] = distributions
    return poisson
    
def run_simulation(
    expected_per_hour: List[int],
    hourly_usage_array: List[np.ndarray],
    *,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    dist, event_times = build_distributions(expected_per_hour, hourly_usage_array, rng)

    G = build_complete_digraph(travel_time)
    dest_probs = build_dest_probs()

    no_bike, no_park, _ = simulation(
        G, dist, dest_probs, max_bikes_per_hub = 10, initial_bikes_per_hub = 5, rng = rng
    )

    print("Hourly no-bike events: ", no_bike)
    print("Hourly no-parking events: ", no_park)
    print("Totals -> no-bike: ", no_bike.sum())
    print("Totals -> no-bike: ", no_park.sum())
    print("Sample events times from Hub 0: ", np.round(event_times[0][:10], 2))

    return no_bike, no_park

if __name__ == "__main__":
    num_hubs = 10

    base_usage = np.array([0, 2, 0, 0, 0, 3, 3, 3, 2, 2, 2, 1,
        1, 0, 3, 1, 1, 1, 6, 7, 0, 1, 1, 0], dtype = float)
    hourly_usage_arrays = [base_usage.copy() for _ in range(num_hubs)]

    expected_per_hour = [3] * num_hubs

    run_simulation(expected_per_hour, hourly_usage_arrays)
