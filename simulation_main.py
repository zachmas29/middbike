from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import non_homogenous_poisson as nhp
import size_dictionary as sd
import elevation_matrix as em
import travel_matrix as tm
import hourly_usage_and_probability as dist_prob
import hourly_usage_data as hud
from simulation_code import build_complete_digraph, simulation


def build_distributions(
    hourly_usage: Dict[str, np.ndarray],
    T: int,
    max_rate: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Uses hourly_requests per hub dictionary to createpoisson distribution of num requests per hour and corresponding time stamps for each different hub
    params: 
        hourly_usage- hourly requests per hub
        T- run time
        max_rate: max num of bikes per hub per hour
    returns:
        poisson distribution of num requests per hour and corresponding time stamps
    """
    poisson: Dict[str, np.ndarray] = {}
    timestamps: Dict[str, np.ndarray] = {}
    for hub_str, usage_profile in hourly_usage.items():
        rate_function = nhp.rate_from_hourly_profile(usage_profile.tolist())
        distributions = nhp.nonhomogenous_poisson(T, max_rate, rate_function)
        poisson[hub_str] = nhp.bin_events_by_hour(distributions, T)
        timestamps[hub_str] = distributions
    return poisson, timestamps


def run_simulation(
    hourly_usage: Dict[str, np.ndarray],
    *,
    T: int = 24,
    max_rate: int = 10,
    day: str = "T",
    seed: int = 42
) -> Tuple[np.ndarray[int], np.ndarray[int], Dict[str, List[float]]]:
    rng = np.random.default_rng(seed)

    distribution, timestamps = build_distributions(hourly_usage, T, max_rate)
    distribution = {str(k): v for k, v in distribution.items() if str(k) in sd.size_dictionary}

    G = build_complete_digraph(tm.travel_time, sd.size_dictionary)

    res = simulation(
        G,
        distribution,
        T=T,
        day=day,
        size_dictionary=sd.size_dictionary,
        elevation_matrix=em.elevation_matrix,
        travel_matrix=tm.travel_time,
    )

    return res[0], res[1], timestamps

if __name__ == "__main__":

    day = 'M'
    hourly_usage_5 = {}
    for key in hud.hourly_rates.keys():
        hourly_usage_5[key] = np.array(list(hud.hourly_rates[key][day]), dtype=int)


    res_5 = run_simulation(hourly_usage_5, T=24, max_rate=5)

    #hourly_usage profile checks
    print("\nExpected vs Simulated Events per Hub:")
    for hub, profile in hourly_usage_5.items():
        expected = profile.sum()
        simulated = len(res_5[2][hub])
        print(f"Hub {hub}: Expected {expected}, Simulated {simulated}")

    no_bike_events = np.asarray([res_5[0]])
    no_parking_events = np.asarray([res_5[1]])

    print(f"max rate = 10")
    print(f"no bike events: {res_5[0]}")
    print(f"no parking events: {res_5[1]}")
    print(f"timestamps: {res_5[2]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    (ax11, ax12) = axes

    ax11.bar(list(hourly_usage_5.keys()), [len(v) for v in res_5[2].values()])
    ax11.set_xlabel("Hub")
    ax11.set_ylabel("Total Events")
    ax11.set_title("Total Rental Events per Hub")

    ax12.plot(res_5[0], label="No-bike")
    ax12.plot(res_5[1], label="No-parking")
    ax12.set_xlabel("Hour")
    ax12.set_ylabel("Events")
    ax12.legend()
    ax12.set_title("Hourly Events")

    plt.tight_layout()
    plt.show()
