import numpy as np
from typing import List
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def rate_from_hourly_profile(hourly_requests: List[int]):
    """
    Creates a rate function based on the given hourly profile.
    Uses the raw hourly values as λ(t).
    """
    def get_rate_for_time(t: float) -> float:
        hour = int(t % len(hourly_requests))
        return hourly_requests[hour]
    return get_rate_for_time


def nonhomogenous_poisson(T: float, max_rate: float, rate_function) -> List[float]:
    """
    Simulates ride start times using the thinning algorithm for NHPP.
    T: total simulation time
    max_rate: maximum λ(t) used for thinning
    rate_function: function returning λ(t) for a given time
    """
    current_time = 0.0
    events = []
    while current_time < T:
        wait_time = np.random.exponential(1 / max_rate)
        current_time += wait_time
        if current_time >= T:
            break
        #accept with probability λ(t)/max_rate
        if np.random.uniform(0, 1) < (rate_function(current_time) / max_rate):
            events.append(round(current_time, 3))
    return events


def bin_events_by_hour(event_times: List[float], T: int) -> NDArray[np.int_]:
    """
    Bins continuous-time events into hourly counts.
    """
    hourly_bins = np.zeros(T, dtype=int)
    for time in event_times:
        hour = int(time % T)
        hourly_bins[hour] += 1
    return hourly_bins


def plot_poisson(hourly_bins: NDArray[np.int_]) -> None:
    """
    Bar chart of hourly bike ride counts
    """
    time_labels = [f"{h % 12 or 12}{' AM' if h < 12 else ' PM'}" for h in range(len(hourly_bins))]
    plt.figure(figsize=(12, 6))
    plt.bar(time_labels, hourly_bins, edgecolor='black')
    plt.xticks(rotation=45)
    plt.title("Bike Rides per Hour (NHPP Simulation)")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Rides")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hourly_requests = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1,
                       1, 0, 3, 1, 1, 1, 5, 5, 0, 1, 1, 0]

    total_simulation_hours = len(hourly_requests)
    max_hourly_rate = max(hourly_requests)

    rate_function = rate_from_hourly_profile(hourly_requests)
    simulated_events = nonhomogenous_poisson(total_simulation_hours, max_hourly_rate, rate_function)
    hourly_bins = bin_events_by_hour(simulated_events, total_simulation_hours)
    print("Hourly bins:", hourly_bins)

    plot_poisson(hourly_bins)

    rate_vals = [rate_function(t) for t in range(total_simulation_hours)]
    plt.figure(figsize=(10, 4))
    plt.plot(rate_vals, marker='o')
    plt.title("Rate Function Over Time (based on hourly profile)")
    plt.xlabel("Hour")
    plt.ylabel("Request Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
