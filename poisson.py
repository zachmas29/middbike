"""
Zachary Okayli Masaryk
Bikeshare Hourly Activations rendered into Poisson Distribution
6/11/2025
"""
import numpy as np
from numpy.typing import NDArray
from typing import List
import matplotlib.pyplot as plt

def poisson(expected_rides: int, hourly_usage: List[int]) -> NDArray[np.int_]:
    """
    Creates and prints a poisson distribution with each index representing an hour of the 24-hour clock, and each value being the number of bikes activated in that hour
    params: 
        expected_rides_per_hour: expected number of bikes used per hour
        hourly_usage: number of bikes used per hour
    returns: 
        simulated distribution of bikes activations per hour based on data in hourly_usage
    """
    hourly_array = np.array(hourly_usage)
    L = hourly_array / np.sum(hourly_array) * expected_rides
    return np.random.poisson(L)


def plot_poisson(poisson_dist: NDArray[np.int_])-> None:
    """
    Plot hourly distribution
    """
    time_labels = [f"{h % 12 or 12}{' AM' if h < 12 else ' PM'}" for h in range(24)]

    plt.figure(figsize=(12, 6))
    plt.bar(time_labels, poisson_dist, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45)
    plt.title("Usage by Hour (Poisson Model)")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Rides")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def best_time_frames(hourly_usage: List[int], poisson_dist: np.ndarray)-> List[str]:
    """
    Organizes poisson distribution into time frames and sorts them from highest bike activations to lowest
    params: 
        hourly_usage: number of bikes used per hour
        poisson_dist: distribution of usage frequencies based on time of day
    returns:
        array sorting highest frequency time frames to lowest
    """
    # Form overlapping 3-hour windows from the poisson dist
    ranges = []
    labels = []
    for i in range(0, len(hourly_usage) - 2):
        ranges.append(poisson_dist[i:i+3])
        labels.append(i)  # record starting hour of each range

    # Get the total rides in each 3-hour window
    sums = [np.sum(r) for r in ranges]

    # Sort 3-hour windows from highest usage frequency to lowest
    ranking_distr_ranges = []
    for _ in range(len(ranges)):
        index_of_max = sums.index(max(sums))
        start = labels[index_of_max]
        end = start + 2
        start_label = f"{start % 12 or 12}{' AM' if start < 12 else ' PM'}"
        end_label = f"{end % 12 or 12}{' AM' if end < 12 else ' PM'}"
        ranking_distr_ranges.append(f"{start_label} to {end_label}")
        sums[index_of_max] = -np.inf  # mark this one as used so it's not picked again
    return "Time frames sorted by highest to lowest usage: " + str(ranking_distr_ranges)


if __name__ == "__main__":
    hourly_usage = [0, 1, 1, 1, 2, 3, 3, 5, 8, 8, 6, 5,
                    9, 6, 4, 6, 7, 8, 7, 5, 4, 4, 3, 2]
    expected_rides_per_hour = 55
    poisson_dist = poisson(expected_rides_per_hour, hourly_usage)
    sorted_best_times = best_time_frames(hourly_usage, poisson_dist)
    print("poisson_dist:", poisson_dist)
    print(sorted_best_times)
