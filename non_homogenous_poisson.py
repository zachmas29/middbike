import numpy as np
from typing import List
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def rate_from_hourly_profile(hourly_requests: List[int], max_rate: float):
    """
    Returns the rate of requests at a given hour, scaled to the maximum requests number per hour
    params: 
        hourly_requests: list of num of requests at each hour of the clock
        max_rate: the upper bound of num of requests per hour
    returns: 
        get_rate_for_time(t): scaled number of requests at time t
    """
    hourly_array = np.array(hourly_requests, dtype=float) #number of activations per hour
    total = np.sum(hourly_array)
    frequencies = hourly_array / total #frequency of request per hour

    #scale frequency by the max rate. Shows roughly how much of the max rate is used per hour
    def get_rate_for_time(t): 
        hour = int(t % 24) #get the hour by moduling the time by 24
        return frequencies[hour] * max_rate #normalize 
    return get_rate_for_time


def nonhomogenous_poisson(total_hours: float, max_rate: float, rate_function) -> List[float]:
    """
    Simulates ride start times using a non-homogeneous Poisson process.
    params:
        total_hours: range of hours in which bike requests are recorded
        max_rate: upper bound of num of requests per hour
        rate_function: rate_from_hourly_profile above
    returns:
        a distribution of probable times in which a user requested a bike
    """
    current_time = 0
    events = []
    while current_time < total_hours:
        wait_time = np.random.exponential(1 / max_rate) #random increments of time
        current_time += wait_time
        if current_time >= total_hours:
            break
        if np.random.rand() < rate_function(current_time) / max_rate: #if a random probability is less than the current rate of usage, the rate will be accepted
            time = round(current_time, 3)
            events.append(time)
    return events


def bin_events_by_hour(event_times: List[float]) -> NDArray[np.int_]:
    """
    Bins continuous-time events into hourly counts.
    params:
        event_times: list of nonhomogenous times in which a bike was requested
    returns:
        hourly_bins: slots of hours on the clock, each with the number of requests that happened within that hour
    """
    hourly_bins = np.zeros(24, dtype=int)
    for time in event_times:
        hour = int(time % 24) #find hour of event in the day
        hourly_bins[hour] += 1 #adds a recorded event to each hour slot
    return hourly_bins


def plot_poisson(hourly_bins: NDArray[np.int_]) -> None:
    """
    bar chart of hourly bike ride counts
    """
    time_labels = [f"{h % 12 or 12}{' AM' if h < 12 else ' PM'}" for h in range(24)]

    plt.figure(figsize=(12, 6))
    plt.bar(time_labels, hourly_bins, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45)
    plt.title("Bike Rides per Hour (NHPP Simulation)")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Rides")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hourly_requests = [5, 4, 4, 5, 5, 3, 5, 4, 6, 7, 2, 5,
                    5, 6, 4, 6, 6, 1, 7, 2, 11, 4, 3, 8]

    total_simulation_hours = 24
    max_hourly_rate = 100

    rate_function = rate_from_hourly_profile(hourly_requests, max_hourly_rate)
    print(str(rate_function))
    simulated_events = nonhomogenous_poisson(total_simulation_hours, max_hourly_rate, rate_function)
    print("Distribution:",simulated_events)
    hourly_bins = bin_events_by_hour(simulated_events)
    print(hourly_bins)
    # plot_poisson(hourly_bins)
