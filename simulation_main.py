from typing import Dict, Tuple, List
from numpy.typing import NDArray
import numpy as np
from converted_population import converted_population
from testdata import size_dictionary
from constants import elevation_matrix, travel_time, size_dictionary
import non_homogenous_poisson as nhp
import hourly_usage_and_probability as dest_prob
import simulation_code as code
import matplotlib.pyplot as plt
import new_probability as nwp


def build_distributions(
    hourly_lambdas: Dict[int, Dict[str, Dict[int, int]]],
    T: int, 
    day: str,
    num_hubs: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Calls data from converted_population. Extracts a specific day's distribution data from each station key.
    Calls nonhomogenous poisson function to build a set of new bike request timestamps for each station, then bins the timestamps into hourly
    slots within the 24 hour clock. Returns dictionary with each station as key and its corresponding 24-hour request distribution as value.
    params:
        hourly lambdas: original dictionary with station id's as keys (0-9), and days of the week dictionaries (keys = "M","T","W","R","F") as values. Those dictionaries
        contain days of the week as keys, and distribution dictionaries of size 24 as values. Those innermost dictionaries contain hour ints as keys,
        and int num requests per hour as values. These numbers of bike activations per hour are inserted into nonhomogenous poisson to generate new, probable
        timestamps for new runs of the simultation.
        T: the amount of time nonhomogenous poisson runs for (should be 24 hours)
        day: each hourly lambda array represents 1 day worth of data. This day parameter specifies which day from the data you're using.
        num_hubs: the number of bike stations
    returns:
        poisson: a dictionary with each station as a key and their corresponding nonhomogenous request distribution as a value.
        timestamps: the nonhomogenous set of times at which each request occurs
    """
    poisson: Dict[int, np.ndarray] = {}
    timestamps: Dict[int, np.ndarray] = {}

    for hub in range(num_hubs):
        timestamps[hub] = nhp.nhp(hourly_lambdas[hub][day]) #timestamps within T = 24 hours for station "hub"
        poisson[hub] = nhp.bin_events_by_hour(timestamps[hub], T) #bin timestamps withn 24 hour slots for each hub
    return poisson, timestamps


def build_probabilities(
    num_hubs: int
    ) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For each source, calculates the probs to travel from source to destinations 0-9 for each hour of the day.
    params: 
        num_hubs: number of bike stations
    returns: 
        res: dictionaries with station (sources) as keys, and dictionaries of size 24 as values.
        Within these 24 value dictionaries, each key is an hour of the day, and each value is an array of size num_hubs - 1 that
        shows the probabilites of going from the source (the outermost key) to each other station at a specific hour.
    """
    res = {}
    for source in range(num_hubs):
        inner = {}
        for hour in range(24):
            temp = []
            for dest in range(num_hubs):
                if source == dest:
                    prob = 0.0
                prob = nwp.calculate_probability(hour,source,dest)
                temp.append(prob)
            inner[hour] = temp
        res[source] = inner
    return res


def run_simulation(
    max_bikes_per_hub: int,
    initial_bikes_per_hub: int,
    ) -> Tuple[int, int]:
    """
    Runs simulation_code for a specific day of the week. 
    """
    no_bike_sum = 0
    no_parking_sum = 0

    for _ in range(100):
        distribution = build_distributions(converted_population, 24, "W", 10)
        poisson = distribution[0]
        timestamps = distribution[1]
        probs = build_probabilities(10)
        graph = code.build_complete_digraph(travel_time)
        no_bike, no_parking, requests = code.simulation(graph, poisson, probs, max_bikes_per_hub=max_bikes_per_hub, initial_bikes_per_hub=initial_bikes_per_hub)
        no_bike_sum += no_bike.sum()
        no_parking_sum += no_parking.sum()

    return no_bike_sum/100, no_parking_sum/100  

if __name__ == "__main__":
    bikestock = [5, 10, 15, 20, 25]
    bikestands = [10, 20, 30, 40, 50]
    res = []
    for _ in range(5, 30, 5):
        events = run_simulation(_*2, _)
        res.append(events)

    no_bike = []
    no_parking = []
    for _ in res:
        no_bike.append(_[0])
        no_parking.append(_[1])
    
    print(no_bike)
    print(no_parking)
    
    fig, axes = plt.subplots(2, 1, figsize = (8, 6))
    (ax11, ax12) = axes

    ax11.scatter(bikestock, no_bike, s=60)
    ax11.set_xlabel("Initial Per-Hub Bikestock")
    ax11.set_ylabel("No-bike events")
    ax11.set_title("No-Bike Events vs. Bike Stock Per-Hub")

    ax12.scatter(bikestands, no_parking, s=60)
    ax12.set_xlabel("Initial Per-Hub Bikestands")
    ax12.set_ylabel("No-parking events")
    ax12.set_title("No-Parking Events vs. Num Bike Stands Per-Hub")
    ax12.set_ylim(0, 20)


    fig.subplots_adjust(hspace=0.5)
    plt.show()
