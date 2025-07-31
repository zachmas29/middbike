from typing import Dict, Tuple, List
from numpy.typing import NDArray
import numpy as np
from converted_population import converted_population
from testdata import size_dictionary
from constants import elevation_matrix, travel_time
import non_homogenous_poisson as nhp
import hourly_usage_and_probability as dest_prob
import simulation_code as code
import matplotlib.pyplot as plt

def build_distributions(
    hourly_lambdas: Dict[int, Dict[str, Dict[int, int]]],
    T: int, 
    day: str,
    num_hubs: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    
    poisson: Dict[int, np.ndarray] = {}
    timestamps: Dict[int, np.ndarray] = {}

    for hub in range(num_hubs):
        timestamps[hub] = nhp.nhp(hourly_lambdas[hub][day])
        poisson[hub] = nhp.bin_events_by_hour(timestamps[hub], T)
    return poisson, timestamps

def build_probabilities(
    size_dictionary: Dict[int, Dict[str, Dict[str, int]]],
    b1: float,
    b2: float,
    b3: float,
    day: str,
    num_hubs: int
    ) -> Dict[int, Dict[str, np.ndarray]]:

    res = {}
    for source in range(num_hubs):
        inner = {}
        for hour in range(24):
            temp = []
            for dest in range(num_hubs):
                if source == dest:
                    prob = 0.0
                    continue
                prob = dest_prob.probability(str(source), str(dest), size_dictionary, day, str(hour),
                                            b1, b2, b3,elevation_matrix,travel_time)
                temp.append(prob)
            inner[hour] = temp
        res[source] = inner

    return res

def run_simulation(
    max_bikes_per_hub: int,
    initial_bikes_per_hub: int,
    ) -> Tuple[int, int]:

    no_bike_sum = 0
    no_parking_sum = 0

    for _ in range(100):
        distribution = build_distributions(converted_population, 24, "M", 10)
        poisson = distribution[0]
        timestamps = distribution[1]
        probs = build_probabilities(size_dictionary, 0.25, 0.25, 0.75, "M", 10)
        graph = code.build_complete_digraph(travel_time)
        no_bike, no_parking, requests = code.simulation(graph, poisson, probs, max_bikes_per_hub=max_bikes_per_hub, initial_bikes_per_hub=initial_bikes_per_hub)
        no_bike_sum += no_bike.sum()
        no_parking_sum += no_parking.sum()

    return no_bike_sum/100, no_parking_sum/100  

if __name__ == "__main__":
    bikestock = [5, 10, 15, 20, 25]
    bikestands = [15, 30, 45, 60, 75]
    res = []
    for _ in range(5, 30, 5):
        events = run_simulation(_ * 2, _)
        res.append(events)
    print(res)

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
    ax11.set_title("Plot 1")

    ax12.scatter(bikestands, no_parking, s=60)
    ax12.set_xlabel("Initial Per-Hub Bikestands")
    ax12.set_ylabel("No-parking events")
    ax12.set_title("Plot 2")

    fig.subplots_adjust(hspace=0.5)
    plt.show()
