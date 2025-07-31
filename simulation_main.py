from typing import Dict, Tuple, List
from numpy.typing import NDArray
import numpy as np
from converted_population import converted_population
from testdata import size_dictionary
from constants import elevation_matrix, travel_time
import non_homogenous_poisson as nhp
import hourly_usage_and_probability as dest_prob
import simulation_code as code

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

if __name__ == "__main__":
    distribution = build_distributions(converted_population, 24, "M", 10)
    poisson = distribution[0]
    timestamps = distribution[1]
    probs = build_probabilities(size_dictionary, 0.25, 0.25, 0.75, "M", 10)
    graph = code.build_complete_digraph(travel_time)
    no_bike, no_parking, requests = code.simulation(graph, poisson, probs)
    print(no_bike)
    print(no_parking)
    
    