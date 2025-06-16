from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

def build_complete_digraph(travel_time: np.ndarray) -> nx.DiGraph:
    """
    Build a complete digraph whose edge attribute 'time' holds one-way travel
    time in minutes. 
    ----------------
    Parameters:
    travel_time - [i, j] represents the travel time in mins from i to j

    Returns:
    G - complete digraph G populated based on data provided
    """
    n = travel_time.shape[0] # should be 11, since the matrix [11, 11][0] = 11
    G = nx.complete_graph(n, create_using = nx.DiGraph)
    for u, v in G.edges: # for edge uv, the label time = travel_time[u, v]
        G.edges[u, v]["time"] = int(travel_time[u, v])
    return G

def simulation(
        G: nx.DiGraph,
        distribution: Dict[int, np.ndarray],
        possibilities: Dict[int, List[float]],
        *,
        max_bikes_per_hub: int = 10,
        initial_bikes_per_hub: int = 5,
        rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    G - K_11 generated from data
    distribution - 24-element np.ndarray hourly rental requests at each hub
    possibilities - 11-element destination probabilities for each hub, [origin][origin] must be 0.0
    keyword args - must be passed with name
        max_bikes_per_hub - 10
        initial_bikes_per_hub - 5 for simplicity 
        rng - NumPy generator for reproducibility

    Returns:
    no_bike_events - 24-element np.ndarray representing the no. of no-bike events every hour in the system
    no_parking_events - 24-element np.ndarray representing the no. of no-space events every hour in the system

    """
    
    # initialize NumPy random number generator
    if rng is None:
        rng = np.random.default_rng()


    num_hubs = G.number_of_nodes()
    bike_stock = np.full(num_hubs, initial_bikes_per_hub, dtype = int) # 11-element array, no. of bikes at each hub

    # trips currently on the road, each element (minutes_remaining, destination_hub)
    in_transit: List[Tuple[int, int]] = []

    no_bike_events = np.zeros(24, dtype = int) # sum of no-bike events, each hour of the day
    no_parking_events = np.zeros(24, dtype = int) # sum of no-parking events, each hour of the day

    for hour in range(24):

        # for simplicity, advance all in-transit bikes by 60 mins
        # dock whose remaining time has hit zero or below
        updated_trips: List[Tuple[int, int]] = []
        
        for minutes_left, dest in in_transit:
            minutes_left -= 60
            if minutes_left <= 0:
                # attempt to dock a bike at dest
                if bike_stock[dest] < max_bikes_per_hub:
                    bike_stock[dest] += 1
                else:
                    no_parking_events[hour] += 1
            else:
                updated_trips.append((minutes_left, dest)) # if a trip not done then append to next hour
        in_transit = updated_trips

        # process rental requests that occur during this hour
        for hub in range(num_hubs):
            requests = int(distribution[hub][hour]) # no. of requests from distribution array
            for _ in range(requests):
                # attempt to rent at hub
                if bike_stock[hub] == 0: # if no bike left
                    no_bike_events[hour] += 1
                    continue

                # successful checkout
                bike_stock[hub] -= 1
                dest = rng.choice(num_hubs, p=possibilities[hub]) # assign destination based on probability array

                #trip duration from edge attribute in G
                minutes = int(G.edges[hub, dest]["time"])
                in_transit.append((minutes, dest))

    return no_bike_events, no_parking_events

    
