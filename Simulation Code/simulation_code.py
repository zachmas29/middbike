from __future__ import annotations
import random
import numpy as np
import networkx as nx
import hourly_usage_and_probability as pr
from typing import Dict, List, Tuple
from request import Request

def build_complete_digraph(travel_time, size_dictionary):
    """
    Build a complete digraph using only the hubs in size_dictionary.
    Each edge's 'time' attribute holds one-way travel time in minutes.

    Parameters:
    travel_time - 2D array where [i, j] represents the travel time in minutes from i to j
    size_dictionary - dictionary of active hubs (e.g., {'0': ..., '2': ..., '8': ...})

    Returns:
    G - directed graph with only relevant hubs and correct travel times
    """
    hub_ids = sorted(int(h) for h in size_dictionary.keys())  # get numeric hub IDs
    G = nx.DiGraph()
    G.add_nodes_from(hub_ids)

    for u in hub_ids:
        for v in hub_ids:
            if u != v:
                G.add_edge(u, v, time=int(travel_time[u][v]))

    return G

def simulation(
        G,
        distribution,  # expects { "0": array([...]), "2": array([...]), ... }
        *,
        day,
        T,
        max_bikes_per_hub=10,
        initial_bikes_per_hub=5,
        rng=None,
        size_dictionary=None,
        elevation_matrix=None,
        travel_matrix=None,
):
    if rng is None:
        rng = np.random.default_rng()

    hub_list = sorted(int(h) for h in size_dictionary.keys())
    hub_to_index = {hub: idx for idx, hub in enumerate(hub_list)}
    index_to_hub = {idx: hub for hub, idx in hub_to_index.items()}

    bike_stock = np.full(len(hub_list), initial_bikes_per_hub, dtype=int)

    req_pool = {}
    all_requests = []
    for hub_str in size_dictionary.keys():
        hub = int(hub_str)
        n_req = int(distribution[hub_str].sum())
        bag = [Request() for _ in range(n_req)]
        req_pool[hub] = bag
    
    in_transit = []

    no_bike_events = np.zeros(T, dtype=int)
    no_parking_events = np.zeros(T, dtype=int)

    for hour in range(T):
        on_road = []

        for req in in_transit:
            req.minutes_left -= 60
            if req.minutes_left > 0:
                on_road.append(req)
                continue

            dest = req.dest
            if bike_stock[hub_to_index[dest]] < max_bikes_per_hub:
                bike_stock[hub_to_index[dest]] += 1
                continue

            no_parking_events[hour] += 1
            candidate = None
            for offset in range(1, len(hub_list)):
                for new_hub in (dest - offset, dest + offset):
                    if new_hub in hub_to_index:
                        candidate = new_hub
                        break
                if candidate is not None:
                    break

            if candidate is None:
                req.minutes_left = 60
                on_road.append(req)
                continue

            extra_time = int(G.edges[dest, candidate]["time"])
            req.dest = candidate
            req.minutes_left = extra_time
            on_road.append(req)

        in_transit = on_road

        for hub_str in size_dictionary.keys():
            hub = int(hub_str)
            for _ in range(int(distribution[hub_str][hour])):
                req = req_pool[hub].pop() if req_pool[hub] else Request()
                all_requests.append(req)
                req.origin = hub

                if bike_stock[hub_to_index[hub]] == 0:
                    no_bike_events[hour] += 1
                    req.success = False
                    continue

                bike_stock[hub_to_index[hub]] -= 1

                probs = []
                destinations = [d for d in size_dictionary.keys() if d != hub_str]
                probs = []
                for dest in destinations:
                    p = pr.probability(str(hub), dest, size_dictionary, day, str(hour),
                    beta1=0.04, beta2=0.05, lnSize=0.003,
                    elevation_matrix=elevation_matrix,
                    travel_matrix=travel_matrix)
                    probs.append(p)

                probs = np.array(probs)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
                dest_str = random.choices(destinations, weights=probs, k=1)[0]
                dest = int(dest_str)

                dest_str = random.choices(destinations, weights = probs,k=1)[0]
                dest = int(dest_str)

                prob_selected = probs[destinations.index(dest_str)]
                print(f"Hour {hour}: Source {hub} → Dest {dest} (P={prob_selected:.4f})")

                if hub in G.nodes and dest in G.nodes and G.has_edge(hub, dest):
                    req.minutes_left = int(G.edges[hub, dest]["time"])
                else:
                    req.minutes_left = 999

                req.dest = dest
                req.success = True
                in_transit.append(req)

    return no_bike_events, no_parking_events, all_requests


