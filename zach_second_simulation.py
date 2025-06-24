"""
Zachary Okayli Masaryk
Frequency of Usage and Full/Empty Events
6/24/2025
"""

import random
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from non_homogenous_poisson import (
    rate_from_hourly_profile,
    nonhomogenous_poisson,
    bin_events_by_hour
)

def simulation(hourly_usage, travel_matrix, T, max_rate, max_bikes_per_hub):
    """
    Simulates the flow of bikes from station (hub) to station, updating every time a bike is activated. The
    activation times are pre-diagrammed using a nonhomogenous poisson distribution, which is based on previous bike_share data (hourly_usage).
    The function also tracks the total stock of bikes at each station during each activation, and records when bike stand is empty or full
    during an attempting activation or docking.
    params:
        hourly_usage: a pre-recorded array diagramming the number of times a bike was activated per hour on the 24-hour clock
        travel_matrix: an adjacency matrix, with each index of rows and columns being a bike station. Each row, column value 
        is the time it takes to get from one station to another.
        T: the number of hours the simulation should run (should be a multiple of 24)
        max_bikes_per_hub: the maximum capacity of each bike station
    returns:
        prints a dictionary of both number of times bike station is full (value) per hour (key) and the number of times a station is empty
        prints a dictionary of the stock (value) at each bike station (key)
    """
    rate_function = rate_from_hourly_profile(hourly_usage, max_rate) #calls an inner function, rate(t), which returns the hourly_usage at a specified time (t) from hourly_usage, divides it by total_usages in hourly_usage, and scales that rate value by max_rate
    non_hom_poisson = nonhomogenous_poisson(T, max_rate, rate_function) #distribution array with each value being each hour:minute a bike was activated within time range T
    hubs = list(range(10)) #each bike stand, labeled 0->9
    bike_stock = {hub: max_bikes_per_hub // 2 for hub in hubs} #initial num of bikes at each hub

    no_bike_events = defaultdict(int)
    no_parking_events = defaultdict(int)
    dest_IDs = defaultdict(list) #bike IDs that arrive and are docked at each destination (dest)
    durations_per_ID = defaultdict(float) #total travel time from source (src) to destination

    sources = [] #sources each time a bike is activation
    dests = [] #destinations each time a bike is docked
    durations = []
    for t in non_hom_poisson: #iterates every time a new bike is activated (each value of nonhomogenous poisson distribution)
        src = random.choice(hubs) #randomly chooses hub (CHANGE: make it based on probability)

         #if bike station is empty, do not activate bike
        if bike_stock[src] == 0:
            hour = int(t % 24)
            no_bike_events[hour] += 1
        else:
            #bike successfully activated
            sources.append(src)
            user_ID = f"user_{random.randint(1000, 9999)}" #assign value to a userID
            bike_stock[src] -= 1

            #randomly choose dest station
            dest = random.choice([h for h in hubs if h != src]) 
            travel_time = travel_matrix[src][dest] #travel time btw src and dest
            end_time = t + travel_time #hour:minute a bike completes a ride

            #station is full, do not dock
            if bike_stock[dest] == max_bikes_per_hub:
                hour = int(end_time % 24)
                no_parking_events[hour] += 1
                dests.append('n/a')
                durations.append('not ended')
            else:
                #successful docking
                dests.append(dest)
                bike_stock[dest] += 1
                dest_IDs[dest].append(user_ID) #dock specified bike at station
                ride_time = end_time - t
                durations_per_ID[user_ID] = ride_time
                durations.append(ride_time)


    print("Distribution:",nonhomogenous_poisson)
    print("Source:",sources)
    print("Destinations:",dests),
    print("Durations:",durations)
    print("No Bike Events:", dict(no_bike_events))
    print("No Parking Events:", dict(no_parking_events))
    print("Bike Stock:", bike_stock)



if __name__ == "__main__":

    hourly_usage = [0, 2, 0, 0, 0, 3, 3, 3, 2, 2, 2, 1, 1, 0, 3, 1, 1, 1, 6, 7, 0, 1, 1, 0]

    #10x10 matrix of travel time between destinations (hubs 0-9)
    travel_matrix = np.array([
    [0, 11, 11, 6, 6, 13, 6, 10, 8, 7],
    [7, 0, 6, 4, 8, 9, 5, 6, 3, 4],
    [11, 7, 0, 5, 10, 12, 7, 9, 4, 4],
    [5, 5, 5, 0, 6, 9, 3, 6, 3, 1],
    [4, 11, 11, 5, 0, 7, 4, 10, 8, 6],
    [11, 11, 13, 10, 8, 0, 8, 7, 10, 10],
    [5, 8, 9, 3, 4, 5, 0, 8, 6, 4],
    [8, 6, 9, 5, 9, 7, 6, 0, 6, 6],
    [6, 5, 3, 2, 7, 9, 4, 5, 0, 1],
    [5, 5, 5, 1, 6, 9, 3, 6, 2, 0]])

    T = 24
    max_rate = 50
    max_bikes_per_hub = 10
    simulation(hourly_usage, travel_matrix, T, max_rate, max_bikes_per_hub)

    


