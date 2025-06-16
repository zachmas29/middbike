# middbike
Potential bikeshare program in Middlebury

# Simulation Attempt One

To simulate a bikeshare program in Middlebury, we use Python to write a simple program to calculate user satisfaction. The system has 11 bike hubs, with roads connecting each pair of them. We use an 11-node, complete, directed graph to represent the system. Each node represents a hub. Between each pair of hubs is a pair of edges, each representing the time it takes to travel in one of the two directions (ex., between nodes a and b, an edge can be labeled 12 and points from a to b because it takes 12 minutes to travel from a to b; the other edge is labeled 20 because it takes longer to travel from b back to a). Based on the Poisson distribution, we provide a 24-element array "distribution" for each hub, each element representing the number of bike rentals happening in the past hour of the day (thus 24 elements). We also provide a 10-element array "possibilities" for each hub, each element representing the possibility that a user travels to one of the 10 hubs. So for example, if we're at hub a, and the possibility that a user rents a bike there to travel to location b is 20%, the first element of the "possibilities" array for location a should be 0.2. Assume, for the simplicity of discussion, bike rentals only happen at exact hours (ex., 2:00, 3:00, etc.), in the main function of this simulation algorithm, we calculate the bike distribution of our system at the next exact hour (i.e., how many bikes are free-floating in the system, how many bikes at each hub). We then calculate the sum of the number of times a user can't find an available bike at a hub (i.e., the number of remaining bikes at the hub is 0 at a given hour). We also calculate the sum of the number of times a user can't find a spot to park their bike (i.e., when a user arrives at a given hub, the number of bikes there is 10 or more). The algorithm returns these two numbers for each hour of the day.

# TODO: add Poisson distribution, probabilities array for each hub, and travel times

import numpy as np
import matplotlib.pyplot as plt

expected_rides=48
hourly_usage = [ 5,  4,  4,  5,  5,  3,  5,  4,  6,  7,  2,  5, 
5,  6,  4,  6,  6,  1,  7,  2, 11,  4,  3,  8]
num_bikes_hour = np.array(hourly_usage)
L = num_bikes_hour / np.sum(num_bikes_hour) * expected_rides
poisson_dist = np.random.poisson(L)
print(poisson_dist)

# #percent plot
# percentages = poisson_dist / poisson_dist.sum()
# plt.plot(range(24), percentages, marker='o', label='Usage %')
# plt.bar(range(24), percentages, alpha=0.4)
# plt.ylabel('Percent of Daily Usage')
# plt.show()

# # Create time labels
# time_labels = [f"{h % 12 or 12}{' AM' if h < 12 else ' PM'}" for h in range(24)]

# Plot
plt.figure(figsize=(12, 6))
plt.bar(time_labels, poisson_dist, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.title("Simulated Ebike Usage by Hour (Poisson Model)")
plt.xlabel("Time of Day")
plt.ylabel("Number of Rides")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
