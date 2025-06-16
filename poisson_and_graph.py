import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

"""
generate poisson distribution array
"""
expected_rides=48 #change manually
hourly_usage = [ 5,  4,  4,  5,  5,  3,  5,  4,  6,  7,  2,  5, 
5,  6,  4,  6,  6,  1,  7,  2, 11,  4,  3,  8] #change manually
num_bikes_hour = np.array(hourly_usage)
L = num_bikes_hour / np.sum(num_bikes_hour) * expected_rides
poisson_dist = np.random.poisson(L)
print(poisson_dist)

"""
plot poisson distribution
"""
plt.figure(figsize=(12, 6))
plt.bar(time_labels, poisson_dist, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.title("Simulated Ebike Usage by Hour (Poisson Model)")
plt.xlabel("Time of Day")
plt.ylabel("Number of Rides")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

matrix = [[0, 11, 11, 6, 6, 13, 6, 10, 8, 7],
    [7, 0, 6, 4, 8, 9, 5, 6, 3, 4],
    [11, 7, 0, 5, 10, 12, 7, 9, 4, 4],
    [5, 5, 5, 0, 6, 9, 3, 6, 3, 1],
    [4, 11, 11, 5, 0, 7, 4, 10, 8, 6],
    [11, 11, 13, 10, 8, 0, 8, 7, 10, 10],
    [5, 8, 9, 3, 4, 5, 0, 8, 6, 4],
    [8, 6, 9, 5, 9, 7, 6, 0, 6, 6],
    [6, 5, 3, 2, 7, 9, 4, 5, 0, 1],
    [5, 5, 5, 1, 6, 9, 3, 6, 2, 0]]

"""
param: matrix with each index representing the time it takes to get from x hub to y hub
returns: map with hubs as nodes and edges as bike time between nodes (minutes)
"""
def create_graph(matrix):
    num_hubs = len(matrix)
    map = nx.DiGraph()
    for i in range(num_hubs):
        map.add_node(i)
    for i in range(num_hubs):
        for j in range(num_hubs):
            if i != j:
                map.add_edge(i, j, time=matrix[i][j]) 
    return map
