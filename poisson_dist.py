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
