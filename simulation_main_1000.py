import numpy as np
import matplotlib.pyplot as plt
from simulation_main import run_simulation
import hourly_usage_data as hud

def run_batch_simulation(iterations=1000, day='M', T=24, max_rate=5):
    hourly_usage = {}
    for key in hud.hourly_rates.keys():
        hourly_usage[key] = np.array(list(hud.hourly_rates[key][day].values()), dtype=int)

    total_no_bike = np.zeros(T, dtype=float)
    total_no_parking = np.zeros(T, dtype=float)

    for i in range(iterations):
        res = run_simulation(hourly_usage, T=T, max_rate=max_rate)
        total_no_bike += res[0]
        total_no_parking += res[1]

    avg_no_bike = np.round(total_no_bike / iterations).astype(int)
    avg_no_parking = np.round(total_no_parking / iterations).astype(int)

    print(f"After {iterations} runs:")
    print("Average no-bike events per hour:", avg_no_bike.tolist())
    print("Average no-parking events per hour:", avg_no_parking.tolist())
    print(f"Total average no-bike events: {avg_no_bike.sum()}")
    print(f"Total average no-parking events: {avg_no_parking.sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    (ax1, ax2) = axes

    ax1.bar(range(T), avg_no_bike, color="blue", alpha=0.7)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Avg No-bike Events")
    ax1.set_title("Average No-bike Events per Hour")

    ax2.bar(range(T), avg_no_parking, color="orange", alpha=0.7)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Avg No-parking Events")
    ax2.set_title("Average No-parking Events per Hour")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_batch_simulation(iterations=1000)
