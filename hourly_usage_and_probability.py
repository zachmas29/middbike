import numpy as np
from constants import size_dictionary
from constants import travel_time
from constants import elevation_matrix

#extract size variable used for probability utility function
def extract_size(size_dictionary, hub, day, t):
    hour = str(t)
    hub = str(hub)
    day = str(day)
    
    #navigate through nested dictionary structure
    hub_data = size_dictionary.get(hub, {})
    day_data = hub_data.get(day, {})
    size = day_data.get(hour,0)

    return size


#utility helper function to calculate the usefulness of each route
def utility(source, destination, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix):
    travel_time = travel_matrix[int(source)][int(destination)]
    elevation = elevation_matrix[int(source)][int(destination)]
    size = extract_size(size_dictionary, destination, day, hour) #find size at certain hour of the day
    size = max(size, 1)  # prevent log(0)
    size = size
    return -beta1 * travel_time - beta2 * elevation + lnSize * np.log(size)

#calculate the probability of choosing destination based on source
def probability(source, destination, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix):
    util = utility(source, destination, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix)
    utility_sum = 0
    hubs = list(size_dictionary.keys())
    for k in hubs:
        utility_sum += np.exp(utility(source, k, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix)) #denomenator, all other destination utilites added together
    numerator = np.exp(util)
    denominator = utility_sum
    return numerator / denominator if denominator > 0 else 0

if __name__ == "__main__":
    source = '0'
    destination = '1'
    day = 'T'
    hour = '10'
    beta1 = .25
    beta2 = .25
    lnSize = .75

    #test case
    p = probability(source, destination, size_dictionary, day, str(hour), beta1, beta2, lnSize, elevation_matrix, travel_time)
    print(p)
    print("\n--- Probabilities ---")

    total = 0 #add all probabilities to total value to see if all adds up to 1
    for k in size_dictionary.keys():
        try:
            prob = probability(source, k, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_time)
            print(f"P({k}) = {prob:.5f}") #f string format, ex: P(0) = 0.26508 (.5f = .5 decimal places)
            total += prob #increment till total = 1
        except Exception as e:
            print(f"Error computing P({k}): {e}")

    print(f"\nTotal = {total:.5f}")
