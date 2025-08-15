import numpy as np
from constants import size_dictionary
from constants import travel_time
from constants import elevation_matrix

"""
This program is no longer used in simulation_code or simulation_main. However, if one wants to
revert to using it for another city or town where the probabilites are not easily formable
subjectively/where one wants to use actual size data and geographic data to calculate the probability
of choosing a destination, it is here for use.
"""

#extract population size around certain hub on certain day of week at certain time of day
def extract_size(size_dictionary, hub, day, t):
    """
    Extract an innermost value from size_dictinionary, found under constants, but also available
    under test_data
    param:
        size_dictionary: a dictionary containing how many people are around a certain hub on a certain day of the week
        at a certain time of day
        hub: int, which hub the people are around
        day: int, which day of the week the data is extracted from (ex: "M" for Monday)
        t: int, which time of day the data is extracted from (ex: 14 for 2 pm)
    returns: 
        size: the integer value of the innermost dictionary, or the number of people around a certain hub on a certain day of the week
        at a certain time of day
    """

    #convert to strings to make extractable from size_dictionary
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
    """
    Calculates the attractiveness of each possible destination hub
        param:
            source: str, starting hub
            destination: str, one destination hub
            size_dictionary: data containing number of people around destination hub
            day: str, day of ride
            hour: str, time of day of ride
            beta1: int, weight for the impact of travel time from source to destination on attractiveness/utility
            beta2: int, weight for the impact of elevation difference from source to destination on utility
            lnsize: int weight for the impact of population size of destination hub on utility
            elevation_matrix: int adjacency matrix of size nxn where n = num hubs. rows = sources, cols = destinations. 
            matrix[i][j] = elevation difference between source i and destination j
            travel_matrix: int adjacency matrix of size nxn where n = num hubs. rows = sources, cols = destinations. 
            matrix[i][j] = time travel in minutes between source i and destination j
        returns:
            utility function: a measurement of how attractive a destination is. The higher the attractiveness, 
            the higher the likelihood of going to that destination is.
    """
    travel_time = travel_matrix[int(source)][int(destination)]
    elevation = elevation_matrix[int(source)][int(destination)]
    size = extract_size(size_dictionary, destination, day, hour) #find size at certain hour of the day
    size = max(size, 1)  # prevent log(0)
    size = size
    return -beta1 * travel_time - beta2 * elevation + lnSize * np.log(size)


#calculate the probability of choosing destination based on source
def probability(source, destination, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix):
    """
    Calculates the probability of going to destination d
    args: 
        same as utility
    returns:
        e^utility of destination d divided by e^utility of all other destinations added together
    """
    util = utility(source, destination, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix)
    utility_sum = 0
    hubs = list(size_dictionary.keys())
    for k in hubs:
        utility_sum += np.exp(utility(source, k, size_dictionary, day, hour, beta1, beta2, lnSize, elevation_matrix, travel_matrix)) #denomenator, all other destination utilites added together
    numerator = np.exp(util)
    denominator = utility_sum
    return numerator / denominator if denominator > 0 else 0


if __name__ == "__main__":
    source = '8'
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
