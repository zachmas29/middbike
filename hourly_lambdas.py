import numpy as np
from typing import Dict
from testdata import size_dictionary
import json

rng = np.random.default_rng()
def hourly_lambdas(
    population_distribution: Dict[int, Dict[str, Dict[str, int]]],
    prob: float,
) -> Dict[int, Dict[str, Dict[str, int]]]:
    """
    Generate hourly lambda values using Poisson sampling for variability.
    param:
        population_distribution: 3 embedded dictinionaries. The first is keys = hubs (ints 0-9) and values = days (dicts). 
        These values are dictiories with each day of the week as keys, represented by their starting letter. So keys = ['M','T','W','R','F'].
        The values of these dictionaries are dictionaries of size 24, representing each hour of the day, so the keys are
        ["0","1",...,n=24]. The values of each hour represent a certain number of people. So returning one value from
        this dictionary would tell you the number of people at a certain hour at a certain hub on a certain day of the week. See 
        test_data.py's size_dictionary for reference.
    return:
        lambdas: dictinionary of the same structure as population_distribution
        containing the distribution of bike rental requests at each hub on a certain day of the week. However, the 
        innermost dictionary has
    """
    if not (0.0 < prob <= 1.0):
        raise ValueError("prob must be in (0, 1]")
    lambdas: Dict[int, Dict[str, Dict[str, int]]] = {}

    for hub_id, days in population_distribution.items():
        lambdas[hub_id] = {}
        for day, hours in days.items():
            lambdas[hub_id][day] = {}
            for hour, pop in hours.items():
                if hub_id in ["2","8"] and 8<=int(hour)<=14:
                    mean = pop / 20 * prob
                elif hub_id == "1" and int(hour) == 7:
                    mean = pop / 15 * prob
                else:
                    mean = pop/8 * prob
                lam = int(rng.poisson(mean))  # introduce randomness
                lambdas[hub_id][day][hour] = lam
    return lambdas

#VERY IMPORTANT COMMENT
#new hourly lambdas dictinoary is written to a new file called CONVERTED_POPULATION.PY
def write_converted_population_file(data: Dict[int, Dict[str, Dict[str, int]]], filename):
    """
    Write the converted population dictionary to a Python file in
    the same style as population_distribution.
    """
    pretty_str = json.dumps(data, indent=4)

    # Fix JSON quirks: remove quotes from top-level integer keys
    lines = pretty_str.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('"') and line.strip()[1].isdigit():
            line = line.replace('"', '', 1)  # remove first quote
            line = line.replace('":', ':', 1)  # remove quote before colon
        new_lines.append(line)
    formatted_output = "\n".join(new_lines)

    with open(filename, "w") as f:
        f.write("converted_population = ")
        f.write(formatted_output)

    print(f"Formatted converted_population has been written to {filename}")


if __name__ == "__main__":

    rng = np.random.default_rng()
    combined_result = {}
    for hub_id, data in size_dictionary.items():
        prob = 0.1 if hub_id in ["0", "2", "8"] else 0.1
        hub_result = hourly_lambdas({hub_id: data}, prob)
        combined_result[hub_id] = hub_result[hub_id]


    # Write combined result to file
    write_converted_population_file(combined_result, "/Users/zachokaylimasaryk/Downloads/middbike/middbike/converted_population.py")

