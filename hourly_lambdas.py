import numpy as np
from typing import Dict
from testdata import size_dictionary
import json


def hourly_lambdas(
    population_distribution: Dict[int, Dict[str, Dict[str, int]]],
    prob: float,
    cap: int = 5
) -> Dict[int, Dict[str, Dict[str, int]]]:
    """
    Convert a nested population distribution into hourly integer lambdas
    for every hub and weekday.

    param:
        population_distribution : Dict[int -> Dict[str -> Dict[str -> int]]]
            Outer keys: hub IDs (0-9)
            Middle keys: day of week ("M", "T", "W", "R", "F")
            Inner dict: hour as string ("0" to "23"), value = population count
        prob : float
            Per-student trip probability (0 < prob <= 1).
        cap : int
            Maximum allowed lambda value. Default is 5.
    returns:
        Dict[int -> Dict[str -> Dict[str -> int]]]
            Same key structure, but each innermost value is λ
            (rounded integer Poisson intensity).
    """
    if not (0.0 < prob <= 1.0):
        raise ValueError("prob must be in the interval (0, 1]")

    lambdas: Dict[int, Dict[str, Dict[str, int]]] = {}

    for hub_id, days in population_distribution.items():
        lambdas[hub_id] = {}
        for day, hours in days.items():
            lambdas[hub_id][day] = {}
            for hour, pop in hours.items():
                lam = int(round(min(pop * prob, cap)))  # Apply min rule
                lambdas[hub_id][day][hour] = lam
    return lambdas


def write_converted_population_file(data: Dict[int, Dict[str, Dict[str, int]]], filename: str = "converted_population.py"):
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
    combined_result: Dict[int, Dict[str, Dict[str, int]]] = {}

    for hub_id, data in size_dictionary.items():
        if hub_id in [0, 1, 2, 8]:
            prob = 0.05
        else:
            prob = 0.03
        hub_result = hourly_lambdas({hub_id: data}, prob, cap=5)
        combined_result[hub_id] = hub_result[hub_id]

    # Write combined result to file
    write_converted_population_file(combined_result)
