#PROGRAM NO LONGER IN USE, ASK SHERMAN TO SEE IF IT'S IMPORTANT

from typing import Dict, Tuple, List
from numpy.typing import NDArray
import numpy as np
from converted_population import converted_population
from testdata import size_dictionary
from constants import elevation_matrix, travel_time, size_dictionary
import non_homogenous_poisson as nhp
import hourly_usage_and_probability as dest_prob
import simulation_code as code
import matplotlib.pyplot as plt


def build_probabilities(
    size_dictionary: Dict[int, Dict[str, Dict[str, int]]],
    b1: float,
    b2: float,
    b3: float,
    day: str,
    num_hubs: int
    ) -> Dict[int, Dict[str, np.ndarray]]:

    res = {}
    for source in range(num_hubs):
        inner = {}
        for hour in range(24):
            temp = []
            for dest in range(num_hubs):
                if source == dest:
                    prob = 0.0
                    continue
                prob = dest_prob.probability(str(source), str(dest), size_dictionary, day, str(hour),
                                            b1, b2, b3,elevation_matrix,travel_time)
                temp.append(prob)
            inner[hour] = temp
        res[source] = inner
    return res