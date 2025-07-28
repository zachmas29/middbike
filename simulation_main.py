from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from non_homogenous_poisson import nhpp
from hourly_lambdas import hourly_lambdas
from simulation_code import build_complete_digraph, simulation
from testdata import population_distribution

def daily_hub_demands(
    prob: float,
    day: int,                # 0 = Monday ... 6 = Sunday
    seed: int | None = None
) -> Dict[int, np.ndarray]:
    
    lambdas = hourly_lambdas(population_distribution, prob)

    rng = np.random.default_rng(seed)
    demands: Dict[int, np.ndarray] = {}

    for hub_id, day_dict in lambdas.items():
        hourly_vec = day_dict[day]           # 24‑element int array
        demands[hub_id] = nhpp(hourly_vec)
    
    return demands

def build_destination_probabilities(
    day: int,
    hour: int,
    *,
    beta1: float = 0.04,
    beta2: float = 0.05,
    lnSize: float = 0.03
) -> Dict[int, Dict[int, float]]:
    pass

def run_simulation():
    pass