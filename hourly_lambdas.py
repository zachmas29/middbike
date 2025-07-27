import numpy as np
from typing import Dict, Sequence

def hourly_lambdas(
    population_distribution: Dict[int, Dict[int, np.ndarray]],
    prob: float
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Convert a nested population distribution into hourly integer lambdas
    for every hub and day of the week.

    Parameters
    ----------
    population_distribution : Dict[int -> Dict[int -> np.ndarray]]
        Outer keys 0-9  : hub IDs
        Inner keys 0-6  : day of week (Mon = 0 ,... ,Sun = 6)
        Leaf value      : 24-element 1-D array with the number of students
                          present each hour (index 0 = 00:00-01:00, ..., 23 = 23:00-24:00).
    prob : float
        Per-student trip probability (0 < prob <= 1).  Each hourly λ is
        calculated as population * prob

    Returns
    -------
    Dict[int -> Dict[int -> np.ndarray]]
        Same key structure, but each leaf array contains integer
        hourly Poisson intensities (lambdas)

    Raises
    ------
    ValueError
        If `prob` is not in the (0, 1] interval or a leaf array is not
        exactly length 24.
    """
    # probability validation 
    if not (0.0 < prob <= 1.0):
        raise ValueError("prob must be in the interval (0, 1]")

    lambdas: Dict[int, Dict[int, np.ndarray]] = {}

    for hub_id, day_dict in population_distribution.items():
        lambdas[hub_id] = {}
        for day, pop_vec in day_dict.items():
            pop_vec = np.asarray(pop_vec, dtype=float)

            if pop_vec.shape != (24,):
                raise ValueError(
                    f"population_distribution[{hub_id}][{day}] must be a 24-element vector."
                )

            # Compute λ = round(pop * prob) -> int array, non‑negative
            lam_vec = np.rint(pop_vec * prob).astype(int)
            lambdas[hub_id][day] = lam_vec

    return lambdas