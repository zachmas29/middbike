import numpy as np
from typing import Sequence
from hourly_lambdas import hourly_lambdas
from testdata import population_distribution

def nhpp(
    hourly_lambdas: Sequence[int],
    *,
    seed: int | None = None
) -> np.ndarray:
    """
    Simulate one 24-hour day of requests as a non-homogeneous Poisson
    process (NHPP) using Lewis-Shedler thinning

    Parameters
    ----------
    hourly_lambdas : Sequence[int] (length = 24) - Expected events in each hour (0-23)
    seed : int or None, optional - Seed for NumPy's random generator

    Returns
    -------
    Sorted array of event times (floats) in fractional hours between 0 and 24
    """
    # validate hourly_lambdas
    if len(hourly_lambdas) != 24:
        raise ValueError("hourly_lambdas must contain exactly 24 values")

    # validate integer property
    lam = np.asarray(hourly_lambdas, dtype=int)
    if np.any(lam < 0):
        raise ValueError("hourly_lambdas must all be non-negative integers.")

    lam_max = lam.max()
    if lam_max == 0:
        return np.empty(0, dtype=float)       # no arrivals at all if max-lambda = 0

    rng = np.random.default_rng(seed)
    horizon = 24.0
    t = rng.exponential(1 / lam_max)   # first candidate time
    events = []

    # Piece‑wise constant λ(t)
    def lam_t(time: float) -> int:
        idx = min(int(time), 23)              # clamp 23.999… to 23
        return lam[idx]

    # thinning loop
    while t < horizon:
        if rng.random() < lam_t(t) / lam_max:  # accept with prob λ(t)/λ_max
            events.append(t)
        t += rng.exponential(1 / lam_max)      # next candidate

    return np.asarray(events)

if __name__ == "__main__":
    print(nhpp(hourly_lambdas(population_distribution, 0.15)[1][1]))