import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple
from hourly_lambdas import hourly_lambdas
from converted_population import converted_population

def nhp(
    raw_hourly_lambdas: Dict[int, int],
    *,
    seed: Optional[int] = None
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
    hourly_lambdas = list(raw_hourly_lambdas.values())

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

    return np.sort(np.asarray(events))


def bin_events_by_hour(event_times: List[float], T: int) -> NDArray[np.int_]:
    """
    Bins continuous-time events into hourly counts.
    params:
        event_times: list of nonhomogenous times in which a bike was requested
    returns:
        hourly_bins: slots of hours on the clock, each with the number of requests that happened within that hour
    """
    hourly_bins = np.zeros(T, dtype=int)
    for time in event_times:
        hour = int(time % 24) #find hour of event in the day
        hourly_bins[hour] += 1 #adds a recorded event to each hour slot
        hour = int(time % T)
        hourly_bins[hour] += 1
    return hourly_bins

if __name__ == "__main__":
    dist = nhp(converted_population[1]["M"])
    print(bin_events_by_hour(dist, 24))
    print(len(dist))
    print(dist)
    