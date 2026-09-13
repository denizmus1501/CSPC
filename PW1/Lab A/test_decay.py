"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


# TODO 1: test_rejects_negative_rate
#   Check that calling simulate(...) with a negative lam raises a ValueError.
#   Which pytest tool checks that an error is raised? - pytest.raises()

def test_rejects_negative_rate():
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


# TODO 2: test_matches_law
#   Check that the simulation's AVERAGE over many seeds is close to the
#   physical law  N0 * exp(-lam * t).
#   Which pytest tool compares floating-point values with a tolerance? - pytest.approx()

def test_matches_law():
    N0 = 1000
    lam = 0.4
    steps = 200
    dt = 0.05
    n_runs = 100

    simulations = []

    for seed in range(n_runs):
        result = simulate(N0, lam, dt=dt, steps=steps, seed=seed)
        simulations.append(result)

    average = np.mean(simulations, axis=0)

    t = np.arange(steps+1) * dt
    expected = N0 * np.exp(-lam * t)

    assert average[-1] == pytest.approx(expected[-1], rel = 0.05)