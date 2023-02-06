import itertools
import pytest
import random
import numpy as np
import msprime

import yaca.sim as sim
import yaca.likelihood as lik


class TestYacaSim:
    def test_simple_sim(self):
        seeds = np.random.randint(1, 2**16, 10)
        rho = 0.25
        for seed in seeds:
            ts = sim.sim_yaca(6, rho, 100, seed=seed)
            assert lik.log_ts_likelihood(ts, rho) < 0


class TestMsprimeSim:
    def test_msprime_sim(self):
        rho = 0.2
        msts = msprime.sim_ancestry(
            4, sequence_length=10, recombination_rate=rho / 2, random_seed=11
        )
        assert lik.log_ts_likelihood(msts, rho) < 0
