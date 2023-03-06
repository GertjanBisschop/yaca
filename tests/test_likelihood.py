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
        seeds = np.random.randint(1, 2**16, 10)
        rho = 1e-1
        for seed in seeds:
            msts = msprime.sim_ancestry(
                4,
                ploidy=1,
                sequence_length=500,
                recombination_rate=rho / 2,
                random_seed=seed,
                record_unary=True,
            )
            assert lik.log_ts_likelihood(msts, rho) < 0


class TestNoRec:
    def test_no_rec(self):
        sequence_length = 1
        rho = 0
        n = 4
        seeds = [10, 100, 11000, 1110010]
        for seed in seeds:
            mts = msprime.sim_ancestry(
                n,
                population_size=0.5,
                ploidy=2,
                recombination_rate=rho / 2,
                sequence_length=sequence_length,
                record_full_arg=True,
                discrete_genome=False,
                random_seed=seed,
            )
            ts = mts.simplify()
            # in yaca things are formulated for haploids
            # adapt!
            obs = lik.log_ts_likelihood(ts, rho)
            exp = msprime.log_arg_likelihood(mts, rho / 2, 0.5)
            assert np.isclose(obs, exp)
