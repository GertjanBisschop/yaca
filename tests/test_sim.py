import itertools
import pytest
import random
import numpy as np
import msprime

import yaca.sim as sim
import yaca.sim_hudson as simh


class TestInverseExpectationFunction:
    def f(self, x, k, rho, T):
        """
        Integral of c + 2*(T+t)*rho over 0 to x
        T is mean time of nodes to last event.
        """
        c = k * (k - 1) / 2
        return c * x + x**2 * rho + 2 * x * T * rho

    def test_inverse(self):
        k = 4
        rho = 0.1
        T = 0.25
        for _ in range(10):
            x = random.random()
            f_x = self.f(x, k, rho, T)
            assert np.isclose(
                sim.inverse_expectation_function_extended(f_x, rho, k, T), x
            )

@pytest.mark.overlap
class TestIntersect:
    def test_intersect_segment(self):
        a = sim.Lineage(
            0,
            [
                sim.AncestryInterval(1, 20, 0),
                sim.AncestryInterval(25, 29, 0),
                sim.AncestryInterval(32, 40, 0),
            ],
        )
        b = sim.Lineage(
            1,
            [
                sim.AncestryInterval(0, 15, 0),
                sim.AncestryInterval(17, 30, 0),
                sim.AncestryInterval(32, 40, 0),
            ],
        )
        exp = [
            sim.AncestryInterval(1, 15, 0),
            sim.AncestryInterval(17, 20, 0),
            sim.AncestryInterval(25, 29, 0),
            sim.AncestryInterval(32, 40, 0),
        ]
        overlap, overlap_length = list(sim.intersect_lineages(a, b))
        assert overlap == exp
        assert overlap_length == sum(el.span for el in exp)
        # swap a and b
        overlap, overlap_length = list(sim.intersect_lineages(b, a))
        assert overlap == exp
        assert overlap_length == sum(el.span for el in exp)

    def test_intersect_segment2(self):
        edge = random.random()
        a = sim.Lineage(0, [sim.AncestryInterval(0.0, edge, 0)])
        b = sim.Lineage(0, [sim.AncestryInterval(edge, 2.0, 0)])
        overlap, overlap_length = list(sim.intersect_lineages(a, b))
        assert overlap_length == 0
        assert len(overlap) == 0

    def test_intersect_segment3(self):
        edge = 132.65365493568527
        edge2 = 774.2345
        a = sim.Lineage(
            0,
            [
                sim.AncestryInterval(0.0, edge, 0),
                sim.AncestryInterval(edge2, 1000.0, 0),
            ],
        )
        b = sim.Lineage(0, [sim.AncestryInterval(edge, 1000.0, 0)])
        overlap, overlap_length = list(sim.intersect_lineages(a, b))
        assert len(overlap) == 1
        overlap_length == 1000 - edge2

    def test_intersect_segment4(self):
        edges = [random.random() for _ in range(10)]
        a = sim.Lineage(0, [
            sim.AncestryInterval(1.0, edges[0], 0),
            sim.AncestryInterval(edges[1], edges[2], 0),
            sim.AncestryInterval(edges[4], edges[6], 0),
                ]   
            )
        b = sim.Lineage(0, [
            sim.AncestryInterval(edges[0], edges[1], 0),
            sim.AncestryInterval(edges[1], edges[2], 0),
            sim.AncestryInterval(edges[3], edges[5], 0),
                ]   
            )
        overlap, overlap_length = sim.intersect_lineages(a, b)
        exp_length = (edges[2] - edges[1]) + (edges[5] - edges[4])


    def test_remove_segment(self):
        lineage = sim.Lineage(
            0,
            [
                sim.AncestryInterval(10, 20, 3),
                sim.AncestryInterval(23, 30, 2),
                sim.AncestryInterval(35, 50, 1),
            ],
            0,
        )
        to_remove = [
            sim.AncestryInterval(11, 15, 4),
            sim.AncestryInterval(17, 20, 3),
            sim.AncestryInterval(23, 30, 1),
        ]
        expected = [
            sim.AncestryInterval(10, 11, 3),
            sim.AncestryInterval(15, 17, 3),
            sim.AncestryInterval(35, 50, 1),
        ]
        test_result = list(sim.remove_segment(lineage, to_remove))
        assert len(test_result) == len(expected)
        for test, exp in zip(test_result, expected):
            assert test == exp

    def test_remove_segment2(self):
        lineage = sim.Lineage(
            0,
            [
                sim.AncestryInterval(10, 20, 3),
                sim.AncestryInterval(30, 40, 2),
                sim.AncestryInterval(50, 60, 1),
            ],
            0,
        )
        to_remove = [
            sim.AncestryInterval(11, 15, 4),
            sim.AncestryInterval(55, 58, 3),
        ]
        expected = [
            sim.AncestryInterval(10, 11, 3),
            sim.AncestryInterval(15, 20, 3),
            sim.AncestryInterval(30, 40, 2),
            sim.AncestryInterval(50, 55, 1),
            sim.AncestryInterval(58, 60, 1),
        ]
        test_result = list(sim.remove_segment(lineage, to_remove))
        assert len(test_result) == len(expected)
        for test, exp in zip(test_result, expected):
            assert test == exp

    def test_remove_segment_all(self):
        lineage = sim.Lineage(
            0,
            [
                sim.AncestryInterval(10, 20, 3),
                sim.AncestryInterval(23, 30, 2),
                sim.AncestryInterval(35, 50, 1),
            ],
            0,
        )

        test_result = list(sim.remove_segment(lineage, lineage.ancestry))
        assert len(test_result) == 0

    def test_remove_segment_simple(self):
        lineage = sim.Lineage(
            0,
            [
                sim.AncestryInterval(0, 100, 1),
            ],
            0,
        )
        to_remove = [
            sim.AncestryInterval(18, 62, 2),
        ]
        expected = [
            sim.AncestryInterval(0, 18, 1),
            sim.AncestryInterval(62, 100, 1),
        ]
        test_result = list(sim.remove_segment(lineage, to_remove))
        assert len(test_result) == len(expected)
        for interval, exp in zip(test_result, expected):
            assert interval == exp

    def test_combinadic_map(self):
        n = 10
        for pair in itertools.combinations(range(n), 2):
            assert (
                tuple(list(sim.reverse_combinadic_map(sim.combinadic_map(pair)))[::-1])
                == pair
            )


class TestSimulate:
    @pytest.mark.full_sim
    @pytest.mark.timeout(2)
    def test_basic_coalescent_no_rec(self):
        n = 4
        sequence_length = 5
        ts = sim.sim_yaca(n, L=sequence_length, rho=0.0, seed=1)
        assert ts.num_samples == n
        assert ts.sequence_length == sequence_length
        assert ts.num_trees == 1
        assert all(tree.num_roots == 1 for tree in ts.trees())
        assert max(tree.depth(u) for tree in ts.trees() for u in ts.samples()) == n - 1

    @pytest.mark.full_sim
    @pytest.mark.timeout(2)
    def test_basic_coalescent_rec(self):
        seed = 3
        n = 4
        ts = sim.sim_yaca(n, L=100, rho=1, seed=seed)
        assert ts.num_samples == n
        assert ts.sequence_length == 100
        assert ts.num_trees > 1
        assert all(tree.num_roots == 1 for tree in ts.trees())
        assert max(tree.depth(u) for tree in ts.trees() for u in ts.samples()) == n - 1

    @pytest.mark.full_sim
    @pytest.mark.timeout(2)
    def test_basic_coalescent_rec_pairwise_rates(self):
        seed = 3
        n = 4
        ts = sim.sim_yaca(n, L=100, rho=1, seed=seed, rejection=False)
        assert ts.num_samples == n
        assert ts.sequence_length == 100
        assert ts.num_trees > 1
        assert all(tree.num_roots == 1 for tree in ts.trees())
        assert max(tree.depth(u) for tree in ts.trees() for u in ts.samples()) == n - 1


class TestAux:
    def test_merge_intervals(self):
        lineage1 = sim.Lineage(
            0,
            [
                sim.AncestryInterval(10, 20, 3),
                sim.AncestryInterval(23, 30, 2),
                sim.AncestryInterval(35, 50, 1),
            ],
            0,
        )
        lineage2 = sim.Lineage(
            0,
            [
                sim.AncestryInterval(0, 10, 3),
                sim.AncestryInterval(20, 23, 3),
                sim.AncestryInterval(30, 35, 2),
            ],
            0,
        )
        result = sim.merge_lineages_test((lineage1, lineage2))


@pytest.mark.hudson
class TestSimHudson:
    @pytest.mark.full_sim
    def test_time_to_next_event(self):
        for _ in range(100):
            rng = random.Random()
            n = 4
            rho = 1e-4
            sequence_length = 10_000
            lineages = [
                self.generate_lineage(sequence_length, i, rng) for i in range(n)
            ]

            t, _, _ = simh.time_to_next_coalescent_event(lineages, rho, 0)
            assert t > 0
            assert False

    def generate_ancestry(self, L, rng):
        intervals = [0]
        while intervals[-1] < L:
            new = rng.uniform(1, L / 10)
            intervals.append(new + intervals[-1])
        return [
            sim.AncestryInterval(left, right, 1)
            for left, right in zip(intervals[::2], intervals[1::2])
        ]

    def generate_lineage(self, L, i, rng):
        ancestry = self.generate_ancestry(L, rng)
        node_time = rng.random()
        return sim.Lineage(i, ancestry, node_time)

class TestExtractLineages:
    def test_extraction(self):
        n = 16
        rho = 1e-3
        L = 1e5
        num_replicates = 500
        run_until = 2.5
        param_str = f"n{n}_rho{rho}_L{L}"

        coal_time_msp = np.zeros(num_replicates, dtype=np.float64)
        coal_time_yaca = np.zeros_like(coal_time_msp)
        
        _, lineages, lineages_all = self.generate_lineages(n, rho, L, run_until)
        total, ownt, pairs_count = sim.update_total_overlap_brute_force(lineages, run_until)
        total_all, ownt_all, pairs_count_all = sim.update_total_overlap_brute_force(lineages_all, run_until)
        
        assert total == total_all
        assert pairs_count == pairs_count_all
        assert ownt == ownt_all

    def ts_to_lineages(self, ts, include_all=False):
        lineages = dict()
        ts = ts.simplify() # remove unary nodes
        num_samples = ts.num_samples
        
        for tree in ts.trees():
            for root in tree.roots:
                ancestral_to = tree.num_samples(root) 
                if ancestral_to < num_samples or include_all:
                    left, right = tree.interval.left, tree.interval.right
                    new_ancestry_interval = sim.AncestryInterval(left, right, ancestral_to)
                    if root not in lineages:
                        lineages[root] = sim.Lineage(root, [new_ancestry_interval], tree.time(root))
                    else:
                        prev = lineages[root].ancestry[-1]
                        if left == prev.right and ancestral_to == prev.ancestral_to:
                            prev.right = right            
                        else:
                           lineages[root].ancestry.append(new_ancestry_interval)
        


        return list(lineages.values())

    def generate_lineages(self, n, rho, L, run_until):
        ret = False

        while not ret:
            seed = random.randint(1,2**16)
            ts = msprime.sim_ancestry(
                samples=n,
                sequence_length=L,
                recombination_rate=rho / 2,
                ploidy=1,
                discrete_genome=False,
                population_size=1,
                end_time=run_until,
                random_seed=seed
            )
            ret = max(tree.num_roots for tree in ts.trees())>1
        
        lineages = self.ts_to_lineages(ts)
        lineages_all = self.ts_to_lineages(ts, True)

        return ts, lineages, lineages_all