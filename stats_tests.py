import argparse
import inspect
import itertools
import msprime
import msprime._msprime as _msp
import numpy as np
import random
import scipy
import tskit
import math
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import sys
from tqdm import tqdm

matplotlib.use("Agg")
import statsmodels.api as sm

from yaca import sim
from yaca import sim_hudson as simh


class Test:
    def __init__(self, basedir, cl_name):
        self.set_output_dir(basedir, cl_name)

    def set_output_dir(self, basedir, cl_name):
        output_dir = pathlib.Path(basedir) / cl_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

    def _get_tests(self):
        return [
            value
            for name, value in inspect.getmembers(self)
            if name.startswith("test_")
        ]

    def _run_tests(self):
        all_results = self._get_tests()
        print(f"Collected {len(all_results)} test(s).")
        for method in all_results:
            method()

    def _build_filename(self, filename):
        return self.output_dir / (filename + ".png")

    def plot_qq(self, v1, v2, x_label, y_label, filename):
        sm.graphics.qqplot(v1)
        sm.qqplot_2samples(v1, v2, line="45")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def plot_histogram(self, x, x_label, filename):
        n, bins, patches = plt.hist(x, density=True, bins="auto")
        plt.xlabel(x_label)
        plt.ylabel("density")
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def get_seeds(self, num_replicates, seed=None):
        rng = np.random.default_rng(seed)
        max_seed = 2**16
        return rng.integers(1, max_seed, size=num_replicates)

    def run_yaca(self, rho, L, n, num_replicates, rejection, expectation):
        seeds = self.get_seeds(num_replicates)
        for seed in tqdm(seeds, desc="Running yaca"):
            yield sim.sim_yaca(
                n, rho, L, seed=seed, rejection=rejection, expectation=expectation
            )


class TestMargTBL(Test):
    """
    Test marginal total branch length
    """

    def no_test_tbl(self):
        rho = 7.5e-4
        L = 1000
        param_str = f"L_{L}_rho_{rho}_ownt"
        num_replicates = 500

        for n in [2, 4, 8]:
            # expecation based coalescence rates
            for rejection in True, False:
                self.verify_single_model(rho, L, n, num_replicates, rejection, param_str=param_str)
            # same tests for pairwise rates
            self.verify_single_model(
                rho, L, n, num_replicates, rejection=False, expectation=False, param_str=param_str
            )

    def test_tbl2(self):
        rho = 5e-4
        L = 1e5
        param_str = f"L_{L}_rho_{rho}_ownt"
        num_replicates = 500
        for n in [2, 4, 8]:
            # expecation based coalescence rates
            for rejection in True, False:
                self.verify_single_model(rho, L, n, num_replicates, rejection, param_str=param_str)
            # same tests for pairwise rates
            #self.verify_single_model(
            #    rho, L, n, num_replicates, rejection=False, expectation=False, param_str=param_str
            #)

    def verify_single_model(
        self, rho, L, n, num_replicates, rejection=True, expectation=True, param_str=""
    ):
        trt, tbl, t1, tn = self.get_marginal_tree_stats(
            rho, L, n, num_replicates, rejection, expectation
        )
        if expectation:
            model = "rejection" if rejection else "weighted"
        else:
            model = "pairwise"
        self.verify_marginal_tree_stats(trt, tbl, n, model, t1, tn, param_str)

    def no_test_run_specific_seeds(self):
        n = 8
        parameters = {
            "rho": 7.5e-4,
            "L": 1000,
        }
        num_replicates = 500

        seeds = [54918]
        for seed in seeds:
            ts = sim.sim_yaca(
                n,
                parameters["rho"],
                parameters["L"],
                seed=seed,
                rejection=True,
                verbose=True,
            )
            num_trees = ts.num_trees
            if num_trees > 50:
                print(seed)
            print("num_trees:", ts.num_trees)

    def no_test_smc(self):
        num_replicates = 500
        sample_size = 8
        ploidy = 1
        n = ploidy * sample_size
        recombination_rate = 1e-8
        rho = 2 * ploidy * recombination_rate * 1e4
        sequence_length = 100000
        obs = np.zeros(num_replicates, dtype=np.float64)

        for i, ts in tqdm(
            enumerate(
                msprime.sim_ancestry(
                    samples=sample_size,
                    ploidy=ploidy,
                    num_replicates=num_replicates,
                    recombination_rate=rho,
                    sequence_length=sequence_length,
                    model="SMC",
                )
            ),
            total=num_replicates,
        ):
            tree = ts.first()
            obs[i] = tree.time(tree.root)

        exp = self.sample_marginal_tree_depth(n, num_replicates)
        self.plot_qq(obs, exp, "smc", "analytical", "tree_depth_smc")

    def get_marginal_tree_stats(
        self, rho, L, sample_size, num_replicates, rejection, expectation
    ):
        ts_iter = self.run_yaca(
            rho, L, sample_size, num_replicates, rejection, expectation
        )
        # keeping track of total root time and total branch length
        trt = np.zeros(num_replicates, dtype=np.float64)
        tbl = np.zeros(num_replicates, dtype=np.float64)
        t1 = np.zeros(num_replicates, dtype=np.float64)
        tn = np.zeros(num_replicates, dtype=np.float64)

        check_count = 0
        for i, ts in enumerate(ts_iter):
            tree = ts.first()
            trt[i] = tree.time(tree.root)
            tbl[i] = tree.total_branch_length
            check_count += ts.num_trees
            tn[i] = tree.time(tree.root) - max(
                tree.time(u) for u in tree.children(tree.root)
            )
            t1[i] = min(tree.time(tree.parent(u)) for u in tree.samples())
        return trt, tbl, t1, tn

    def verify_marginal_tree_stats(self, trt, tbl, n, model, t1=None, tn=None, param_str=""):
        num_replicates = trt.size
        exp_trt = self.sample_marginal_tree_depth(n, num_replicates)
        self.require_output_dir(f"n_{n}")
        self.plot_qq(trt, exp_trt, "yaca", "sum_exp", f"n_{n}/tree_depth_{model}_{param_str}")
        if isinstance(t1, np.ndarray) and isinstance(tn, np.ndarray):
            exp_t1 = np.random.exponential(
                scale=1 / math.comb(n, 2), size=num_replicates
            )
            exp_tn = np.random.exponential(scale=1, size=num_replicates)
            self.plot_qq(t1, exp_t1, "yaca", "t1", f"n_{n}/t1_{model}_{param_str}")
            self.plot_qq(tn, exp_tn, "yaca", "tn", f"n_{n}/tn_{model}_{param_str}")

    def sample_marginal_tree_depth(self, n, num_replicates):
        result = np.zeros(num_replicates, dtype=np.float64)
        for i in range(n, 1, -1):
            rate = math.comb(i, 2)
            result += np.random.exponential(scale=1 / rate, size=num_replicates)
        return result

    def mean_marginal_tree_depth(self, n):
        return 2 * (1 - 1 / n)

    def get_analytical_tbl(self, n, t):
        """
        Returns the probabily density of the total branch length t with
        a sample of n lineages. Wakeley Page 78.
        """
        t1 = (n - 1) / 2
        t2 = math.exp(-t / 2)
        t3 = pow(1 - math.exp(-t / 2), n - 2)
        return t1 * t2 * t3


class TestRecombination(Test):
    """
    Test code that deals with recombination

    """

    def generate_intervals(self, L, seed=None):
        intervals = [0]
        while intervals[-1] < L:
            new = random.uniform(1, L / 10)
            intervals.append(new + intervals[-1])
        return intervals

    def no_test_pick_segment_function(self):
        """
        Generate intervals and split intervals randomly
        given recombination rate and verify whether the
        length of the resulting random segment follows
        the correct distribution
        """
        L = 1e5
        n = 2
        rho = 1e-4
        num_reps = 1000
        T = 1
        node_times = (0, 0)

        intervals = self.generate_intervals(L)
        intervals = [
            sim.AncestryInterval(left, right, 1)
            for left, right in zip(intervals[::2], intervals[1::2])
        ]
        total_length = sum(interval.span for interval in intervals)
        tract_lengths = np.zeros(num_reps, dtype=np.float64)
        seeds = self.get_seeds(num_reps)
        for i in tqdm(range(num_reps)):
            rng = np.random.default_rng(seed[i])
            tract_lengths[i] = sum(
                interval.span
                for interval in sim.pick_segment(intervals, rho, T, node_times, rng)
            )

        # compare against exponential
        rng = np.random.default_rng()
        exp = rng.exponential(1 / rho, num_reps)
        self.plot_qq(
            tract_lengths, exp, "obs", "exp", f"tract_lengths_pick_segments_qq_n{n}"
        )

    def no_test_pick_segment_function2(self):
        "node times of both lineages differ"
        L = 1e5
        n = 2
        rho = 5e-4
        num_reps = 1000
        T = 1.5
        node_times = (0.5, 1.0)
        total_tree_length = sum(T - t for t in node_times)

        intervals = self.generate_intervals(L)
        intervals = [
            sim.AncestryInterval(left, right, 1)
            for left, right in zip(intervals[::2], intervals[1::2])
        ]
        total_length = sum(interval.span for interval in intervals)
        tract_lengths = np.zeros(num_reps, dtype=np.float64)
        rng = np.random.default_rng()
        for i in tqdm(range(num_reps)):
            tract_lengths[i] = sum(
                interval.span
                for interval in sim.pick_segment(intervals, rho, T, node_times, rng)
            )

        # compare against exponential
        exp = rng.exponential(1 / (rho / 2 * total_tree_length), num_reps)
        self.plot_qq(
            tract_lengths,
            exp,
            "obs",
            "exp",
            f"tract_lengths_pick_segments_diff_node_times_qq_n{n}",
        )

    def test_marginal_tree_span_distribution(self):
        """
        Test span of first marginal tree.

        """
        rho = 5e-4
        L = 1e5
        n = 4
        expectation = True
        rejection = True
        num_reps = 500
        rng = np.random.default_rng()
        tree_span_exp = self.sample_tree_span(n, rho, rng, num_reps)
        tree_span_yaca, tree_span_yaca_exp, num_trees_yaca = self.sample_yaca_tree_stats(
            n, rho, L, num_reps, rejection, expectation
        )

        # self.plot_qq(
        #    tree_span_yaca, tree_span_yaca_exp, "yaca", "exp", "marginal_tree_span_yaca_yaca_exp"
        # )
        # self.plot_qq(
        #    tree_span_yaca_exp, tree_span_exp, "yaca", "exp", "marginal_tree_span_yaca_exp_exp"
        # )
        # self.plot_qq(
        #    tree_span_yaca, tree_span_exp, "yaca", "exp", "marginal_tree_span_yaca_exp"
        # )
        tree_span_hudson, num_trees_hudson = self.sample_hudson_tree_stats(
            n, rho, L, num_reps
        )
        # self.plot_qq(tree_span_exp, tree_span_hudson, 'exp', 'hudson', 'marginal_tree_span_exp_hudson')
        self.plot_qq(
            tree_span_yaca,
            tree_span_hudson,
            "yaca",
            "hudson",
            "marginal_tree_span_yaca_expb_hudson",
        )
        self.plot_qq(
            num_trees_yaca,
            num_trees_hudson,
            "yaca", "hudson",
            "num_trees_yaca_expb_hudson"
        )

    def no_test_msp_msp_exact(self):
        rho = 5e-4
        L = 1e5
        n = 4
        num_reps = 1000

        num_trees_msp = self.sample_msprime_tree_stats(n, rho, L, num_reps)

        _, num_trees_exact = self.sample_hudson_tree_stats(n, rho, L, num_reps)
        self.plot_qq(
            num_trees_msp, num_trees_exact, "smc", "hudson", "num_trees_hudson_smc"
        )

    def sample_total_branch_length(self, n, rng, num_replicates):
        result = np.zeros(num_replicates, dtype=np.float64)
        for i in range(n, 1, -1):
            rate = (i - 1) / 2
            result += rng.exponential(scale=1 / rate, size=num_replicates)
        return result

    def sample_tree_span(self, n, rho, rng, num_reps):
        tbl = self.sample_total_branch_length(n, rng, num_reps)
        rates = tbl * rho / 2
        return rng.exponential(scale=1 / rates)

    def sample_yaca_tree_stats(self, n, rho, L, num_reps, rejection, expectation):
        tss = self.run_yaca(rho, L, n, num_reps, rejection, expectation)
        results = np.zeros((3, num_reps), dtype=np.float64)
        for i, ts in enumerate(tss):
            results[0, i] = ts.first().span
            rate = ts.first().total_branch_length * rho / 2
            results[1, i] = np.random.exponential(1 / rate)
            results[1, i] = np.clip(results[1, i], 0, L)
            results[2, i] = ts.num_trees
        return results

    def sample_hudson_tree_stats(self, n, rho, L, num_reps):
        simulator = msprime.ancestry._parse_sim_ancestry(
            samples=n,
            sequence_length=L,
            recombination_rate=rho / 2,
            ploidy=1,
            discrete_genome=False,
            population_size=1,
        )
        results = np.zeros((2, num_reps), dtype=np.float64)
        for i in tqdm(range(num_reps)):
            simulator.run()
            breakpoints = simulator.breakpoints
            if len(breakpoints) > 0:
                results[0, i] = float(breakpoints[0])
            else:
                results[0, i] = float(L)
            results[1, i] = simulator.num_breakpoints + 1
            simulator.reset()
        return results

    def sample_msprime_tree_stats(self, n, rho, L, num_reps):
        tss = msprime.sim_ancestry(
            samples=n,
            sequence_length=L,
            recombination_rate=rho / 2,
            ploidy=1,
            discrete_genome=False,
            population_size=1,
            num_replicates=num_reps,
            model="smc",
        )
        results = np.zeros(num_reps, dtype=np.float64)
        for i, ts in tqdm(enumerate(tss), total=num_reps):
            results[i] = ts.num_trees
        return results


class TestRecAgainstMsp(Test):
    """
    Test number of trees against msprime

    """

    def test_num_trees(self):
        rho = 5e-4
        L = 1e5
        n = 4
        num_reps = 1000

        # num_trees_yaca = self.yaca_num_trees(n, rho, L, num_replicates, False)
        num_trees_msp_exact = self.msp_exact_num_trees(n, rho / 2, L, num_reps)
        num_trees_msp = self.msp_num_trees(n, rho, L, num_reps, ploidy=1)
        self.plot_qq(
            num_trees_msp,
            num_trees_msp_exact,
            "msp",
            "msp_exact",
            f"num_trees_msp_msp_exact_n{n}",
        )
        # self.plot_qq(num_trees_yaca, num_trees_msp, "yaca", "msp", f"num_trees_n{n}")

    def no_test_num_trees_yaca(self):
        n = 2
        L = 5e4
        rho = 1e-4
        num_replicates = 500

        num_trees_yaca = self.yaca_num_trees(n, rho, L, num_replicates, False)
        for model in "hudson", "smc":
            num_trees_msp = self.msp_num_trees(n, rho, L, num_replicates, 1, model)
            self.plot_qq(
                num_trees_yaca, num_trees_msp, "yaca", "msp", f"num_trees_n{n}_{model}"
            )

    def yaca_num_trees(self, n, rho, L, num_replicates, rejection):
        tss = self.run_yaca(rho, L, n, num_replicates, rejection)
        result = np.zeros(num_replicates, dtype=np.int64)
        for i, ts in enumerate(tss):
            result[i] = ts.num_trees
        return result

    def msp_exact_num_trees(self, n, rho, L, num_replicates):
        # IMPORTANT!! We have to use the get_num_breakpoints method
        # on the simulator as there is a significant drop in the number
        # of trees if we use the tree sequence. There is a significant
        # number of common ancestor events that result in a recombination
        # being undone.
        num_breakpoints = np.zeros(num_replicates, dtype=np.int64)
        # ploidy is 2, see msprime.ancestry l374
        exact_sim = msprime.ancestry._parse_simulate(
            sample_size=n, recombination_rate=rho, length=L
        )
        for k in tqdm(range(num_replicates), desc="Running msprime exact"):
            exact_sim.run()
            num_breakpoints[k] = exact_sim.num_breakpoints
            exact_sim.reset()
        return num_breakpoints + 1

    def msp_num_trees(self, n, rho, L, num_replicates, ploidy=1, model="hudson"):
        num_trees = np.zeros(num_replicates, dtype=np.int64)
        for i, ts in tqdm(
            enumerate(
                msprime.sim_ancestry(
                    samples=n,
                    ploidy=ploidy,
                    num_replicates=num_replicates,
                    recombination_rate=rho,
                    sequence_length=L,
                    model=model,
                )
            ),
            total=num_replicates,
            desc="running msprime sim_ancestry",
        ):
            num_trees[i] = ts.num_trees

        return num_trees


class TestAgainstSmc(Test):
    def test_n8(self):
        num_replicates = 500
        sample_size = 8
        ploidy = 1
        n = ploidy * sample_size
        recombination_rate = 1e-4
        rho = 2 * ploidy * recombination_rate
        sequence_length = 100000
        obs = np.zeros(num_replicates, dtype=np.float64)

        for i, ts in tqdm(
            enumerate(
                msprime.sim_ancestry(
                    samples=sample_size,
                    ploidy=ploidy,
                    num_replicates=num_replicates,
                    recombination_rate=rho,
                    sequence_length=sequence_length,
                    model="SMC",
                )
            ),
            total=num_replicates,
        ):
            tree = ts.first()
            obs[i] = tree.time(tree.root)

        exp = self.sample_marginal_tree_depth(n, num_replicates)
        self.plot_qq(obs, exp, "smc", "analytical", "tree_depth_smc")


class TestVisualize(Test):
    def test_show_msp(self):
        self.show_trees(msp=True)

    def show_trees(self, msp=False):
        # | less -S
        n = 2
        rho = 1e-4
        L = 5e4
        Ne = 1e4
        r = rho / (4 * Ne)

        reps = 5
        seeds = self.get_seeds(reps)
        if not msp:
            print("yaca trees")
            for seed in seeds:
                ts = sim.sim_yaca(n, rho, L, seed=seed)
                print(ts.draw_text())

        else:
            print("msp generated trees")
            for _ in range(reps):
                ts_msp = msprime.sim_ancestry(
                    samples=2,
                    ploidy=1,
                    sequence_length=L,
                    recombination_rate=rho,
                    model="SMC",
                )
                print(ts_msp.draw_text())


class TestWaitingTimesSimple(Test):
    """
    Test distribution of waiting times against msp
    """

    def test_time_to_first_event_n2(self):
        sim_reps = 1000
        parameters = {
            "sequence_length": 1_000_000,
            "recombination_rate": 1e-8,
            "population_size": 1e4,
            "samples": 1,
            "ploidy": 2,
        }
        exp = self.get_first_coalescence_time(sim_reps, parameters)
        obs = self.get_times_non_hom_poiss(parameters)
        self.plot_qq(obs, exp, "yaca", "msp", "test_time_to_first_event_n2")

    def test_time_to_first_event_n4(self):
        parameters = {
            "sequence_length": 1_000_000,
            "recombination_rate": 1e-8,
            "population_size": 1e4,
            "samples": 2,
            "ploidy": 2,
        }
        exp = self.get_first_coalescence_time(1000, parameters)
        obs = self.get_times_non_hom_poiss(parameters)
        self.plot_qq(obs, exp, "yaca", "msp", "test_time_to_first_event_n4")

    def test_time_to_first_event_n8(self):
        parameters = {
            "sequence_length": 1_000_000,
            "recombination_rate": 1e-8,
            "population_size": 1e4,
            "samples": 4,
            "ploidy": 2,
        }
        exp = self.get_first_coalescence_time(1000, parameters)
        obs = self.get_times_non_hom_poiss(parameters)
        self.plot_qq(obs, exp, "yaca", "msp", "test_time_to_first_event_n8")

    def get_first_coalescence_time(self, reps, parameters):
        """
        Simulate reps and extract the time to the first coalescence event.
        """
        num_samples = parameters["ploidy"] * parameters["samples"]
        results = np.zeros(reps, dtype=np.float64)
        sim_reps = msprime.sim_ancestry(**parameters, num_replicates=reps)
        for i, ts in tqdm(enumerate(sim_reps), total=reps):
            results[i] = ts.node(num_samples).time
        results /= 2 * parameters["population_size"]
        return results

    def get_times_non_hom_poiss(self, parameters):
        num_lineages = parameters["samples"] * parameters["ploidy"]
        rho = (
            4
            * parameters["population_size"]
            * parameters["recombination_rate"]
            * parameters["sequence_length"]
        )
        total_overlap = math.comb(num_lineages, 2) * rho

        # return self.draw_value_non_hom(10_000, total_overlap, num_lineages)
        return self.draw_min_non_hom(10_000, rho, num_lineages)

    def draw_value_non_hom(self, reps, rho_overlap, num_lineages):
        """
        Given expected coalescence rate (k choose 2)* (2*T*rho), draw  reps random values
        from the non-homogeneous exponential distribution.
        rho is the total overlap of all combinations expressed in recombination rate units.
        length as recombination rate = length * rho
        rho should be total amount of overlap across all combinations
        """
        rng = random.Random()
        results = np.zeros(reps, dtype=np.float64)
        for i in range(reps):
            results[i] = sim.draw_event_time(num_lineages, rho_overlap, rng)
        return results

    def draw_min_non_hom(self, reps, rho_overlap, num_lineages):
        rng = random.Random()
        results = np.zeros(reps, dtype=np.float64)
        for i in range(reps):
            time = np.inf
            for _ in range(math.comb(num_lineages, 2)):
                temp = sim.draw_event_time(2, rho_overlap, rng)
                time = min(time, temp)
            results[i] = time

        return results


class TestWaitingTimes(Test):
    """
    Test distribution of waiting times against modified hudson algorithm
    """

    def test_waiting_times(self):
        n = [12, 20, 50]
        fixed_time = [0.0, None]
        # what kind of prior to expect here
        # on lineage node times
        time_distribution = None

        for combo in itertools.product(fixed_time, n):
            self.compare_waiting_time_distributions(*combo)

    def compare_waiting_time_distributions(self, fixed_time, n):
        rng = random.Random()
        rho = 1e-4
        sequence_length = 10_000
        expectation = True
        num_reps = 1000

        # generate_segments
        start_lineages = [
            self.generate_lineage(sequence_length, i, rng, fixed_time) for i in range(n)
        ]
        time_last_event = max(lineage.node_time for lineage in start_lineages)

        # draw bunch of waiting times
        waiting_times_hudson = np.zeros(num_reps, dtype=np.float64)
        for i in tqdm(range(num_reps), desc="running hudson ..."):
            lineages = start_lineages[:]
            t, _, _ = simh.time_to_next_coalescent_event(lineages, rho, time_last_event)
            waiting_times_hudson[i] = t

        waiting_times_yaca = np.zeros_like(waiting_times_hudson)

        mean_time_to_last_event = (
            sum(time_last_event - lin.node_time for lin in lineages) / n
        )
        total_overlap = sim.update_total_overlap_brute_force(lineages)
        re_rate = total_overlap * rho

        for i in tqdm(range(num_reps), desc="running yaca ..."):
            waiting_times_yaca[i] = self.sample_waiting_time_yaca_expectation(
                n, re_rate, rng, mean_time_to_last_event
            )
        waiting_times_yaca += time_last_event

        plot_str = "start" if fixed_time != None else "mid_run"
        self.plot_qq(
            waiting_times_yaca,
            waiting_times_hudson,
            "yaca",
            "hudson",
            f"time_to_next_coalescence_{plot_str}_n{n}_qq",
        )

    def sample_waiting_time_yaca_expectation(
        self, n, re_rate, rng, mean_time_to_last_event
    ):
        return sim.draw_event_time(n, re_rate, rng, mean_time_to_last_event)

    def sample_waiting_time_yaca_pairwise(self, rho, lineages, rng, T):
        _, _, _, _, t = sim.sample_pairwise_times(lineages, rng, T, rho)
        return t + T

    def generate_ancestry(self, L, rng):
        intervals = [0]
        while intervals[-1] < L:
            new = rng.uniform(1, L / 10)
            intervals.append(new + intervals[-1])
        return [
            sim.AncestryInterval(left, right, 1)
            for left, right in zip(intervals[::2], intervals[1::2])
        ]

    def generate_lineage(self, L, i, rng, fixed_time=None):
        ancestry = self.generate_ancestry(L, rng)
        # how much variance do we want to add to the node times?
        if fixed_time != None:
            node_time = fixed_time
        else:
            node_time = rng.random()
        return sim.Lineage(i, ancestry, node_time)


class TestFoo(Test):
    def no_test_foo(self):
        n = 2
        rho = 1e-4
        L = 1e4
        num_replicates = 10
        rejection = True
        max_trees = 0
        for i, (ts, seed) in enumerate(
            self.run_yaca(rho, L, n, num_replicates, rejection)
        ):
            print(seed, ts.num_trees)
            max_trees = max(ts.num_trees, max_trees)
        print("yaca_max_trees:", max_trees)

    def test_num_breakpoints(self):
        n = 2
        rho = 1e-4
        L = 5e4
        Ne = 1e4
        r = rho / (4 * Ne)

        print("4Ner * L:", rho * L)
        reps = 5
        num_trees = np.zeros(reps, dtype=np.int64)
        for i in tqdm(range(reps), "running yaca"):
            seed = random.randint(1, 2**16)
            ts = sim.sim_yaca(n, rho, L, seed=seed)
            ts_msp = msprime.sim_ancestry(
                samples=2,
                ploidy=1,
                sequence_length=L,
                recombination_rate=rho,
                model="SMC",
            )
            # print(ts.draw_text())
            print(ts_msp.draw_text())
            num_trees[i] = ts.num_trees
        num_breakpoints = num_trees - 1
        self.plot_histogram(num_trees, "num_trees", f"histogram_n{n}")

        num_trees_msp = self.msp_num_breakpoints(n, r, Ne, L, reps)
        self.plot_qq(num_trees, num_trees_msp, "yaca", "msp", f"qq_num_trees_n{n}")
        self.verify_breakpoint_distribution(num_breakpoints, rho, L)

    def no_test_run_with_seed(self):
        # expected number of breakpoints: 4*Ne*r*L = 5
        n = 2
        rho = 1e-4
        L = 5e4

        seed = 44883
        ts = sim.sim_yaca(n, rho, L, seed=seed)
        print(ts.num_trees)
        print(list(ts.breakpoints()))

    def msp_num_breakpoints(self, n, r, Ne, L, num_replicates):
        # IMPORTANT!! We have to use the get_num_breakpoints method
        # on the simulator as there is a significant drop in the number
        # of trees if we use the tree sequence. There is a significant
        # number of common ancestor events that result in a recombination
        # being undone.
        num_trees = np.zeros(num_replicates, dtype=np.int64)
        # ploidy is 2, see msprime.ancestry l374
        exact_sim = msprime.ancestry._parse_simulate(
            sample_size=n, recombination_rate=r, Ne=Ne, length=L
        )
        for k in tqdm(range(num_replicates), desc="Running msprime"):
            exact_sim.run()
            num_trees[k] = exact_sim.num_breakpoints
            exact_sim.reset()
        return num_trees

    def verify_breakpoint_distribution(self, x, rho, L):
        scipy.stats.probplot(x, dist=scipy.stats.poisson(rho * L), plot=plt)
        path = self.output_dir / f"num_breakpoints_analytical_qq.png"
        plt.savefig(path)
        plt.close("all")

    def no_test_foo_3(self):
        a = [
            (0, 2238.0, 1),
            (37933.0, 37949.0, 1),
            (40696.0, 40882.0, 1),
            (49653.0, 50000.0, 1),
        ]
        a = [
            sim.AncestryInterval(left, right, ancestral_to)
            for left, right, ancestral_to in a
        ]
        b = [
            (0, 2238.0, 1),
            (37933.0, 37949.0, 1),
            (40696.0, 40882.0, 1),
            (49653.0, 50000.0, 1),
        ]
        a = [
            sim.AncestryInterval(left, right, ancestral_to)
            for left, right, ancestral_to in b
        ]
        n = 2
        rho = 1e-4
        L = 5e4
        total_overlap = 2787.0
        for _ in range(10):
            seed = random.randint(0, 2**16)
            left, right = sim.pick_breakpoints(a, total_overlap, rho, 2, (0, 0), seed)
            print(left, right)

class TestSingleStep(Test):

    def test_single(self):
        n = 16
        rho = 1e-3
        L = 1e5
        num_replicates = 500
        run_until = 2.5
        param_str = f"n{n}_rho{rho}_L{L}"

        coal_time_msp = np.zeros(num_replicates, dtype=np.float64)
        coal_time_yaca = np.zeros_like(coal_time_msp)
        coal_time_test = np.zeros_like(coal_time_msp)
        seeds = self.get_seeds(num_replicates)
        ts, lineages = self.generate_lineages(n, rho, L, run_until)
        #print(ts.tables.nodes.time)
        for i in tqdm(range(num_replicates)):
            coal_time_msp[i] = self.run_msp_single_step(rho, ts, run_until)
            coal_time_yaca[i] = self.draw_waiting_time_yaca(rho, lineages, seeds[i])
                       
        print(np.min(coal_time_msp), np.min(coal_time_yaca))
        print(np.min(coal_time_msp) - np.min(coal_time_yaca))
        self.plot_qq(coal_time_yaca, coal_time_msp, 'yaca', 'msp', f"simpl_single_step_{param_str}")
        # self.plot_qq(coal_time_test, coal_time_msp, 'test', 'msp', f"test_single_step_{param_str}")

    def ts_to_lineages(self, ts):
        lineages = dict()
        ts = ts.simplify()
        
        for tree in ts.trees():
            for root in tree.roots:
                ancestral_to = tree.num_samples(root) 
                left, right = tree.interval.left, tree.interval.right
                new_ancestry_interval = sim.AncestryInterval(left, right, ancestral_to)
                if root not in lineages:
                    lineages[root] = sim.Lineage(root, [new_ancestry_interval], tree.time(root))
                else:
                    if left == lineages[root].ancestry[-1].right:
                        lineages[root].ancestry[-1].right = right            
                    else:
                        lineages[root].ancestry.append(new_ancestry_interval)
        
        return list(lineages.values())

    def generate_lineages(self, n, rho, L, run_until):
        ret = False
        
        while not ret:
            ts = msprime.sim_ancestry(
                samples=n,
                sequence_length=L,
                recombination_rate=rho / 2,
                ploidy=1,
                discrete_genome=False,
                population_size=1,
                end_time=run_until
            )
            ret = max(tree.num_roots for tree in ts.trees())>1
        lineages = self.ts_to_lineages(ts)

        return ts, lineages

    def run_msp_single_step(self, rho, ts, sim_start_time, time_step=0.5):
        simulator = msprime.ancestry._parse_sim_ancestry(
            recombination_rate=rho / 2,
            ploidy=1,
            discrete_genome=False,
            population_size=1,
            initial_state=ts
            )
        old_time = simulator.time
        # extract time to next coalescence
        new_time = old_time
        num_nodes = simulator.num_nodes
        while simulator.num_nodes == num_nodes:
            new_time += time_step
            simulator._run_until(new_time)

        tables = tskit.TableCollection.fromdict(simulator.tables.asdict())
        tables.simplify()
        times = tables.nodes.time
        last_coal_idx = np.sum(times<=old_time)
        return times[last_coal_idx] - times[last_coal_idx - 1]
        #return times[last_coal_idx] - old_time

    def draw_waiting_time_yaca(self, rho, lineages, seed):
        last_event = max(lineage.node_time for lineage in lineages)
        total, overlap_weighted_node_times, pairs_count = sim.update_total_overlap_brute_force(lineages, last_event)
        
        rng = random.Random(seed)
        total_overlap_rho = rho * total
        new_time = sim.draw_event_time(pairs_count, total_overlap_rho, rng, overlap_weighted_node_times)
        return new_time

def run_tests(suite, output_dir):
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "TestMargTBL",
        "TestRecombination",
        "TestRecAgainstMsp",
        "TestWaitingTimesSimple",
        "TestWaitingTimes",
        "TestVisualize",
        "TestAgainstSmc",
        "TestFoo",
        "TestSingleStep"    ]

    parser.add_argument(
        "--test-class",
        "-c",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all tests for specified test class",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="stats_tests_output",
        help="specify the base output directory",
    )

    args = parser.parse_args()

    run_tests(args.test_class, args.output_dir)


if __name__ == "__main__":
    main()
