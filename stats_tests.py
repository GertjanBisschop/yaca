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
from collections.abc import Iterable

matplotlib.use("Agg")
import statsmodels.api as sm

from yaca import sim


class Test:
    def __init__(self, basedir, cl_name):
        self.basedir = basedir
        self.set_output_dir(basedir, cl_name)

    def set_output_dir(self, basedir, cl_name):
        output_dir = pathlib.Path(self.basedir) / cl_name
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
        print(f"[+] Collected {len(all_results)} subtest(s).")
        for method in all_results:
            method()

    def _build_filename(self, filename, extension=".png"):
        return self.output_dir / (filename + extension)

    def plot_qq(self, v1, v2, x_label, y_label, filename, info=""):
        sm.graphics.qqplot(v1)
        sm.qqplot_2samples(v1, v2, line="45")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(info)
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

    def run_yaca(self, n, rho, L, seeds):
        for seed in tqdm(seeds, desc="Running yaca"):
            yield sim.sim_yaca(n, rho, L, seed=seed)

    def run_msprime(self, n, rho, L, seeds):
        for seed in tqdm(seeds, desc="Running msp"):
            yield msprime.sim_ancestry(
                samples=n,
                recombination_rate=rho / 2,
                sequence_length=L,
                ploidy=1,
                population_size=1,
                discrete_genome=False,
                random_seed=seed,
            )

    def msp_exact_num_trees(self, n, rho, L, num_replicates, seed):
        # IMPORTANT!! We have to use the get_num_breakpoints method
        # on the simulator as there is a significant drop in the number
        # of trees if we use the tree sequence. There is a significant
        # number of common ancestor events that result in a recombination
        # being undone.
        num_breakpoints = np.zeros(num_replicates, dtype=np.int64)
        exact_sim = msprime.ancestry._parse_sim_ancestry(
            samples=n,
            recombination_rate=rho / 2,
            sequence_length=L,
            ploidy=1,
            random_seed=seed,
        )
        for k in tqdm(range(num_replicates), desc="Running msprime exact"):
            exact_sim.run()
            num_breakpoints[k] = exact_sim.num_breakpoints
            exact_sim.reset()
        return num_breakpoints + 1

    def sum_squared_residuals(self, a1, a2):
        a1 = np.sort(a1)
        a2 = np.sort(a2)
        return np.sum((a1 - a2) ** 2)

    def scatter(self, overlap, ssr_array, x_label, y_label, filename):
        plt.scatter(overlap, ssr_array)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        f = self._build_filename(filename)
        plt.savefig(f, dpi=72)
        plt.close("all")

    def log_run(self, filename, info_labels, info_array):
        with open(filename, "w") as logfile:
            print(
                "\t".join(l for l in info_labels),
                file=logfile,
            )
            for line in info_array:
                print(
                    "\t".join(str(entry) for entry in line),
                    file=logfile,
                )

    def marginal_tree_stats(self, ts, positions, dims):
        result = np.zeros(dims, dtype=np.float64)
        # t1, tn, tree_depth, total_branch_length
        for i, pos in enumerate(positions):
            tree = ts.at(pos)
            root_time = tree.time(tree.root)
            result[i, 0] = min(tree.time(tree.parent(u)) for u in tree.samples())
            result[i, 1] = root_time - max(
                tree.time(u) for u in tree.children(tree.root)
            )
            result[i, 2] = root_time
            result[i, 3] = tree.total_branch_length

        return result

    def get_expected_tree_stats(self, n, rng, dims):
        # t1, tn, tree_depth, total_branch_length
        results = np.zeros(dims, dtype=np.float64)
        num_replicates = dims[-1]

        results[0] = rng.exponential(scale=1 / math.comb(n, 2), size=num_replicates)
        results[1] = rng.exponential(scale=1, size=num_replicates)
        results[2:] = self.sample_marginal_tree_depth(n, rng, num_replicates)
        return results

    def sample_marginal_tree_depth(self, n, rng, num_replicates):
        result = np.zeros((2, num_replicates), dtype=np.float64)
        for i in range(n, 1, -1):
            rate = math.comb(i, 2)
            temp = rng.exponential(scale=1 / rate, size=num_replicates)
            result[0] += temp
            result[1] += i * temp
        return result


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

    def test_pick_segment_function(self):
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
            rng = np.random.default_rng(seeds[i])
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

    def test_pick_segment_function2(self):
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
        rec_correction = sim.expected_fraction_observed_rec_events(n)
        num_reps = 100
        rng = np.random.default_rng()
        tree_span_exp = self.sample_tree_span(n, rho, rng, num_reps)
        (
            tree_span_yaca,
            tree_span_yaca_exp,
            num_trees_yaca,
        ) = self.sample_yaca_tree_stats(n, rho, L, num_reps)

        # tree_span_hudson, num_trees_hudson = self.sample_hudson_tree_stats(
        #     n, rho, L, num_reps
        # )

        self.plot_qq(
            tree_span_yaca,
            tree_span_yaca_exp,
            "yaca",
            "yaca_exp",
            "marginal_tree_span_yaca_exp",
        )
        exp_total_branch_length = 2 * sum(1 / i for i in range(1, n))
        num_trees_exp = rng.poisson(exp_total_branch_length * rho/2 * L, size=num_reps)
        self.plot_qq(
            num_trees_yaca,
            num_trees_exp,
            "yaca",
            "exp",
            "num_trees_yaca_exp",
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

    def sample_yaca_tree_stats(self, n, rho, L, num_reps):
        seeds = self.get_seeds(num_reps)
        tss = self.run_yaca(n, rho, L, seeds)
        results = np.zeros((3, num_reps), dtype=np.float64)
        rec_correction = sim.expected_fraction_observed_rec_events(n)
        for i, ts in enumerate(tss):
            results[0, i] = ts.first().span
            rate = ts.first().total_branch_length * rho / 2
            results[1, i] = np.random.exponential(1 / (rate * rec_correction))
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


class TestMargTBL(Test):
    """
    Test marginal total branch length
    """

    def test_tbl(self, seed=None):

        rho = 1e-2
        L = 1000

        for n in [2, 4, 8, 20]:
            self.verify_single_model(n, rho, L, 500, seed)

    def verify_single_model(self, n, rho, L, num_replicates, seed):
        param_str = f"L_{L}_rho_{rho}_n{n}"
        tree_stats_labels = ["t1", "tn", "trt", "tbl"]
        tree_stats, av_num_trees = self.get_all_marginal_tree_stats(
            n, rho, L, num_replicates, seed, len(tree_stats_labels)
        )
        if seed is None:
            seed = random.randint(1, 2**16)
        rng = np.random.default_rng(seed)
        self.verify_marginal_tree_stats(
            tree_stats, n, rng, tree_stats_labels, param_str
        )
        logfile = self._build_filename(f"n_{n}/{param_str}/logfile", extension=".tsv")
        self.log_run(logfile, ["seed", "av_num_trees"], [[seed, av_num_trees]])

    def verify_marginal_tree_stats(self, yaca_stats, n, rng, labels, param_str=""):
        num_labels, num_pos, num_replicates = yaca_stats.shape
        exp_stats = self.get_expected_tree_stats(n, rng, (num_labels, num_replicates))
        self.require_output_dir(f"n_{n}/{param_str}")

        for i in range(num_labels):
            for j in range(num_pos):
                self.plot_qq(
                    yaca_stats[i, j],
                    exp_stats[i],
                    "yaca",
                    f"exp_{labels[i]}",
                    f"n_{n}/{param_str}/{labels[i]}_{param_str}_pos_{j}",
                )

    def get_all_marginal_tree_stats(
        self, sample_size, rho, L, num_replicates, seed, num_tree_stats
    ):
        positions = [0, L // 2, L - 1]
        tree_stats = np.zeros(
            (num_replicates, len(positions), num_tree_stats), dtype=np.float64
        )
        seeds = self.get_seeds(num_replicates, seed)
        tree_counter = 0
        ts_iter = self.run_yaca(sample_size, rho, L, seeds)
        # ts_iter = self.run_msprime(sample_size, rho, L, seeds)
        for idx, ts in enumerate(ts_iter):
            tree_stats[idx] = self.marginal_tree_stats(
                ts, positions, tree_stats.shape[1:]
            )
            tree_counter += ts.num_trees
        print(f"Average number of trees: {tree_counter/num_replicates}")
        return np.swapaxes(tree_stats, 0, 2), tree_counter / num_replicates

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


class TestRecAgainstMsp(Test):
    """
    Test number of trees against msprime

    """

    def test_num_trees(self, seed=None, test_yaca=True):
        rho = 5e-5
        L = 1e5
        num_reps = 500

        for n in [4, 8, 20]:
            self.verify_num_trees(
                n, rho, L, num_reps, seed, test_yaca
            )

    def verify_num_trees(
        self, n, rho, L, num_replicates, seed, test_yaca
    ):
        param_str = f"L_{L}_rho_{rho}_n{n}"
        if seed is None:
            seed = random.randint(1, 2**16)
        seeds = self.get_seeds(num_replicates, seed)
        num_trees_msp_exact = self.msp_exact_num_trees(n, rho, L, num_replicates, seed)

        if test_yaca:
            model = "yaca"
            num_trees = self.yaca_num_trees(n, rho, L, seeds)
            output_str = f"{model}_num_trees_msp_exact_n{n}_with_correction"
        else:
            model = "hudson"
            num_trees = self.msp_num_trees(n, rho, L, num_replicates, model=model)
            output_str = f"{model}_num_trees_msp_exact_n{n}"
        self.require_output_dir(f"n_{n}/{param_str}")
        self.plot_qq(
            num_trees,
            num_trees_msp_exact,
            model,
            "msp_exact",
            f"n_{n}/{param_str}/{output_str}",
        )
        logfile = self._build_filename(
            f"n_{n}/{param_str}/logfile_{model}", extension=".tsv"
        )
        self.log_run(
            logfile,
            [
                "seed",
            ],
            [
                [
                    seed,
                ]
            ],
        )

    def yaca_num_trees(self, n, rho, L, seeds):
        num_replicates = len(seeds)
        tss = self.run_yaca(n, rho, L, seeds)
        result = np.zeros(num_replicates, dtype=np.int64)
        for i, ts in enumerate(tss):
            result[i] = ts.num_trees
        return result

    def msp_num_trees(self, n, rho, L, num_replicates, ploidy=1, model="hudson"):
        num_trees = np.zeros(num_replicates, dtype=np.int64)
        for i, ts in tqdm(
            enumerate(
                msprime.sim_ancestry(
                    samples=n,
                    ploidy=ploidy,
                    num_replicates=num_replicates,
                    recombination_rate=rho / 2,
                    sequence_length=L,
                    model=model,
                )
            ),
            total=num_replicates,
            desc="running msprime sim_ancestry",
        ):
            num_trees[i] = ts.num_trees

        return num_trees


class TestSamplingConsistency(Test):
    def test_sampling_consistency(self, seed=None):

        rho = 1e-2
        L = 1000

        for n_max, n_min in zip([4, 8, 20], [2, 4, 8]):
            self.verify_single_model(
                n_max, n_min, rho, L, 500, seed
            )

    def verify_single_model(
        self, n_max, n_min, rho, L, num_replicates, seed
    ):

        param_str = f"L_{L}_rho_{rho}_n{n_max}_{n_min}"
        if seed is None:
            seed = random.randint(1, 2**16)
        rng = np.random.default_rng(seed)
        seeds = self.get_seeds(num_replicates, seed)

        positions = [0, L // 2, L - 1]
        tree_stats_labels = ["t1", "tn", "trt", "tbl"]
        num_tree_labels = len(tree_stats_labels)
        tree_stats_min = np.zeros(
            (num_replicates, len(positions), num_tree_labels), dtype=np.float64
        )
        num_trees = np.zeros(num_replicates)
        
        ts_iter = self.run_yaca(n_max, rho, L, seeds)
        for idx, ts in enumerate(ts_iter):
            ts_simple = self.downsample_ts(ts, n_min, rng)
            tree_stats_min[idx] = self.marginal_tree_stats(
                ts_simple, positions, tree_stats_min.shape[1:]
            )
            num_trees[idx] = ts_simple.num_trees
        tree_stats_min = np.swapaxes(tree_stats_min, 0, 2)

        exp_tree_stats = self.get_expected_tree_stats(
            n_min, rng, (num_tree_labels, num_replicates)
        )
        exp_num_trees = self.msp_exact_num_trees(n_min, rho, L, num_replicates, seed)
        self.require_output_dir(f"n_{n_max}_{n_min}/{param_str}")
        logfile = self._build_filename(
            f"n_{n_max}_{n_min}/{param_str}/logfile", extension=".tsv"
        )
        self.log_run(logfile, ["seed"], [[seed]])
        self.process_tree_stats(
            exp_tree_stats, tree_stats_min, tree_stats_labels, (n_max, n_min), param_str
        )
        self.process_num_trees(exp_num_trees, num_trees, (n_max, n_min), param_str)

    def process_tree_stats(self, exp_stats, obs_stats, stat_labels, n_tuple, param_str):
        num_labels, num_pos, _ = obs_stats.shape
        for i in range(num_labels):
            for j in range(num_pos):
                self.plot_qq(
                    obs_stats[i, j],
                    exp_stats[i],
                    "yaca",
                    f"exp_{stat_labels[i]}",
                    f"n_{'_'.join(str(t) for t in n_tuple)}/{param_str}/{stat_labels[i]}_pos_{j}",
                )

    def process_num_trees(self, exp_stat, obs_stat, n_tuple, param_str):
        self.plot_qq(
            obs_stat,
            exp_stat,
            "yaca",
            f"exp_num_trees",
            f"n_{'_'.join(str(t) for t in n_tuple)}/{param_str}/num_trees",
        )

    def downsample_ts(self, ts, n, rng):
        reduced_sample_set = rng.choice(ts.samples(), size=n, replace=False)
        return ts.simplify(samples=reduced_sample_set)


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


class TestSingleStep(Test):
    def test_all_single_step(self):
        self._test_single()
        self._test_single_n2()

    def _test_single(self, seeds=None):
        n = 16
        rho = 1e-3
        L = 1e5
        num_replicates = 500
        if isinstance(seeds, Iterable):
            num_runs = len(seeds)
            self.seeds = seeds
        else:
            num_runs = 10
            self.seeds = self.get_seeds(num_runs)
        run_until = 1.0
        param_str = f"first_coal_n{n}_rho{rho}_L{L}_ru{run_until}"
        info_labels = ["num_extant_lineages", "overlap", "num_trees", "diff_nonu_roots"]
        info_size = len(info_labels)
        log_info = np.zeros((num_runs, info_size))
        self.set_output_dir(self.output_dir, "test_first_coalescence_n16")

        for j in tqdm(range(num_runs)):
            log_info[j] = self.run_single_test(
                n,
                rho,
                L,
                num_replicates,
                run_until,
                param_str,
                j,
            )
        self.log_run(param_str, info_labels, log_info)

    def run_single_test(self, n, rho, L, num_replicates, run_until, param_str, j):
        coal_time_msp = np.zeros(num_replicates, dtype=np.float64)
        coal_time_yaca = np.zeros_like(coal_time_msp)
        rng = random.Random(self.seeds[j])
        ts = self.generate_lineages(n, rho, L, run_until, rng)
        lineages = list(self.ts_to_lineages(ts).values())
        num_extant_lineages = len(lineages)
        pairwise_overlap = self._pairwise_overlap(lineages)
        different_non_unary_roots = len(self.get_non_unary_roots(ts))
        rate_adjustor = different_non_unary_roots / len(lineages)

        for i in tqdm(range(num_replicates), disable=True):
            # contains time to first node coalesced after
            coal_time_msp[i] = self.run_msp_single_step(rho, ts, rng, run_until)
            new_time, pair_idx = self._sample_pairwise_times(
                lineages, pairwise_overlap, rng, run_until, rho, rate_adjustor
            )
            coal_time_yaca[i] = new_time

        info_dict = {
            "num_extant_lineages": num_extant_lineages,
            "overlap": round(pairwise_overlap[pair_idx]),
            "num_trees": ts.num_trees,
            "diff_nonu_roots": different_non_unary_roots,
        }
        info = self.make_info_str(info_dict)
        self.plot_qq(
            coal_time_yaca,
            coal_time_msp,
            "yaca",
            "msp",
            f"first_coalescence_{param_str}_{j}",
            info,
        )
        return list(info_dict.values())

    def _pairwise_overlap(self, lineages):
        num_lineages = len(lineages)
        pairwise_overlap = np.zeros(math.comb(num_lineages, 2), dtype=np.float64)
        for a, b in itertools.combinations(range(num_lineages), 2):
            _, overlap_length = sim.intersect_lineages(lineages[a], lineages[b])
            idx = sim.combinadic_map((a, b))
            pairwise_overlap[idx] = overlap_length
        return pairwise_overlap

    def _sample_pairwise_times(
        self, lineages, pairwise_overlap, rng, last_event, rho, p
    ):
        assert p <= 1
        num_lineages = len(lineages)
        pairwise_times = np.zeros(math.comb(num_lineages, 2), dtype=np.float64)
        lower_rec_fraction = sim.expected_fraction_observed_rec_events(2)

        for idx in range(len(pairwise_times)):
            overlap_length = pairwise_overlap[idx]
            if overlap_length > 0:
                a, b = list(sim.reverse_combinadic_map(idx))
                node_time_diff = abs(lineages[a].node_time - lineages[b].node_time)
                oldest_node = max(lineages[a].node_time, lineages[b].node_time)
                start_time_exp_process = last_event - oldest_node
                new_event_time = sim.draw_event_time_downsample(
                    1,
                    rho * overlap_length * lower_rec_fraction,
                    rng,
                    T=node_time_diff,
                    start_time=start_time_exp_process,
                    p=p,
                )
                new_event_time -= start_time_exp_process
                pairwise_times[idx] = new_event_time

        non_zero_times = np.nonzero(pairwise_times)[0]
        selected_idx = non_zero_times[np.argmin(pairwise_times[non_zero_times])]
        return pairwise_times[selected_idx], selected_idx

    def _test_single_pair_n16(self, seeds=None):
        n = 16
        rho = 1e-3
        L = 1e5
        num_replicates = 500
        if isinstance(seeds, Iterable):
            num_runs = len(seeds)
            self.seeds = seeds
        else:
            num_runs = 10
            self.seeds = self.get_seeds(num_runs)
        run_until = 1.5
        param_str = f"random_pair_n{n}_rho{rho}_L{L}_ru{run_until}"
        # info_labels = ["num_extant_lineages", "overlap", "num_segments", "num_trees"]
        info_labels = ["overlap", "num_trees"]
        info_size = len(info_labels)
        log_info = np.zeros((num_runs, info_size))
        self.set_output_dir(self.output_dir, "test_single_pair_n16")

        for j in tqdm(range(num_runs)):
            log_info[j] = self.run_single_test_pair(
                n, rho, L, num_replicates, run_until, param_str, j
            )
        self.log_run(param_str, info_labels, log_info)

    def run_single_test_pair(self, n, rho, L, num_replicates, run_until, param_str, j):
        coal_time_msp = np.zeros(num_replicates, dtype=np.float64)
        coal_time_yaca = np.zeros_like(coal_time_msp)
        rng = random.Random(self.seeds[j])
        ts, pair = self.generate_lineage_pair(n, rho, L, run_until, rng)
        lineages = self.ts_to_lineages(ts)
        num_extant_lineages = len(lineages)
        lineages = {i: lineages[i] for i in pair}
        node_time_diff = abs(lineages[pair[0]].node_time - lineages[pair[1]].node_time)
        overlap_intervals, overlap = sim.intersect_lineages(
            *(lineages[p] for p in pair)
        )
        overlap_rho = rho * overlap
        oldest_node = max([l.node_time for l in lineages.values()])
        last_event = run_until
        start_time_exp_process = last_event - oldest_node

        for i in tqdm(range(num_replicates), disable=True):
            # contains time to first node coalesced after
            coal_time_msp[i] = self.run_msp_single_step_pair(
                pair, rho, ts, rng, run_until
            )
            coal_time_yaca[i] = sim.draw_event_time_downsample(
                1, overlap_rho, rng, node_time_diff, start_time_exp_process
            )

        # oldest_node time is treated as zero
        coal_time_msp -= oldest_node
        info_dict = {
            "num_extant_lineages": num_extant_lineages,
            "overlap": round(overlap),
            "num_segments": len(overlap_intervals),
            "num_trees": ts.num_trees,
        }
        info = self.make_info_str(info_dict)
        self.plot_qq(
            coal_time_yaca,
            coal_time_msp,
            "yaca",
            "msp",
            f"single_step_{param_str}_{j}",
            info,
        )
        return list(info_dict.values())

    def _test_single_n2(self, seed=None):
        n = 2
        rho = 1e-3
        L = 1e5
        num_replicates = 500
        num_runs = 50
        run_until = 0.5
        param_str = f"n{n}_rho{rho}_L{L}"
        ssr_array = np.zeros(num_runs)
        overlap = np.zeros_like(ssr_array)
        self.seeds = self.get_seeds(num_runs)
        info_labels = ["overlap", "num_segments", "num_trees"]
        info_size = len(info_labels)
        log_info = np.zeros((num_runs, info_size))
        self.set_output_dir(self.output_dir, "test_single_pair_n2")

        for j in tqdm(range(num_runs)):
            log_info[j] = self.run_single_test_n2(
                n, rho, L, num_replicates, run_until, param_str, j
            )
        self.log_run(param_str, info_labels, log_info)

    def run_single_test_n2(self, n, rho, L, num_replicates, run_until, param_str, j):
        coal_time_msp = np.zeros(num_replicates, dtype=np.float64)
        coal_time_yaca = np.zeros_like(coal_time_msp)
        rng = random.Random(self.seeds[j])
        ts = self.generate_lineages(n, rho, L, run_until, rng)
        pair = (0, 1)
        lineages = self.ts_to_lineages(ts, pair)
        assert all(l.node_time == 0 for l in lineages.values())
        node_time_diff = 0
        overlap_intervals, overlap = sim.intersect_lineages(
            *(lineages[p] for p in pair)
        )
        overlap_rho = rho * overlap
        lower_rec_fraction = sim.expected_fraction_observed_rec_events(2)

        oldest_node = max([l.node_time for l in lineages.values()])
        last_event = run_until
        start_time_exp_process = last_event - oldest_node

        for i in tqdm(range(num_replicates), disable=True):
            # contains time to first node coalesced after
            coal_time_msp[i] = self.run_msp_single_step_pair(
                pair, rho, ts, rng, run_until
            )
            coal_time_yaca[i] = sim.draw_event_time_downsample(
                1,
                overlap_rho * lower_rec_fraction,
                rng,
                node_time_diff,
                start_time_exp_process,
            )

        # oldest_node time is treated as zero
        coal_time_msp -= oldest_node
        info_dict = {
            "overlap": round(overlap),
            "num_segments": len(overlap_intervals),
            "num_trees": ts.num_trees,
        }
        info = self.make_info_str(info_dict)
        self.plot_qq(
            coal_time_yaca,
            coal_time_msp,
            "yaca",
            "msp",
            f"n2_single_step_{param_str}_{j}",
            info,
        )
        return list(info_dict.values())

    def make_info_str(self, kwargs):
        result = ""
        for key, value in kwargs.items():
            result += f"{key}:{value}, "
        return result

    def ts_to_lineages(self, ts, idxs=None):
        lineages = dict()

        num_samples = ts.num_samples

        for tree in ts.trees():
            for root in tree.roots:
                if tree.num_children(root) == 1:
                    root = tree.children(root)[0]
                ancestral_to = tree.num_samples(root)
                if ancestral_to < num_samples:
                    left, right = tree.interval.left, tree.interval.right
                    new_ancestry_interval = sim.AncestryInterval(
                        left, right, ancestral_to
                    )
                    if root not in lineages:
                        lineages[root] = sim.Lineage(
                            root, [new_ancestry_interval], tree.time(root)
                        )
                    else:
                        prev = lineages[root].ancestry[-1]
                        if left == prev.right and ancestral_to == prev.ancestral_to:
                            prev.right = right
                        else:
                            lineages[root].ancestry.append(new_ancestry_interval)

        if isinstance(idxs, Iterable):
            return {i: lineages[i] for i in idxs}

        return lineages

    def get_non_unary_roots(self, ts):
        found = set()
        for tree in ts.trees():
            if tree.num_roots > 1:
                for root in tree.roots:
                    found.add(root)
        return found

    def generate_lineages(self, n, rho, L, run_until, rng):
        ret = False

        while not ret:
            new_seed = rng.randint(1, 2**16)
            ts = msprime.sim_ancestry(
                samples=n,
                sequence_length=L,
                recombination_rate=rho / 2,
                ploidy=1,
                discrete_genome=False,
                population_size=1,
                end_time=run_until,
                random_seed=new_seed,
            )
            ret = max(tree.num_roots for tree in ts.trees()) > 1

        return ts

    def run_msp_single_step(self, rho, ts, rng, sim_start_time):
        new_seed = rng.randint(1, 2**16)
        ts_new = msprime.sim_ancestry(
            recombination_rate=rho / 2,
            ploidy=1,
            discrete_genome=False,
            population_size=1,
            initial_state=ts,
            random_seed=new_seed,
        )

        old_time = ts.max_root_time
        assert old_time == sim_start_time
        times = ts_new.nodes_time
        last_coal_idx = np.sum(times <= sim_start_time)
        assert times[last_coal_idx - 1] == sim_start_time
        assert times[last_coal_idx] > sim_start_time
        return times[last_coal_idx] - times[last_coal_idx - 1]

    def generate_lineage_pair(self, n, rho, L, run_until, rng):
        ret = True

        while ret:
            ts = self.generate_lineages(n, rho, L, run_until, rng)
            pair = self.find_pair(ts, rng)
            ret = max(pair) == -1
        return ts, pair

    def find_pair(self, ts, rng):

        S = set()
        for tree in ts.trees():
            if tree.num_roots > 1:
                temp = []
                for root in tree.roots:
                    children = tree.children(root)
                    if len(children) == 1:
                        temp.append(children[0])
                for comb in itertools.combinations(temp, 2):
                    S.add(tuple(sorted(comb)))

        S = list(S)
        if len(S) == 0:
            return (-1, -1)
        idx = rng.randrange(len(S))
        return S[idx]

    def find_pair_coalesced(self, ts, pair, after=0.0):
        """
        Returns smallest node time for parent of pair after time after
        """
        temp = math.inf
        for tree in ts.trees():
            if tree.right_sib(pair[0]) == pair[1] or tree.left_sib(pair[0]) == pair[1]:
                node_time_parent = tree.time(tree.parent(pair[0]))
                if node_time_parent > after:
                    temp = min(temp, node_time_parent)
        return temp

    def run_msp_single_step_pair(self, pair, rho, ts, rng, resume_at=0.0):
        temp = math.inf
        while temp == math.inf:
            new_seed = rng.randint(1, 2**16)
            ts_new = msprime.sim_ancestry(
                recombination_rate=rho / 2,
                ploidy=1,
                discrete_genome=False,
                population_size=1,
                initial_state=ts,
                random_seed=new_seed,
            )
            ts_new, node_map = ts_new.simplify(map_nodes=True)
            new_pair = [node_map[i] for i in pair]
            temp = self.find_pair_coalesced(ts_new, new_pair, resume_at)

        return temp

    def log_run(self, param_str, info_labels, info_array):
        filename = self._build_filename("log_" + param_str, ".tsv")
        with open(filename, "w") as logfile:
            print("i\tseed\t" + "\t".join(label for label in info_labels), file=logfile)
            for i, seed in enumerate(self.seeds):
                print(
                    f"{i}\t{seed}\t" + "\t".join(str(a) for a in info_array[i]),
                    file=logfile,
                )


def run_tests(suite, output_dir):
    print(f'[+] Test suite contains {len(suite)} tests.')
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "TestRecombination",
        "TestMargTBL",
        "TestRecAgainstMsp",
        "TestSamplingConsistency",
        "TestSingleStep",
    ]

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
        default="_output/stats_tests_output",
        help="specify the base output directory",
    )

    args = parser.parse_args()

    run_tests(args.test_class, args.output_dir)


if __name__ == "__main__":
    main()
