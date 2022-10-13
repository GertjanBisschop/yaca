import argparse
import inspect

import msprime
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


class Test:
    def __init__(self, basedir, cl_name):
        self.set_output_dir(basedir, cl_name)

    def set_output_dir(self, basedir, cl_name):
        output_dir = pathlib.Path(basedir) / cl_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

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

    def get_seeds(self, num_replicates, seed=None):
        rng = np.random.default_rng(seed)
        max_seed = 2**16
        return rng.integers(1, max_seed, size=num_replicates)

    def run_yaca(self, rho, L, n, num_replicates, rejection):
        seeds = self.get_seeds(num_replicates)
        for seed in tqdm(seeds, desc='Running yaca'):
            # yield sim.sim_yaca_ind(n, parameters["rho"], parameters["L"], seed=seed, rejection=rejection)
            yield sim.sim_yaca(n, rho, L, seed=seed, rejection=rejection)
    

class TestAnalytical(Test):

    def no_test_tbl_n2(self):
        n = 2
        parameters = {
            "rho" :7.5e-4,
            "L" : 1000,
            }
        num_replicates = 100

        for rejection in True, False:
            print(f"rejection sampling : {rejection}")
            trt, tbl, _, _ = self.get_marginal_tree_stats(parameters, n, num_replicates, rejection)
            self.verify_marginal_tree_stats(trt, tbl, n, rejection)

    def no_test_tbl_n4(self):
        n = 4
        parameters = {
            "rho" :7.5e-4,
            "L" : 1000,
            }
        num_replicates = 100

        for rejection in True, False:
            print(f"rejection sampling : {rejection}")
            trt, tbl, t1, tn = self.get_marginal_tree_stats(parameters, n, num_replicates, rejection)
            self.verify_marginal_tree_stats(trt, tbl, n, rejection)

    def no_test_tbl_n8(self):
        n = 8
        parameters = {
            "rho" :7.5e-4,
            "L" : 1000,
            }
        num_replicates = 500

        for rejection in True, False:
            print(f"rejection sampling : {rejection}")
            trt, tbl, t1, tn = self.get_marginal_tree_stats(parameters, n, num_replicates, rejection)
            self.verify_marginal_tree_stats(trt, tbl, n, rejection, t1, tn)
        
    def no_test_against_msprime(self):
        num_replicates = 1000
        sample_size = 8
        ploidy = 1
        n = ploidy * sample_size
        recombination_rate = 1e-8
        rho = 2 * ploidy * recombination_rate * 1e4
        sequence_length = 100000
        check_total = 0
        obs = np.zeros(num_replicates, dtype=np.float64)
        tn = np.zeros_like(obs)
        for i, ts in tqdm(enumerate(msprime.sim_ancestry(
            samples=sample_size,
            ploidy=ploidy, 
            num_replicates=num_replicates,
            recombination_rate=rho,
            sequence_length = sequence_length
            )), 
        total=num_replicates):
            tree = ts.first()
            check_total += ts.num_trees
            obs[i] = tree.time(tree.root)
            tn[i] = max(tree.time(u) for u in tree.children(tree.root))
        tn = obs - tn
        exp = self.sample_marginal_tree_depth(n, num_replicates)
        exp_tn = np.random.exponential(scale=1, size=num_replicates)
        self.plot_qq(obs, exp, 'msprime', 'analytical', 'tree_depth_msp')
        self.plot_qq(tn, exp_tn, 'msprime', 'analytical', 'tn_msp')
        print(f'mean num trees: {check_total/num_replicates}')

    def get_marginal_tree_stats(self, parameters, sample_size, num_replicates, rejection):
        ts_iter = self.run_yaca(parameters['rho'], parameters['L'], sample_size, num_replicates, rejection)
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
            tn[i] = tree.time(tree.root) - max(tree.time(u) for u in tree.children(tree.root))
            t1[i] = min(tree.time(tree.parent(u)) for u in tree.samples())
        print(f"Average number of trees: {check_count/num_replicates}")
        return trt, tbl, t1, tn

    def verify_marginal_tree_stats(self, trt, tbl, n, rejection, t1=None, tn=None):
        rejection_str = 'rejection_sampl' if rejection else 'weighted_sampl'
        num_replicates = trt.size
        exp_trt = self.sample_marginal_tree_depth(n, num_replicates)
        self.plot_qq(trt, exp_trt, 'yaca', 'sum_exp', f'tree_depth_n_{n}_'+rejection_str)
        if isinstance(t1, np.ndarray) and isinstance(tn, np.ndarray):
            exp_t1 = np.random.exponential(scale=1/math.comb(n, 2), size=num_replicates)
            exp_tn = np.random.exponential(scale=1, size=num_replicates)
            self.plot_qq(t1, exp_t1, 'yaca', 't1', f't1_n_{n}_'+rejection_str)
            self.plot_qq(tn, exp_tn, 'yaca', 'tn', f'tn_n_{n}_'+rejection_str)

        # hist_ms, bin_edges = np.histogram(tbl_ms, 20, density=True)
        # index = bin_edges[:-1]
        # analytical = [self.get_analytical_tbl(n, x * 2) for x in index]

    def sample_marginal_tree_depth(self, n, num_replicates):
        result = np.zeros(num_replicates, dtype=np.float64)
        for i in range(n, 1, -1):
            rate = math.comb(i, 2)
            result += np.random.exponential(
                scale=1/rate,
                size=num_replicates
                )
        return result

    def mean_marginal_tree_depth(self, n):
        return 2 * (1- 1 / n)

    def get_analytical_tbl(self, n, t):
        """
        Returns the probabily density of the total branch length t with
        a sample of n lineages. Wakeley Page 78.
        """
        t1 = (n - 1) / 2
        t2 = math.exp(-t / 2)
        t3 = pow(1 - math.exp(-t / 2), n - 2)
        return t1 * t2 * t3

    def no_test_breakpoint_distribution(self):
        rho = 7.5e-4
        L = 5000
        ploidy = 2
        sample_size = 2
        
        rejection = False
        n = ploidy * sample_size
        ts_gen = self.run_yaca(rho, L, n, 1, rejection)
        self.verify_breakpoint_distribution(next(ts_gen), rho, n, L)

    def verify_breakpoint_distribution(self, ts, rho, sample_size, L):
        area = [tree.total_branch_length * tree.span for tree in ts.trees()]
        scipy.stats.probplot(area, dist=scipy.stats.expon(rho/2), plot=plt, fit=False)
        path = self.output_dir / f"verify_breakpoint_distribution_n{sample_size}.png"
        plt.savefig(path)
        plt.close("all")

class TestAgainstMsp(Test):

    def test_num_trees(self):
        sample_size = 1
        n = 2 * sample_size
        r = 1e-9
        Ne = 1e4
        L = 1e4
        rho = 2 * Ne * r
        num_replicates = 100

        num_trees_yaca = self.yaca_num_trees(rho, L, n, num_replicates, False)
        num_trees_msp = self.msp_num_trees(n, r, Ne, L, num_replicates)
        self.plot_qq(num_trees_yaca, num_trees_msp, "yaca", "msp", f"num_trees_n{n}")

    def yaca_num_trees(self, rho, L, n, num_replicates, rejection):
        tss = self.run_yaca(rho, L, n, num_replicates, rejection)
        result = np.zeros(num_replicates, dtype=np.int64)
        for i, ts in enumerate(tss):
            result[i] = ts.num_trees
        return result

    def msp_num_trees(self, n, r, Ne, L, num_replicates):
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
        for k in tqdm(range(num_replicates), desc='Running msprime'):
            exact_sim.run()
            num_trees[k] = exact_sim.num_breakpoints
            exact_sim.reset()
        return num_trees

class TestNonHomExp(Test):
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

        #return self.draw_value_non_hom(10_000, total_overlap, num_lineages)
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
            results[i] = sim.draw_event_time(
                num_lineages,
                rho_overlap,
                rng
            )
        return results

    def draw_min_non_hom(self, reps, rho_overlap, num_lineages):
        rng = random.Random()
        results = np.zeros(reps, dtype=np.float64)
        for i in range(reps):
            time = np.inf
            for _ in range(math.comb(num_lineages, 2)):
                temp = sim.draw_event_time(
                    2,
                    rho_overlap,
                    rng
                )
                time = min(time, temp)
            results[i] = time

        return results

class TestFoo(Test):
    def no_test_foo(self):
        n = 2
        rho = 1e-4
        L = 1e4
        num_replicates = 10
        rejection = True
        max_trees = 0
        for i, (ts, seed) in enumerate(self.run_yaca(rho, L, n, num_replicates, rejection)):
            print(seed, ts.num_trees)
            max_trees = max(ts.num_trees, max_trees)
        print('yaca_max_trees:', max_trees)

    def no_test_against_msprime(self):
        num_replicates = 100
        sample_size = 8
        ploidy = 1
        n = ploidy * sample_size
        rho = 7.5e-4 * 2
        sequence_length = 1000
        max_trees = 0
        
        for i, ts in tqdm(enumerate(msprime.sim_ancestry(
            samples=sample_size,
            ploidy=ploidy, 
            num_replicates=num_replicates,
            recombination_rate=rho,
            sequence_length = sequence_length
            )), 
        total=num_replicates):
            max_trees = max(ts.num_trees, max_trees)
            
        print('msp:', max_trees)

    def no_test_foo_2(self):
        n = 2
        rho = 1e-4
        L = 5e4
        print("4Ner * L:", rho * L)
        reps = 100
        num_trees = np.zeros(reps, dtype=np.int64)
        for i in range(reps):
            seed = random.randint(0, 2**16)
            ts = sim.sim_yaca(n, rho, L, seed=seed)
            num_trees[i] = ts.num_trees
            print(ts.num_trees)
            if ts.num_trees > 50:
                print('seed:', seed)
                break
        print(reps)

    def no_test_foo_msp(self):
        n = 2
        r = 1e-8
        Ne = 2.5e3
        rho = 1e-4
        L = 5e4
        print((self.msp_num_breakpoints(n, r, Ne, L, 100)))

    def no_test_run_with_seed(self):
        #expected number of breakpoints: 4*Ne*r*L = 5
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
        for k in tqdm(range(num_replicates), desc='Running msprime'):
            exact_sim.run()
            num_trees[k] = exact_sim.num_breakpoints
            exact_sim.reset()
        return num_trees

    def no_test_foo_3(self):
        a = [(0, 2238.0, 1), (37933.0, 37949.0, 1), (40696.0, 40882.0, 1), (49653.0, 50000.0, 1)]
        a = [sim.AncestryInterval(left, right, ancestral_to) for left, right, ancestral_to in a]
        b = [(0, 2238.0, 1), (37933.0, 37949.0, 1), (40696.0, 40882.0, 1), (49653.0, 50000.0, 1)]
        a = [sim.AncestryInterval(left, right, ancestral_to) for left, right, ancestral_to in b]
        n = 2
        rho = 1e-4
        L = 5e4
        total_overlap = 2787.0
        for _ in range(10):
            seed = random.randint(0, 2**16)
            left, right = sim.pick_breakpoints(a, total_overlap, rho, 2, (0, 0), seed)
            print(left, right)


def run_tests(suite, output_dir):
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "TestNonHomExp",
        "TestAgainstMsp",
        "TestAnalytical",
        "TestFoo"
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
        default="stats_tests_output",
        help="specify the base output directory",
    )

    args = parser.parse_args()

    run_tests(args.test_class, args.output_dir)


if __name__ == "__main__":
    main()
