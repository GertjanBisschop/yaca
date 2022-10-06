import argparse
import inspect

import msprime
import numpy as np
import random
# import scipy
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
        print(f"Collected {len(all_results)} tests.")
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
        return rng.randint(1, max_seed, size=num_replicates)


class TestAnalytical(Test):

    def no_test_tbl_n2(self):
        sample_size = 2
        parameters = {
            "rho" : 1e-8,
            "L" : 1_000_000,
            "Ne" : 1e4,
            }
        num_replicates = 100

        trt, tbl = self.get_marginal_tree_stats(parameters, sample_size, num_replicates)
        self.verify_marginal_tree_stats(sample_size)
        
    def no_test_against_msprime(self):
        num_replicates = 1000
        sample_size = 4
        ploidy = 1
        n = ploidy * sample_size
        recombination_rate = 1e-8
        rho = 2 * ploidy * recombination_rate * 1e4
        sequence_length = 100

        obs = np.zeros(num_replicates, dtype=np.float64)
        for i, ts in tqdm(enumerate(msprime.sim_ancestry(
            samples=sample_size,
            ploidy=ploidy, 
            num_replicates=num_replicates,
            recombination_rate=rho,
            sequence_length = sequence_length
            )), 
        total=num_replicates):
            tree = ts.first()
            obs[i] = tree.time(tree.root)
        exp = self.sample_marginal_tree_depth(n, num_replicates)
        self.plot_qq(obs, exp, 'msprime', 'analytical', 'tree_depth')

    def run_yaca(parameters, num_replicates, rejection):
        seeds = self.get_seeds(num_replicates)
        for seed in seeds:
            yield sim.yaca(n, rho, L, seed=seed, rejection=rejection)
    
    def get_marginal_tree_stats(num_replicates, rejection):
        ts_iter = run_yaca(parameters, num_replicates, rejection)
        # keeping track of total root time and total branch length
        trt = np.zeros(num_replicates, dtype=np.float64) 
        tbl = np.zeros(num_replicates, dtype=np.float64)
        
        for i, ts in enumerate(ts_iter):
            tree = ts.first()
            trt[i] tree.time(tree.root)
            tbl[i] = tree.total_branch_length

        return trt, tbl

    def verify_marginal_tree_stats(trt, tbl, n):
        num_replicates = trt.size
        exp_trt = self.sample_marginal_tree_depth(n, num_replicates)
        self.plot_qq(obs, exp, 'yaca', 'sum_exp', 'tree_depth')

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

    def get_analytical_tbl(self, n, t):
        """
        Returns the probabily density of the total branch length t with
        a sample of n lineages. Wakeley Page 78.
        """
        t1 = (n - 1) / 2
        t2 = math.exp(-t / 2)
        t3 = pow(1 - math.exp(-t / 2), n - 2)
        return t1 * t2 * t3


class TestAgainstMsp(Test):
    def verify_num_trees(self):
        pass

    def verify_total_branch_length_first_tree(self):
        pass

    def run_replicates(self, num_replicates, seed):
        rng = np.random.default_rng(seed)
        seeds = rng.randint(1, 2**31, size=num_replicates)
        tbl_array = np.zeros(num_replicates, dtype=np.float64)
        num_trees_array = np.zeros_like(tbl_array)
        for i, seed in enumerate(seeds):
            ts = yaca.sim.sim_yaca(seed)
            tbl_array[i] = self.get_tbl(ts)
            num_trees_array[i] = self.get_num_trees(ts)

    def verify_breakpoint_distribution(
        self, name, sample_size, Ne, r, L, ploidy, model, growth_rate=0
    ):
        ts = msprime.sim_ancestry(
            samples=sample_size,
            demography=msprime.Demography.isolated_model(
                [Ne], growth_rate=[growth_rate]
            ),
            ploidy=ploidy,
            sequence_length=L,
            recombination_rate=r,
            model=model,
        )
        area = [tree.total_branch_length * tree.span for tree in ts.trees()]
        scipy.stats.probplot(area, dist=scipy.stats.expon(Ne * r), plot=pyplot)
        path = self.output_dir / f"{name}_growth={growth_rate}_ploidy={ploidy}.png"
        logging.debug(f"Writing {path}")
        pyplot.savefig(path)
        pyplot.close("all")

    def test_analytical_num_trees(self):
        """
        Runs the check for number of trees using the CLI.
        """
        r = 1e-8  # Per generation recombination rate.
        num_loci = np.linspace(100, 10**5, 10).astype(int)
        Ne = 10**4
        n = 100
        ## see num_loci - 1!!!!!
        rho = r * 4 * Ne * (num_loci - 1)
        num_replicates = 100
        ms_mean = np.zeros_like(rho)
        msp_mean = np.zeros_like(rho)
        for j in range(len(num_loci)):
            cmd = f"{n} {num_replicates} -T -r {rho[j]} {num_loci[j]}"
            T = self.get_num_trees(
                _ms_executable + cmd.split() + self.get_ms_seeds(), num_replicates
            )
            ms_mean[j] = np.mean(T)

            T = self.get_num_trees(
                _mspms_executable + cmd.split() + self.get_ms_seeds(), num_replicates
            )
            msp_mean[j] = np.mean(T)
        pyplot.plot(rho, ms_mean, "o")
        pyplot.plot(rho, msp_mean, "^")
        pyplot.plot(rho, rho * harmonic_number(n - 1), "-")
        pyplot.savefig(self.output_dir / "mean.png")
        pyplot.close("all")

    def msp_num_trees(self):
        # IMPORTANT!! We have to use the get_num_breakpoints method
        # on the simulator as there is a significant drop in the number
        # of trees if we use the tree sequence. There is a significant
        # number of common ancestor events that result in a recombination
        # being undone.
        num_trees = np.zeros(num_replicates)
        exact_sim = msprime.ancestry._parse_simulate(
            sample_size=n, recombination_rate=r, Ne=Ne, length=L
        )
        for k in range(num_replicates):
            exact_sim.run()
            num_trees[k] = exact_sim.num_breakpoints
            exact_sim.reset()
        return num_trees

class TestNonHomPoisson(Test):
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

        return self.draw_value_non_hom(10_000, total_overlap, 2)

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

def run_tests(suite, output_dir):
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "TestNonHomPoisson",
        "TestAgainstMsp",
        "TestAnalytical",
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
