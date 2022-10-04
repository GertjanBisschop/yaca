import argparse
import inspect

import msprime
import numpy as np
# import scipy
import tskit
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import sys

matplotlib.use("Agg")
import statsmodels.api as sm

import yaca

class Test:
    def __init__(self, basedir):
        self.set_output_dir(basedir)

    def set_output_dir(self, basedir):
        output_dir = pathlib.Path(basedir) / 'foo'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def _get_tests(self):
        for name, value in inspect.getmembers(self):
            if name.startswith("test_"):
                yield value

    def _run_tests(self):
        for method in self._get_tests():
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


class TestFoo(Test):
    def test_foo(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        self.plot_qq(x, y, 'foo', 'bar', 'test')

class TestBar(Test):
    def test_bar(self):
        print("bar")


class TestAnalyticalExpectation(Test):
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

    def get_analytical_tbl(self, n, t):
        """
        Returns the probabily density of the total branch length t with
        a sample of n lineages. Wakeley Page 78.
        """
        t1 = (n - 1) / 2
        t2 = math.exp(-t / 2)
        t3 = pow(1 - math.exp(-t / 2), n - 2)
        return t1 * t2 * t3

    def makefig(self):
        m.graphics.qqplot(tbl_ms)
        sm.qqplot_2samples(tbl_ms, tbl_msp, line="45")
        pyplot.savefig(self.output_dir / f"qqplot_{n}.png", dpi=72)
        pyplot.close("all")


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
        # use hidden msp API
        #  We need to get direct access to the simulator here because of the
        # "invisible" recombination breakpoints, so we can't run simulations
        # the usual way via sim_ancestry.
        # from msprime.cli -> msprime.cli.SimulationRunner
        pass


class TestNonHomPoisson(Test):
    def test_time_to_first_event(self):
        exp = get_first_coalescence_time(1000, parameters)
        parameters = {
            "sequence_length": 1_000_000,
            "recombination_rate": 1e-7,
            "population_size": 1e4,
            "samples": 1,
            "ploidy": 2,
        }
        obs = self.get_times_non_hom_poiss(parameters)


    def get_first_coalescence_time(self, reps, parameters):
        """
        Simulate reps and extract the time to the first coalescence event.
        """
        num_samples = parameters["ploidy"] * parameters["samples"]
        results = np.zeros(reps, dtype=np.float64)
        for i in range(reps):
            ts = msprime.sim_ancestry(**parameters)
            results[i] = ts.node(num_samples).time
        results /= 2 * parameters["population_size"]
        return results

    def get_times_non_hom_poiss(self, parameters):
        rho = (
            4
            * parameters["population_size"]
            * parameters["recombination_rate"]
            * parameters["sequence_length"]
        )

        result = draw_value_non_hom(10_000, rho, 2)

    def draw_value_non_hom(reps, rho_overlap, num_lineages):
        """
        Given expected coalescence rate (k choose 2)* (2*T*rho), draw  reps random values
        from the non-homogeneous exponential distribution.
        rho is the total overlap of all combinations expressed in recombination rate units.
        Algorithm see Raghu Pasupathy, Generating Nonhomogeneous Poisson process.
        length as recombination rate = length * rho
        rho should be total amount of overlap across all combinations
        """
        results = np.zeros(reps, dtype=np.float64)
        random_values = np.random.rand(reps)
        s = -np.log(random_values)
        for i in range(reps):
            results[i] = inverse_expectation_function(
                s[i], rho_overlap / 2, num_lineages
            )
        return results


def run_tests(suite, output_dir):
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir)
        instance._run_tests()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        # "TestNonHomPoisson",
        # "TestAgainstMsp",
        # "TestAnalyticalExpectation",
        "TestFoo",
        "TestBar",
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
