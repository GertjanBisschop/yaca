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

def run_tests(suite, output_dir):
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_tests()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "",
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