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


class StatP:
    def __init__(self, basedir, cl_name):
        self.set_output_dir(basedir, cl_name)
        self.models = ["yaca", "hudson", "SMC", "SMC"]

    def set_output_dir(self, basedir, cl_name):
        output_dir = pathlib.Path(basedir) / cl_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

    def _get_tests(self):
        return [
            value for name, value in inspect.getmembers(self) if name.startswith("run_")
        ]

    def _run_all(self):
        all_results = self._get_tests()
        print(f"[+] Collected {len(all_results)} StatP method(s).")
        print("[+] Running ...")
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

    def _run_yaca(self, n, rho, L, seeds):
        for seed in tqdm(seeds, desc="Running yaca"):
            yield sim.sim_yaca(n, rho, L, seed=seed)

    def _run_msprime(self, n, rho, L, seeds, model):
        for seed in tqdm(seeds, desc="Running msp"):
            yield msprime.sim_ancestry(
                samples=n,
                recombination_rate=rho / 2,
                sequence_length=L,
                ploidy=1,
                population_size=1,
                discrete_genome=False,
                random_seed=seed,
                model=model,
            )


class StatPNumNodes(StatP):
    def run_this(self):
        ts = sim.sim_yaca(4, 0.15, 100)
        print(ts.draw_text())

    def _run_num_nodes(self, seed=None):
        rho = 5e-5
        L = 1e5
        num_reps = 500

        for n in [4, 8, 20]:
            apply_rec_correction = True
            self.record_num_nodes(
                n, rho, L, num_reps, seed, test_yaca, apply_rec_correction
            )

    def record_num_nodes(self):
        pass


class StatPBreakpoints(StatP):
    def run_num_breakpoints_variance(self, seed=None):
        # result would be a single number (across all replicates)
        # can be compared to other models
        pass


class StatPCorrelations(StatP):
    def run_corr_edge_span(self, seed=None):
        pass

    def _run_corr_edge_span(self):
        result = []
        models = ["yaca", "hudson", "smc", "smc"]
        for model in models:
            result.append(self._run_corr_edge_span_single_model(model))
        self.log_result(result)

    def _run_corr_edge_span_single_model(self, model="yaca"):
        results = []
        max_length = 0
        if model == "yaca":
            gen_ts = self._run_yaca(n, rho, L, seeds)
        else:
            gen_ts = self._run_msprime(n, rho, L, seeds, model)
        for ts in gen_ts:
            results.append(np.mean(span_array(ts)))

        return np.mean(results), np.var(results, ddof=1)

    def span_array(self, ts):
        d = dict()
        for edge in ts.edges():
            d[parent] = d.get(parent, 0) + edge.right - edge.left

        return np.array(d.values())

    def run_corr_tree_height(self, seed=None):
        pass


def run_all(suite, output_dir):
    for cl_name in suite:
        instance = getattr(sys.modules[__name__], cl_name)(output_dir, cl_name)
        instance._run_all()


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "StatPNumNodes",
        "StatPBreakpoints",
        "StatPCorrelations",
    ]

    parser.add_argument(
        "--class_methods",
        "-c",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all the specified test classes.",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="_output/stats_properties",
        help="specify the base output directory",
    )

    args = parser.parse_args()

    run_all(args.class_methods, args.output_dir)


if __name__ == "__main__":
    main()
