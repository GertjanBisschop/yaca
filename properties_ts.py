import argparse
import dataclasses
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


@dataclasses.dataclass
class TsStatRunner:

    num_reps: int
    samples: int
    rho: float
    sequence_length: float
    output_dir: str
    seed: int = None

    def __post_init__(self):
        self.info_str = f"L_{self.sequence_length}_rho_{self.rho}"
        self.set_output_dir()
        #self.models = ["yaca", "hudson", "smc", "smc_prime"]
        self.models = ["yaca", "hudson"]
        self.seeds = self.get_seeds()

    def set_output_dir(self):
        output_dir = pathlib.Path(self.output_dir + f"/n_{self.samples}/" + self.info_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _run_all(self, T):
        a = np.zeros(
            (len(self.models), len(T), self.num_reps),
            dtype=np.float64
            )
        for i, model in enumerate(self.models):
            self._run_single_model(model, a[i], T)
        for i, stat in enumerate(T):
            stat.plot(a[:, i])

    def _run_single_model(self, model, a, T):
        single_run = self._run_yaca() if model == "yaca" else self._run_msprime(model)
        for j, ts in enumerate(single_run):
            for i, stat in enumerate(T):
                a[i, j] = stat.compute(ts)

    def get_seeds(self):
        rng = np.random.default_rng(self.seed)
        max_seed = 2**16
        return rng.integers(1, max_seed, size=self.num_reps)

    def _run_yaca(self):
        for seed in tqdm(self.seeds, desc="Running yaca"):
            yield sim.sim_yaca(self.samples, self.rho, self.sequence_length, seed=seed)

    def _run_msprime(self, model):
        for seed in tqdm(self.seeds, desc="Running msp"):
            yield msprime.sim_ancestry(
                samples=self.samples,
                recombination_rate=self.rho / 2,
                sequence_length=self.sequence_length,
                ploidy=1,
                population_size=1,
                discrete_genome=False,
                random_seed=seed,
                model=model,
            )

@dataclasses.dataclass
class TsStat:
    runner: TsStatRunner

    def __post_init__(self):
        self.set_output_dir()

    @property
    def name(self):
        return self.__class__.__name__

    def set_output_dir(self):
        self.output_dir = self.runner.output_dir

    def _build_filename(self, stat_type, extension=".png"):
        return self.output_dir / (stat_type + self.name + extension)


class NumNodes(TsStat):
    
    def compute(self, ts):
        return ts.num_nodes

    def plot(self, a):
        f = self._build_filename("cdf_")
        plot_cdf(a, self.name, f, self.runner)

class Corrs(TsStat):
    def __init__(self, d=-1):
        self.d = d

    def tmrca_array(ts, u, v, num_points):
        points = np.arange(ts.sequence_length, num_points)
        results = np.zeros(num_points, dtype=np.float64)
        for i in range(num_points):
            tree = ts.at(points[i])
            results[i] = tree.tmrca(u, v)
        return results

    def compute(self, ts):
        # compute probability that lagged
        # tmrc_array has same value
        if d==-1:
            tree1 = ts.at(ts.first())
            tree2 = tree1.next()
        else:
            tree1 = ts.at(0)
            tree2 = ts.at(self.d)
        
        # compute probability that T_i == T_j
    
    def plot(self, a):
        pass

class CorrsD100(Corrs):
    def __init__(self):
        super().__init__(100)

class CorrsD1000(Corrs):
    def __init__(self):
        super().__init__(1000)

class EdgeStat(TsStat):

    def _node_hull_dict(self, ts):
        # {parent_id: [left_min, right_max, span]}
        result = dict()
        for left, right, p in zip(ts.edges_left, ts.edges_right, ts.edges_parent):
            if p not in result:
                result[p] = [left, right, right - left]
            else:
                result[p][1] = right
                result[p][2] += right - left
        return result

    def _bin_node_hull(self, d, bins, hull):
        result = np.zeros(len(bins) + 1, dtype=np.float64)
        bin_counts = np.zeros(len(bins) + 1, dtype=np.int64)
        time_bins = np.digitize(ts.nodes_time, bins)
    
        for key, value in d.items():
            time_bin = time_bins[key]
            if hull:
                result[time_bin] += value[1] - value[0]
            else:
                result[time_bin] += value[2]
            bin_counts[time_bin] += 1
        non_zero = bin_counts > 0
        result[:, non_zero] /= bin_counts[non_zero]
        return result

    def compute(self, ts, hull):
        bins = np.arange(10)
        d = _node_hull_dict(ts)
        return _bin_node_hull(d, bins, hull)

    def plot(self, a):
        f = self._build_filename("cdf_")
        plot_cdf(a, self.name, f, self.runner)

class Hull(EdgeStat):
    
    def compute(self, ts):
        super().compute(ts, True)

class TotalEdges(EdgeStat):

    def compute(self, ts):
        super().compute(ts, False)

def plot_qq(v1, v2, x_label, y_label, filename, stat_obj, info=""):
    sm.graphics.qqplot(v1)
    sm.qqplot_2samples(v1, v2, line="45")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(info)
    f = stat_obj._build_filename(filename)
    plt.savefig(f, dpi=72)
    plt.close("all")

def plot_histogram(x, x_label, filename, stat_obj):
    n, bins, patches = plt.hist(x, density=True, bins="auto")
    plt.xlabel(x_label)
    plt.ylabel("density")
    plt.savefig(filename, dpi=72)
    plt.close("all")

def plot_cdf(a, x_label, filename, stat_obj):
    for i, model in enumerate(stat_obj.models):
        x = np.sort(a[i])
        y = np.arange(len(x))/float(len(x))
        plt.plot(x, y, label=model)
    plt.xlabel(x_label)
    plt.ylabel("cdf")
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=72)
    plt.close("all")

def run_all(suite, output_dir, seed):
    rho = 5e-5
    L = 1e5
    num_reps = 100
    apply_rec_correction = True

    for n in [2,]:
        all_stats = []
        S = TsStatRunner(num_reps, n, rho, L, output_dir, seed)
        for cl_name in suite:
            instance = getattr(sys.modules[__name__], cl_name)(S)
            all_stats.append(instance)
        S._run_all(all_stats)


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "NumNodes",
        "Corrs",
        "CorrsD100",
        "CorrsD1000",
        "Hull",
        "TotalEdges",
    ]

    parser.add_argument(
        "--methods",
        "-m",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all the specified methods.",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="_output/stats_properties_ts",
        help="specify the base output directory",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="specify seed",
    )

    args = parser.parse_args()

    run_all(args.methods, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
