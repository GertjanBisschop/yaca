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
        self.models = ["yaca", "hudson", "smc", "smc_prime"]
        #self.models = ["yaca", "hudson"]
        self.rng = np.random.default_rng(self.seed)

    def set_output_dir(self):
        output_dir = pathlib.Path(self.output_dir + f"/n_{self.samples}/" + self.info_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _run_all(self, T):
        max_size = max(stat.size for stat in T)
        a = np.zeros(
            (len(self.models), len(T), self.num_reps, max_size),
            dtype=np.float64
            )
        for i, model in enumerate(self.models):
            self._run_single_model(model, a[i], T)
        for i, stat in enumerate(T):
            stat.plot(a[:, i,:, :stat.size])

    def _run_single_model(self, model, a, T):
        single_run = self._run_yaca() if model == "yaca" else self._run_msprime(model)
        for j, ts in enumerate(single_run):
            for i, stat in enumerate(T):
                a[i, j, :stat.size] = stat.compute(ts)

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def _run_yaca(self):
        seeds = self.get_seeds()
        for seed in tqdm(seeds, desc="Running yaca"):
            yield sim.sim_yaca(self.samples, self.rho, self.sequence_length, seed=seed)

    def _run_msprime(self, model):
        seeds = self.get_seeds()
        for seed in tqdm(seeds, desc="Running msp"):
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
    size: int = 1

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
        for i in range(self.size):
            f = self._build_filename("cdf_")
            plot_cdf(np.squeeze(a[...,i]), self.name, f, self.runner)

class CovDecay(TsStat):
    def __init__(self, runner):
        num_points = 20
        super().__init__(runner, num_points + 1)

    def compute(self, ts):
        # requires many observations of t_i and t_j at given distances
        # t_i can stay same fixed value
        result = np.zeros(self.size, dtype=np.float64)
        points = np.arange(self.size)/self.size * ts.sequence_length
        u, v = random.sample(range(ts.num_samples), 2)
        for i in range(self.size):
            tree = ts.at(points[i])
            result[i] = tree.tmrca(u, v)
        return result

    def compute_cov(self, a):
        # shape a: num_models, reps, size
        result = np.zeros((len(self.runner.models), self.size - 1), dtype=np.float64)
        for i in range(result.shape[0]):
            for j in range(1, self.size):
                #result[i, j - 1] = np.cov(a[i, :, 0], a[i, :, j])[0][1]
                result[i, j-1] = np.sum(a[i, :, 0] == a[i, :, j])/self.runner.num_reps
        return result

    def expected_cov(self, r, model):
        if model == 'hudson':
            return (r + 18) / (r**2 + 13*r + 18)
        elif model == 'smc':
            return 1 / (1 + r)
        elif model == 'smc_prime':
            # approximately
            return 1 - 2*r/3
        else:
            raise ValueError("not implemented")

    def plot_line(self, a, b, x_label, y_label, filename):
        for i, model in enumerate(self.runner.models):
            x = a[i]
            plt.plot(b, x, label=model)
        exp = np.array([self.expected_cov(2*r, 'hudson') for r in b])
        plt.plot(b, exp, label=f'exp_hudson')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper right')
        plt.savefig(filename, dpi=72)
        plt.close("all")

    def plot(self, a):
        f = self._build_filename("")
        a = self.compute_cov(a)
        b = np.arange(self.size)/self.size * self.runner.sequence_length * self.runner.rho / 2
        self.plot_line(a, b[1:], 'rho', 'cov', f)

class AutoCov(TsStat):
    def __init__(self, runner, d=-1):
        super().__init__(runner)
        self.d = d

    def tmrca_array(self, ts, u, v):
        num_points = int(ts.sequence_length//self.d)
        points = np.arange(num_points) * self.d
        results = np.zeros(num_points, dtype=np.float64)
        for i in range(num_points):
            tree = ts.at(points[i])
            results[i] = tree.tmrca(u, v)
        return results

    def lagged_auto_cov(self, Xi,t):
        """
        for series of values x_i, length N, 
        compute empirical auto-cov with lag t defined: 
        1/(N-1) * sum_{i=0}^{N-t} ( x_i - x_s ) * ( x_{i+t} - x_s )
        """
        N = len(Xi)

        # use sample mean estimate from whole series
        Xs = np.mean(Xi)

        # construct copies of series shifted relative to each other, 
        # with mean subtracted from values
        end_padded_series = np.zeros(N+t)
        end_padded_series[:N] = Xi - Xs
        start_padded_series = np.zeros(N+t)
        start_padded_series[t:] = Xi - Xs

        auto_cov = 1./(N-1) * np.sum( start_padded_series*end_padded_series )
        return auto_cov

    def compute(self, ts):
        # compute probability that lagged
        # tmrc_array has same value
        
        u, v = random.sample(range(ts.num_samples), 2)
        a = self.tmrca_array(ts, u, v)
        return self.lagged_auto_cov(a, 1)
    
    def plot(self, a):
        for i in range(self.size):
            f = self._build_filename("cdf_")
            plot_cdf(np.squeeze(a[...,i]), self.name, f, self.runner)

class AutoCovD100(AutoCov):
    def __init__(self, runner):
        super().__init__(runner, d=100)

class AutoCovD1000(AutoCov):
    def __init__(self, runner):
        super().__init__(runner, d=1000)

class AutoCovD10000(AutoCov):
    def __init__(self, runner):
        super().__init__(runner, d=10_000)

class EdgeStat(TsStat):

    def __init__(self, runner):
        super().__init__(runner)
        self.num_bins = 1
        self.bins = np.array([0, math.inf])
        self.single_bin = 0
        if self.num_bins == 1:
            assert self.single_bin==0

    def _node_hull_dict(self, ts):
        # {parent_id: [left_min, right_max, span]}
        result = dict()
        for left, right, p in zip(ts.edges_left, ts.edges_right, ts.edges_parent):
            if p not in result:
                result[p] = [left, right, right - left]
            else:
                result[p][0] = min(result[p][0], left)
                result[p][1] = max(result[p][1], right)
                result[p][2] += right - left
        return result

    def _bin_node_hull(self, ts, d,  hull):
        result = np.zeros(self.num_bins + 1, dtype=np.float64)
        bin_counts = np.zeros(self.num_bins + 1, dtype=np.int64)
        time_bins = np.digitize(ts.nodes_time, self.bins)
        
        for key, value in d.items():
            time_bin = time_bins[key]
            if hull:
                result[time_bin] += value[1] - value[0]
            else:
                result[time_bin] += value[2]
            bin_counts[time_bin] += 1
        non_zero = bin_counts > 0
        result[non_zero] /= bin_counts[non_zero]
        return result[1:][self.single_bin]

    def compute(self, ts, hull):
        d = self._node_hull_dict(ts)
        return self._bin_node_hull(ts, d, hull)

    def plot(self, a):
        for i in range(self.size):
            f = self._build_filename("cdf_")
            plot_cdf(np.squeeze(a[...,i]), self.name, f, self.runner)

class HullEdge(EdgeStat):
    """
    Computes mean hull width for all edges.
    """
    
    def compute(self, ts):
        return super().compute(ts, True)

class MeanAncMatEdge(EdgeStat):
    """
    Computes mean total ancestral material for each edge.
    """

    def compute(self, ts):
        return super().compute(ts, False)

def plot_qq(v1, v2, x_label, y_label, filename, stat_obj, info=""):
    sm.graphics.qqplot(v1)
    sm.qqplot_2samples(v1, v2, line="45")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(info)
    f = stat_obj._build_filename(filename)
    plt.savefig(f, dpi=120)
    plt.close("all")

def plot_histogram(x, x_label, filename, stat_obj):
    n, bins, patches = plt.hist(x, density=True, bins="auto")
    plt.xlabel(x_label)
    plt.ylabel("density")
    plt.savefig(filename, dpi=120)
    plt.close("all")

def plot_cdf(a, x_label, filename, stat_obj):
    for i, model in enumerate(stat_obj.models):
        x = np.sort(a[i])
        y = np.arange(len(x))/float(len(x))
        plt.plot(x, y, label=model)
    plt.xlabel(x_label)
    plt.ylabel("cdf")
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=120)
    plt.close("all")

def plot_line(a, b, x_label, y_label, filename, stat_obj):
    for i, model in enumerate(stat_obj.models):
        x = a[i]
        plt.plot(b, x, label=model)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    plt.savefig(filename, dpi=120)
    plt.close("all")    

def run_all(suite, output_dir, seed):
    rho = 5e-5
    L = 1e5
    num_reps = 500
    apply_rec_correction = True

    for n in [2, 4, 8, 20]:
        print(f'[+] Running models for n={n}')
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
        "CovDecay",
        "AutoCovD100",
        "AutoCovD1000",
        "AutoCovD10000",
        "HullEdge",
        "MeanAncMatEdge",
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
