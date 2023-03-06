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
        self.rng = np.random.default_rng(self.seed)

    def set_output_dir(self):
        output_dir = pathlib.Path(
            self.output_dir + f"/n_{self.samples}/" + self.info_str
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def require_output_dir(self, folder_name):
        output_dir = self.output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _run_all(self, T):
        max_size = max(stat.size for stat in T)
        a = np.zeros(
            (len(self.models), len(T), self.num_reps, max_size), dtype=np.float64
        )
        for i, model in enumerate(self.models):
            self._run_single_model(model, a[i], T)
        for i, stat in enumerate(T):
            stat.plot(a[:, i, :, : stat.size])

    def _run_single_model(self, model, a, T):
        single_run = self._run_yaca() if model == "yaca" else self._run_msprime(model)
        for j, ts in enumerate(single_run):
            for i, stat in enumerate(T):
                ts = ts.simplify()
                a[i, j, : stat.size] = stat.compute(ts)

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def _run_yaca(self):
        seeds = self.get_seeds()
        for seed in tqdm(seeds, desc="Running yaca"):
            yield sim.sim_yaca(self.samples, self.rho, self.sequence_length, seed=seed)

    def _run_msprime(self, model):
        seeds = self.get_seeds()
        for seed in tqdm(seeds, desc=f"Running {model}"):
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
            plot_cdf(np.squeeze(a[..., i]), self.name, f, self.runner)


class CovDecay(TsStat):
    def __init__(self, runner, num_lineages=2):
        num_points = 20
        super().__init__(runner, num_points + 1)
        if runner.samples > 2:
            self.num_lineages = num_lineages
        else:
            self.num_lineages = 2

    def compute(self, ts):
        # requires many observations of t_i and t_j at given distances
        # t_i can stay same fixed value
        result = np.zeros(self.size, dtype=np.float64)
        points = np.arange(self.size) / (self.size) * ts.sequence_length
        u, v, *w = random.sample(range(ts.num_samples), self.num_lineages)
        # u, v = random.sample(range(ts.num_samples), 2)
        tree = ts.at(points[0])
        result[0] = tree.tmrca(u, v)
        if self.num_lineages == 3:
            v = w[0]
            # v = random.sample(range(ts.num_samples), 1)[0]
            # while u==v:
            #    v = random.sample(range(ts.num_samples), 1)[0]
        if self.num_lineages == 4:
            u, v = w
            # u, v = random.sample(range(ts.num_samples), 2)
        for i in range(1, self.size):
            tree = ts.at(points[i])
            result[i] = tree.tmrca(u, v)
        return result

    def compute_cov(self, a):
        # shape a: num_models, reps, size
        result = np.zeros((len(self.runner.models), self.size - 1), dtype=np.float64)
        for i in range(result.shape[0]):
            for j in range(1, self.size):
                result[i, j - 1] = (
                    np.sum(a[i, :, 0] == a[i, :, j]) / self.runner.num_reps
                )
        if self.num_lineages == 2:
            mean_result = np.zeros((len(self.runner.models)), dtype=np.float64)
            for i in range(mean_result.shape[0]):
                mean_result[i] = np.mean(a[i])
            with open(self._build_filename("mean_coal", extension=".txt"), "w") as file:
                for i in range(mean_result.shape[0]):
                    print(
                        self.runner.models[i] + "\t" + str(mean_result[i]),
                        file=file,
                    )

        with open(self._build_filename("", extension=".txt"), "w") as file:
            for i in range(result.shape[0]):
                print(
                    self.runner.models[i] + "\t" + "\t".join(str(v) for v in result[i]),
                    file=file,
                )
        return result

    def expected_cov(self, r, model="hudson"):
        if self.num_lineages == 2:
            if model == "hudson":
                return (r + 18) / (r**2 + 13 * r + 18)
            elif model == "smc":
                return 1 / (1 + r)
            elif model == "smc_prime":
                # approximately
                return 1 - 2 * r / 3
            else:
                raise ValueError("not implemented")
        elif self.num_lineages == 3:
            if model == "hudson":
                return 6 / (r**2 + 13 * r + 18)
            else:
                raise ValueError("not implemented")
        else:
            if model == "hudson":
                return 4 / (r**2 + 13 * r + 18)
            else:
                raise ValueError("not implemented")

    def plot_line(self, a, b, x_label, y_label, filename):
        marker = itertools.cycle((".", "+", "v", "^"))
        for i, model in enumerate(self.runner.models):
            x = a[i]
            plt.plot(
                b, x, label=model, marker=next(marker), markersize=10, linestyle="None"
            )
        exp = np.array([self.expected_cov(r, "hudson") for r in b])
        plt.plot(b, exp, marker="o", label=f"exp_hudson")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper right")
        plt.savefig(filename, dpi=72)
        plt.close("all")

    def plot(self, a):
        f = self._build_filename("")
        a = self.compute_cov(a)
        b = (
            np.arange(self.size)
            / self.size
            * self.runner.sequence_length
            * self.runner.rho
        )
        self.plot_line(a, b[1:], "rho", "cov", f)


class CovDecayIK(CovDecay):
    def __init__(self, runner):
        super().__init__(runner, 3)


class CovDecayKL(CovDecay):
    def __init__(self, runner):
        super().__init__(runner, 4)


class AutoCov(TsStat):
    def __init__(self, runner, d=-1):
        super().__init__(runner)
        self.d = d

    def tmrca_array(self, ts, u, v):
        num_points = int(ts.sequence_length // self.d)
        points = np.arange(num_points) * self.d
        results = np.zeros(num_points, dtype=np.float64)
        for i in range(num_points):
            tree = ts.at(points[i])
            results[i] = tree.tmrca(u, v)
        return results

    def lagged_auto_cov(self, Xi, t):
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
        end_padded_series = np.zeros(N + t)
        end_padded_series[:N] = Xi - Xs
        start_padded_series = np.zeros(N + t)
        start_padded_series[t:] = Xi - Xs

        auto_cov = 1.0 / (N - 1) * np.sum(start_padded_series * end_padded_series)
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
            plot_cdf(np.squeeze(a[..., i]), self.name, f, self.runner)


class AutoCovD100(AutoCov):
    def __init__(self, runner):
        super().__init__(runner, d=100)


class AutoCovD1000(AutoCov):
    def __init__(self, runner):
        super().__init__(runner, d=1000)


class AutoCovD10000(AutoCov):
    def __init__(self, runner):
        super().__init__(runner, d=10_000)


class BinnedCovDecay(TsStat):
    def __init__(self, runner):
        num_points = 4
        self.bins = np.array([0.5])
        self.num_bins = len(self.bins) + 2
        super().__init__(runner, (num_points + 1) * self.num_bins)

    def compute(self, ts):
        # requires many observations of t_i and t_j at given distances
        # t_i can stay same fixed value
        result = np.zeros(self.size, dtype=np.float64).reshape((-1, self.num_bins))
        num_points = int(self.size // self.num_bins)
        points = np.arange(num_points) / (num_points) * ts.sequence_length
        binned_pairs_array = self.binned_pairs(ts)

        for j in range(self.num_bins):
            u, v = binned_pairs_array[j]
            if u != v:
                tree = ts.at(points[0])
                result[0, j] = tree.tmrca(u, v)
                for i in range(1, num_points):
                    tree = ts.at(points[i])
                    result[i, j] = tree.tmrca(u, v)
            else:
                result[1:, j] = -1

        return result.reshape(-1)

    def binned_pairs(self, ts):
        tree = ts.first()
        value_pairs = dict()
        for u, v in itertools.combinations(ts.samples(), 2):
            bin_idx = np.digitize(tree.tmrca(u, v), self.bins)
            if bin_idx not in value_pairs:
                value_pairs[bin_idx] = set()
            value_pairs[bin_idx].add((u, v))
        result = np.zeros((self.num_bins, 2), dtype=np.int64)
        for bin_idx, pairs in value_pairs.items():
            result[bin_idx] = random.choice(tuple(pairs))

        return result

    def compute_cov(self, a):
        # size = (num_points + 1) * num_bins
        num_points = int(self.size // self.num_bins)
        result = np.zeros(
            (len(self.runner.models), self.num_bins, num_points - 1), dtype=np.float64
        )
        # shape a: num_models, reps, num_points + 1, num_bins
        a = a.reshape((len(self.runner.models), self.runner.num_reps, num_points, -1))

        for i in range(result.shape[0]):
            for j in range(self.num_bins):
                empty_bin = np.sum(a[i, :, 1, j] < 0)
                for k in range(1, num_points):
                    if empty_bin == self.runner.num_reps:
                        result[i, j, k - 1] = -1
                    else:
                        result[i, j, k - 1] = np.sum(a[i, :, 0, j] == a[i, :, k, j]) / (
                            self.runner.num_reps - empty_bin
                        )

        return result

    def expected_cov(self, r, model="hudson"):
        if model == "hudson":
            return (r + 18) / (r**2 + 13 * r + 18)
        elif model == "smc":
            return 1 / (1 + r)
        elif model == "smc_prime":
            # approximately
            return 1 - 2 * r / 3
        else:
            raise ValueError("not implemented")

    def plot_line(self, a, b, x_label, y_label, filename):
        num_models, num_bins, num_points = a.shape
        color = iter(plt.cm.rainbow(np.linspace(0, 1, num_models)))
        for j in range(num_models):
            marker = itertools.cycle((".", "+", "v", "^"))
            c = next(color)
            for i in range(num_bins):
                x = a[j, i]
                if np.all(x >= 0):
                    plt.plot(
                        b,
                        x,
                        label=self.runner.models[j] + f"_bin_{i}",
                        color=c,
                        marker=next(marker),
                        markersize=10,
                        linestyle="None",
                    )
        exp = np.array([self.expected_cov(2 * r, "hudson") for r in b])
        plt.plot(b, exp, marker="o", label=f"exp_hudson")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        color = iter(plt.cm.rainbow(np.linspace(0, 1, num_models)))
        handles = [f("s", next(color)) for i in range(num_models)]
        marker = itertools.cycle((".", "+", "v", "^"))
        handles += [f(next(marker), "k") for i in range(num_bins)]
        labels = self.runner.models[:] + [f"bin_{i}" for i in range(num_bins)]
        plt.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        # plt.legend(loc="lower right")
        plt.savefig(filename, dpi=72)
        plt.close("all")

    def plot(self, a):
        # a shape: num_models, num_bins, num_points
        a = self.compute_cov(a)
        num_points = int(self.size // self.num_bins)
        b = (
            np.arange(num_points)
            / num_points
            * self.runner.sequence_length
            * self.runner.rho
            / 2
        )
        f = self._build_filename("")
        self.plot_line(a, b[1:], "rho", "cov", f)


class EdgeStat(TsStat):
    def __init__(self, runner):
        super().__init__(runner)
        self.num_bins = 1
        self.bins = np.array([0, math.inf])
        self.single_bin = 0
        if self.num_bins == 1:
            assert self.single_bin == 0

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

    def _bin_node_hull(self, ts, d, hull):
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
            plot_cdf(np.squeeze(a[..., i]), self.name, f, self.runner)


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


class NonMarkovian(TsStat):
    def __init__(self, runner):
        num_points = 6
        super().__init__(runner, num_points)
        self.positions = np.array([0.01 * 2**i for i in range(num_points + 1)])

    def compute(self, ts):
        """
        Determine whether we observe a topology change between
        two points along the genome that have the same TMRCA for
        a random pair of lineages.
        """
        results = -np.ones(self.size, dtype=np.int8)
        u, v = random.sample(range(ts.num_samples), 2)
        tree_0 = ts.at(0)
        mrca_0 = tree_0.mrca(u, v)
        pos = self.positions[0] * self.runner.sequence_length
        tree_half = ts.at(pos)
        mrca_half = tree_half.mrca(u, v)
        mrca_0_half_equal = mrca_0 == mrca_half
        for i in range(1, self.size + 1):
            pos = self.positions[i] * self.runner.sequence_length
            tree_1 = ts.at(pos)
            mrca_1 = tree_1.mrca(u, v)
            mrca_01_equal = mrca_0 == mrca_1
            if mrca_01_equal:
                results[i - 1] = mrca_0_half_equal
            mrca_half = mrca_1
            mrca_0_half_equal = mrca_01_equal
        return results

    def compute_q(self, a):
        result = np.zeros((len(self.runner.models), self.size), dtype=np.float64)
        for i in range(result.shape[0]):
            for j in range(self.size):
                result[i, j] = np.sum(a[i, :, j] == 0) / np.sum(a[i, :, j] >= 0)
        return result

    def compute_p(self, a):
        result = np.zeros((len(self.runner.models), self.size), dtype=np.float64)
        for i in range(result.shape[0]):
            for j in range(self.size):
                result[i, j] = np.sum(a[i, :, j] >= 0) / a[i, :, j].size
        return result

    def plot_line(self, a, b, x_label, y_label, filename):
        marker = itertools.cycle((".", "+", "v", "^", "o"))
        for i, model in enumerate(self.runner.models):
            x = a[i]
            plt.plot(
                b, x, label=model, marker=next(marker), markersize=10, linestyle="None"
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper left")
        plt.savefig(filename, dpi=72)
        plt.close("all")

    def plot(self, a):
        f = self._build_filename("Q_")
        r = self.compute_q(a)
        b = self.positions * self.runner.sequence_length * self.runner.rho
        self.plot_line(r, b[1:], "rho", "Q", f)
        r = self.compute_p(a)
        f = self._build_filename("P_")
        self.plot_line(r, b[1:], "rho", "P", f)


class KC(TsStat):
    def compute(self, ts):
        tree_0 = ts.first()
        if tree_0.span == ts.sequence_length:
            return 0

        tree_0 = ts.first(sample_lists=True)
        for i, bp in enumerate(ts.breakpoints()):
            if i == 0:
                pass
            elif i == 1:
                return tree_0.kc_distance(ts.at(bp, sample_lists=True), 1.0)
            else:
                break

    def plot(self, a):
        for i in range(self.size):
            f = self._build_filename("cdf_")
            plot_cdf(np.squeeze(a[..., i]), self.name, f, self.runner)

class KCDecay(TsStat):
    def __init__(self, runner):
        num_points = 20
        super().__init__(runner, num_points - 1)

    def compute(self, ts):
        result = np.zeros(self.size, dtype=np.float64)
        points = np.arange(self.size + 1) / (self.size + 1) * ts.sequence_length
        first_tree = ts.first(sample_lists=True)
        for i in range(self.size):
            tree = ts.at(points[i + 1], sample_lists=True)
            result[i] = first_tree.kc_distance(tree, 1.0)
        return result

    def plot_line(self, a, b, x_label, y_label, filename):
        marker = itertools.cycle((".", "+", "v", "^"))
        for i, model in enumerate(self.runner.models):
            x = a[i]
            plt.plot(
                b, x, label=model, marker=next(marker), markersize=10, linestyle="None"
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper right")
        plt.savefig(filename, dpi=72)
        plt.close("all")

    def plot(self, a):
        f = self._build_filename("")
        #shape a: num_models, reps, num_positions
        mean_a = np.mean(a, axis=1)
        b = (
            np.arange(self.size + 1)
            / self.size + 1
            * self.runner.sequence_length
            * self.runner.rho
            / 2
        )
        self.plot_line(mean_a, b[1:], "rho", "kc", f)


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
        y = np.arange(len(x)) / float(len(x))
        plt.plot(x, y, label=model)
    plt.xlabel(x_label)
    plt.ylabel("cdf")
    plt.legend(loc="lower right")
    plt.savefig(filename, dpi=120)
    plt.close("all")


def plot_line(a, b, x_label, y_label, filename, stat_obj):
    for i, model in enumerate(stat_obj.models):
        x = a[i]
        plt.plot(b, x, label=model)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper right")
    plt.savefig(filename, dpi=120)
    plt.close("all")


def run_all(suite, output_dir, seed):
    rho = 1e-4
    # rho = 5e-5
    L = 1e5
    num_reps = 1000
    apply_rec_correction = True

    for n in [2, 4, 8, 10, 20]:
        print(f"[+] Running models for n={n}")
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
        "CovDecayIK",
        "CovDecayKL",
        "BinnedCovDecay",
        "AutoCovD100",
        "AutoCovD1000",
        "AutoCovD10000",
        "HullEdge",
        "MeanAncMatEdge",
        "NonMarkovian",
        "KC",
        "KCDecay",
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
