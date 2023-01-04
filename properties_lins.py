import argparse
import dataclasses
import inspect
import itertools
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
from collections.abc import Iterable
from typing import Callable
from typing import List

matplotlib.use("Agg")
import statsmodels.api as sm

from yaca import sim


@dataclasses.dataclass
class SimTracker:

    num_reps: int
    num_bins: int
    time_step: float
    samples: int
    rho: float
    sequence_length: float
    extract_info: List[Callable[[msprime.ancestry.Simulator], np.float64]]
    seed: int
    discrete_genome: bool = False


    def __post_init__(self):
        self.num_functions = len(self.extract_info)
        self.shape = (self.num_functions, self.num_reps, self.num_bins)
        self.rng = np.random.default_rng(self.seed)

    def get_seeds(self):
        max_seed = 2**16
        return self.rng.integers(1, max_seed, size=self.num_reps)

    def run_sim_stepwise(self, model="hudson", disable_tqdm=False):
        seeds = self.get_seeds()
        results = np.zeros(
            (self.num_functions, self.num_reps, self.num_bins), dtype=np.float64
        )
        for i in tqdm(range(self.num_reps), total=self.num_reps, disable=disable_tqdm):    
            time_iter = 0
            if model == 'yaca':
                simulator = sim.Simulator(
                    self.samples,
                    self.sequence_length,
                    self.rho,
                    seeds[i]
                    )
            else:
                simulator = msprime.ancestry._parse_sim_ancestry(
                    samples=self.samples,
                    recombination_rate=self.rho / 2,
                    sequence_length=self.sequence_length,
                    discrete_genome=self.discrete_genome,
                    model=model,
                    ploidy=1,
                    random_seed=seeds[i],
                )
            ret = msprime._msprime.EXIT_MAX_TIME

            while ret == msprime._msprime.EXIT_MAX_TIME or ret == 0:
                time = self.time_step * time_iter
                ret = simulator._run_until(time)
                for j, f in enumerate(self.extract_info):
                    results[j, i, time_iter] = f(simulator)
                time_iter += 1
                if time_iter >= self.num_bins:
                    break
            if time_iter < self.num_bins:
                for j, f in enumerate(self.extract_info):
                    results[j, i, time_iter:] = f(simulator)

        return results

    def run_models(self, *models):
        assert len(models) > 1, "At least 2 model names are required for a comparison"
        results = np.zeros((len(models), *self.shape), dtype=np.float64)
        for i, model in enumerate(models):
            results[i] = self.run_sim_stepwise(model=model)

        return results

    def process_result(self, f, result):
        # (num_models, num_functions, num_reps, num_time_steps)
        return np.apply_along_axis(f, result.ndim - 2, result)


def extract_num_lineages(sim):
    return sim.num_ancestors


def extract_total_anc_mat(sim):
    total = 0
    if sim.model == 'yaca':
        for anc in sim.ancestors:
            total += anc.ancestry[-1].right - anc.ancestry[0].left
    else:
        for anc in sim.ancestors:
            total += anc[-1][1] - anc[0][0]
    return total


def extract_num_nodes(sim):
    return sim.num_nodes


def extract_sim_width(sim):
    max_hull = 0
    if sim.model == 'yaca':
        for anc in sim.ancestors:
            hull = anc.ancestry[-1].right - anc.ancestry[0].left
            max_hull = max(max_hull, hull)
    else:
        for anc in sim.ancestors:
            hull = anc[-1][1] - anc[0][0]
            max_hull = max(max_hull, hull)
    return max_hull


def extract_mean_num_segments(sim):
    total = 0
    if sim.num_ancestors == 0:
        return 0
    if sim.model == 'yaca':
        for anc in sim.ancestors:
            total += len(anc.ancestry)
    else:
        for anc in sim.ancestors:
            total += len(anc)
    return total / sim.num_ancestors


def extract_median_num_segments(sim):
    temp = np.zeros(sim.num_ancestors, dtype=np.int64)
    if sim.num_ancestors == 0:
        return 0
    for i, anc in enumerate(sim.ancestors):
        temp[i] = len(anc)
    return np.median(temp)


def extract_mean_hull_width(sim):
    if sim.num_ancestors == 0:
        return 0
    total = 0
    if sim.model == 'yaca':
        for anc in sim.ancestors:
            total += anc.ancestry[-1].right - anc.ancestry[0].left
    else:
        for anc in sim.ancestors:
            total += anc[-1][1] - anc[0][0]
    return total / sim.num_ancestors / sim.sequence_length


def extract_mean_anc_material(sim):
    if sim.num_ancestors == 0:
        return 0
    total_anc_material = 0
    if sim.model == 'yaca':
        for anc in sim.ancestors:
            for segment in anc.ancestry:
                total_anc_material += segment.right - segment.left
    else:
        for anc in sim.ancestors:
            for segment in anc:
                total_anc_material += segment[1] - segment[0]
    return total_anc_material / sim.num_ancestors / sim.sequence_length

def compare_models_plot(result, time_step, models, functions, shape, figsize, filename, error=None):
    # shape of results is (models, functions, num_time_steps)
    if isinstance(error, np.ndarray):
            assert error.shape == result.shape
    
    fig, ax = plt.subplots(*shape, figsize=figsize)
    ax_flat = ax.flat
    x = np.arange(result.shape[-1]) * time_step
    for i, function in enumerate(functions):
        for j, label in enumerate(models):
            ax_flat[i].plot(x, result[j, i], label=models[j])
            if isinstance(error, np.ndarray):
                ax_flat[i].fill_between(
                    x,
                    result[j, i] - error[j, i],
                    result[j, i] + error[j, i], 
                    alpha=0.25
                )
            ax_flat[i].set_title(function)
    
    handles, labels = ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.savefig(filename, dpi=70)

def set_output_dir(output_dir, samples, info_str):
    output_dir = pathlib.Path(output_dir + f"/n_{samples}/" + info_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_all(fs, output_dir, seed):
    rho = 5e-5
    L = 1e5
    num_reps = 500
    #simulation will be tracked until time num_bins * timestep 
    num_bins = 10
    timestep = 0.5
    info_str = f"L_{L}_rho_{rho}"
    basedir = output_dir
    all_fs = []
    for key, value in globals().items():
        if callable(value) and value.__module__ == __name__:
            if key in fs:
                all_fs.append(value)
    models = ['yaca', 'hudson', 'smc']
    subplots_shape = (math.ceil(len(all_fs)/2), 2) 
    fig_dims = tuple(4 * i for i in subplots_shape[::-1])
    for n in [2, 4, 8, 20]:
        print(f'[+] Running sims for n = {n} ...')
        simtracker = SimTracker(
            num_reps,
            num_bins,
            timestep,
            n,
            rho,
            L,
            all_fs,
            seed
        )

        results = simtracker.run_models(*models)
        # graph results
        mresults = np.mean(results, axis=2)
        eresults = scipy.stats.sem(results, axis=2)
        
        output_dir = set_output_dir(basedir, n, info_str)
        filename = output_dir / 'simtrack_plot.png'
        compare_models_plot(
            mresults,
            timestep, 
            models,
            [f.__name__.lstrip('extract_') for f in all_fs],
            subplots_shape, # shape
            fig_dims, # size of fig
            filename,
            error=eresults
        )

def main():
    parser = argparse.ArgumentParser()
    choices = [
        "extract_num_lineages",
        "extract_mean_hull_width",
        "extract_mean_anc_material",
        "extract_mean_num_segments",
    ]

    parser.add_argument(
        "--functions",
        "-f",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all the specified functions.",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="_output/stats_properties_lins",
        help="specify the base output directory",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="specify used seed",
    )

    args = parser.parse_args()

    run_all(args.functions, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
