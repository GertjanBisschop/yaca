import itertools
import math
import numpy as np
import random
import tskit

from yaca import sim


def decap_to_overlap(ts, node):
    overlap = dict()
    children_pairs = []

    for tree in ts.trees():
        all_lineages = []
        for root in tree.roots:
            while tree.num_children(root) == 1:
                root = tree.children(root)[0]
            all_lineages.append(root)
            if root == node:
                children_pairs.append(tree.children(root))
        all_lineages.sort()
        for pair in itertools.combinations(all_lineages, 2):
            if pair not in overlap:
                overlap[pair] = 0
            overlap[pair] = overlap.get(pair, 0) + tree.span

    return overlap, children_pairs


def single_step_prob(pairwise_overlap, pairs, last_event, t):
    single_event = 0
    for pair in pairs:
        single_event += sim.coal_rate(1, rho, time, last_event, 1)
    mean_intensity = rate_integral(time, rho, overlap)
    p_next_event = np.exp(-np.sum(mean_intensity_all))
    return single_event * p_next_event


def get_p(ts):
    num_nodes = ts.num_nodes
    num_samples = ts.num_samples
    sequence_length = ts.sequence_length
    tables = ts.tables
    p = 1
    t = 0
    last_event = 0
    pairwise_overlap = {
        pair: sequence_length for pair in itertools.combinations(range(num_samples), 2)
    }

    for i in range(num_samples, num_nodes):
        t = tables.nodes[i].time
        # nodes[i] is registers the next coalescence event back in time
        # we need a list of all pairs of children of nodes[i]
        decap = ts.decapitate(t)
        updated_pairwise_overlap, pairs = decap_to_overlap(decap, tables.nodes[i])
        p *= single_step_prob(pairwise_overlap, pairs, last_event, t)
        pairwise_overlap = updated_pairwise_overlap.copy()
        last_event = t

    return p
