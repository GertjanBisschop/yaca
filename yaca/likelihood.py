import itertools
import math
import numpy as np
import random
import tskit

from yaca import sim


def decap_to_overlap(ts, node):
    # returns all possible coalescing lineages at time t
    # children pairs of node
    overlap = dict()
    node_children = set()

    for tree in ts.trees():
        all_lineages = []
        for root in tree.roots:
            if tree.num_children(root) == 1 and root != node:
                root = tree.children(root)[0]
            all_lineages.append(root)
            if root == node:
                children_sorted = tuple(sorted(tree.children(root)))
                if len(children_sorted) > 1:
                    node_children.add(children_sorted)

        all_lineages.sort()
        for pair in itertools.combinations(all_lineages, 2):
            overlap[pair] = overlap.get(pair, 0) + tree.span

    return overlap, node_children


def rate(time, T, overlap):
    """
    if time to overal last event is set as 0
    T = 2 * (TL-t_o) + d with TL the time of the last_event
    and d the time difference between both involved nodes.
    t_o time to the oldest node
    """
    return 1 + (2 * time + T) * overlap


def rate_integral(time, T, overlap):
    t2 = overlap
    t1 = overlap * T + 1
    return time * (t2 * time + t1)


def single_step_prob(
    pairwise_overlap, coalescing, last_event, delta_t, rho, node_times
):
    single_event = 1
    mean_intensity = 0

    for pair in coalescing:
        # in hudson algorithm CA_EVENTS can lead to multiple nodes
        # merging in single event (all on same lineage however)

        T = 2 * last_event - sum(node_times[i] for i in pair)
        single_event += (2 * delta_t + T) * pairwise_overlap[pair]
    for pair, overlap in pairwise_overlap.items():
        overlap = pairwise_overlap[pair]
        if overlap > 0 or rho == 0:
            T = 2 * last_event - sum(node_times[i] for i in pair)
            mean_intensity += rate_integral(delta_t, T, overlap)
    # in case multiple nodes merge in single event, these are all
    # on same lineage. For these lineages we need to correct
    # the computed mean intensity.
    mean_intensity -= (len(coalescing) - 1) * delta_t

    assert single_event > 0
    assert mean_intensity > 0
    return np.log(single_event) - mean_intensity


def log_ts_likelihood(ts, rho):
    num_nodes = ts.num_nodes
    num_samples = ts.num_samples
    sequence_length = ts.sequence_length
    tables = ts.tables
    p = 0
    t = 0
    last_event = 0
    pairwise_overlap = {
        pair: sequence_length * rho / 2
        for pair in itertools.combinations(range(num_samples), 2)
    }

    for i in range(num_samples, num_nodes):
        t = tables.nodes[i].time
        delta_t = t - last_event
        # nodes[i] is the next coalescence event back in time
        # we need a list of all pairs of children of nodes[i]
        decap = ts.decapitate(t)
        updated_pairwise_overlap, coalescing = decap_to_overlap(decap, i)
        p += single_step_prob(
            pairwise_overlap, coalescing, last_event, delta_t, rho, ts.nodes_time
        )
        pairwise_overlap = {k: v * rho / 2 for k, v in updated_pairwise_overlap.items()}
        last_event = t

    return p
