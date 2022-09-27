import dataclasses
import itertools
import math
import numpy as np
import random
import tskit

from typing import List


@dataclasses.dataclass
class AncestryInterval:
    """
    Records that the specified interval contains genetic material ancestral
    to the specified number of samples.
    """

    left: int
    right: int
    ancestral_to: int

    @property
    def span(self):
        return self.right - self.left

    @property
    def num_links(self):
        return self.span - 1


@dataclasses.dataclass
class Lineage:
    """
    A single lineage that is present during the simulation of the coalescent
    with recombination. The node field represents the last (as we go backwards
    in time) genome in which an ARG event occured. That is, we can imagine
    a lineage representing the passage of the ancestral material through
    a sequence of ancestral genomes in which it is not modified.
    """

    node: int
    ancestry: List[AncestryInterval]
    node_time: int = 0

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str((interval.left, interval.right, interval.ancestral_to))
            s += ", "
        if len(self.ancestry) > 0:
            s = s[:-2]
        return s + "]"

    @property
    def num_recombination_links(self):
        """
        The number of positions along this lineage's genome
        at which a recombination event can occur.
        """
        return self.right - self.left - 1

    @property
    def left(self):
        """
        Returns the leftmost position of ancestral material.
        """
        return self.ancestry[0].left

    @property
    def right(self):
        """
        Returns the rightmost position of ancestral material.
        """
        return self.ancestry[-1].right


@dataclasses.dataclass
class Node:
    time: float
    flags: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)


def inverse_expectation_function(x, rho, k):
    c = k * (k - 1) / 2
    return (-c + math.sqrt(c**2 + 4 * x * rho)) / (2 * rho)


def inverse_expectation_function_extended(x, rho, k, T):
    """
    T is mean time of nodes to last coalescenc event.
    """
    c = k * (k - 1) / 2
    d = 2 * T * rho
    return (-c - d + math.sqrt((c + d) ** 2 + 4 * x * rho)) / (2 * rho)


def draw_event_time(num_lineages, rho, rng, T=0):
    """
    Given expected coalescence rate (k choose 2)* (2*T*rho),
    draw single random values from the non-homogeneous
    exponential distribution
    rho is the total overlap of all combinations expressed
    in recombination rate units.
    Cinclar Algorithm (3) adapted from
    "Raghu Pasupathy, Generating Nonhomogeneous Poisson process".
    """
    u = rng.uniform(0, 1)
    s = math.log(u)
    return inverse_expectation_function_extended(s, rho / 2, num_lineages, T)


def intersect_lineages(a, b):
    """
    Returns list with the overlap between the ancestry intervals
    of Lineages a and b.
    """
    n = len(a.ancestry)
    m = len(b.ancestry)
    i = j = 0
    overlap = []
    overlap_length = 0
    while i < n and j < m:
        if a.ancestry[i].right <= b.ancestry[j].left:
            i += 1
        elif a.ancestry[i].left > b.ancestry[j].right:
            j += 1
        else:
            left = max(a.ancestry[i].left, b.ancestry[j].left)
            right = min(a.ancestry[i].right, b.ancestry[j].right)
            overlap.append(
                AncestryInterval(
                    left,
                    right,
                    a.ancestry[i].ancestral_to + b.ancestry[j].ancestral_to
                )
            )
            overlap_length += right - left
            if a.ancestry[i].right < b.ancestry[j].right:
                i += 1
            else:
                j += 1
    return (overlap, overlap_length)


def pick_breakpoints(overlap, total_overlap, rho, T, node_times, seed):
    rng = np.random.default_rng(None)
    # divide rho by 2, num breakpoints for single lineage
    overlap_rec_units = total_overlap * rho / 2
    # num_breakpoints = 0
    breakpoints = np.zeros(0, dtype=np.int64)
    for t in node_times:
        # numpy poisson pass lambda = expected number of events
        num_breakpoints = rng.poisson((T - t) * overlap_rec_units)
        num_breakpoints = min(num_breakpoints, total_overlap - 1)
        breakpoints = np.hstack(
            (
                breakpoints,
                rng.choice(
                    np.arange(1, total_overlap),
                    num_breakpoints,
                    replace=False
                ),
            )
        )

    # some breakpoints might be identical
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) > 0:
        # sample breakpoint to define single interval
        idx = rng.integers(len(breakpoints))
        left = breakpoints[idx - 1] if idx else 0
        right = breakpoints[idx]
    else:
        left = 0
        right = total_overlap
    return (left, right)


def merge_segment(overlap, breakpoints):
    start_break, end_break = breakpoints
    remaining = start_break

    i = 0
    while overlap[i].span <= remaining:
        remaining -= overlap[i].span
        i += 1
    left = overlap[i].left + remaining
    num_links = overlap[i].right - left - 1

    remaining = max(end_break - start_break - 1, 1)
    while num_links < remaining:
        yield AncestryInterval(left, overlap[i].right, overlap[i].ancestral_to)
        remaining -= num_links
        i += 1
        left = overlap[i].left
        num_links = overlap[i].span

    yield AncestryInterval(left, left + remaining, overlap[i].ancestral_to)


def remove_segment(current_lineage, to_remove):
    n = len(current_lineage.ancestry)
    m = len(to_remove)
    i = j = 0
    left = current_lineage.ancestry[i].left
    while i < n and j < m:
        if current_lineage.ancestry[i].right <= to_remove[j].left:
            yield current_lineage.ancestry[i]
            i += 1
        else:
            if current_lineage.ancestry[i].left != to_remove[j].left:
                yield AncestryInterval(
                    current_lineage.ancestry[i].left,
                    to_remove[j].left,
                    current_lineage.ancestry[i].ancestral_to,
                )
            while j < m - 1:
                left = to_remove[j].right
                j += 1
                right = min(
                    to_remove[j].left,
                    current_lineage.ancestry[i].right
                    )
                if right == current_lineage.ancestry[i].right:
                    i += 1
                    break
                yield AncestryInterval(
                    left, right, current_lineage.ancestry[i].ancestral_to
                )
            i += 1
            j += 1

    while i < n:
        yield current_lineage.ancestry[i]
        i += 1


def fully_coalesced(lineages, n):
    """
    Returns True if all segments are ancestral to n samples in all
    lineages.
    """
    for lineage in lineages:
        for segment in lineage.ancestry:
            if segment.ancestral_to < n:
                return False
    return True


def combinadic_map(sorted_pair):
    return int((sorted_pair[0]) + sorted_pair[1] * (sorted_pair[1] - 1) / 2)


def reverse_combinadic_map(idx, k=2):
    """
    To test this code block.
    for combo in itertools.combinations(range(10),2):
    result = tuple(list(reverse_combinadic_map(combinadic_map(combo)))[::-1])
    assert result== combo, 'fail'
    """
    while k > 0:
        i = k - 1
        num_combos = 0
        while num_combos <= idx:
            i += 1
            num_combos = math.comb(i, k)
        yield i - 1
        idx -= math.comb(i - 1, k)
        k -= 1


def sample_pairwise_rates(lineages, t, rng):
    # total_overlap_test = 0
    pairwise_rates = np.zeros(math.comb(len(lineages), 2), dtype=np.int64)
    for a, b in itertools.combinations(range(len(lineages)), 2):
        _, overlap_length = intersect_lineages(lineages[a], lineages[b])
        rate = (
            1
            + (2 * t - (lineages[a].node_time + lineages[b].node_time))
            * overlap_length
            )
        pairwise_rates[combinadic_map((a, b))] = rate
        # total_overlap_test += overlap_length
    # assert total_overlap == total_overlap_test, 'total_overlap wrong'

    # draw random pair based on rates
    selected_idx = rng.choices(
        range(pairwise_rates.size), weights=pairwise_rates
        )[0]
    a, b = reverse_combinadic_map(selected_idx)
    overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    return a, b, overlap, overlap_length


def sample_rejection(lineages, rng):

    overlap_length = 0
    while overlap_length == 0:
        a, b = rng.sample(range(len(lineages), k=2))
        overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    return a, b, overlap, overlap_length


def sim_yaca(n, rho, L, seed=None, rejection=True):
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    nodes = []
    total_overlap = L * n * (n - 1) / 2
    t = 0

    for _ in range(n):
        lineages.append(Lineage(len(nodes), [AncestryInterval(0, L, 1)], t))
        nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

    while total_overlap > 0:
        # waiting time to next coalescence event
        mean_time_to_last_event = sum(
            t - lineage.node_time for lineage in lineages
            ) / len(lineages)
        new_event_time = draw_event_time(
            len(lineages), total_overlap * rho, rng, mean_time_to_last_event
        )
        t += new_event_time

        if rejection:
            a, b, overlap, overlap_length = sample_rejection(lineages, rng)
        else:
            a, b, overlap, overlap_length = sample_pairwise_rates(
                                                        lineages, t, rng
                                                        )

        # pick breakpoints at rate (new_event_time - lineage.node_time)
        node_times = ()
        breakpoints = pick_breakpoints(
            overlap, overlap_length, rho, t, node_times, seed
        )
        c = Lineage(len(nodes), [], t)
        for interval in merge_segment(overlap, breakpoints):
            for lineage in lineages[a], lineages[b]:
                tables.edges.add_row(
                    interval.left, interval.right, c.node, lineage.node
                )

            c.ancestry.append(interval)
        # remove interval from old lineage
        for lineage in lineages[a], lineages[b]:
            lineage.ancestry = list(remove_segment(lineage, c.ancestry))

        # update total_overlap
        total_overlap -= overlap_length

    assert total_overlap == 0, "total_overlap less than 0!"
    assert fully_coalesced(lineages, n), \
        "Not all segments are ancestral to n samples."
    for node in nodes:
        tables.nodes.add_row(
            flags=node.flags,
            time=node.time,
            metadata=node.metadata
            )
    tables.sort()
    # tables.edges.squash()?
    return tables.tree_sequence()
