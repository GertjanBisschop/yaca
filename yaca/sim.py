import dataclasses
import itertools
import math
import numpy as np
import random
import tskit

from typing import List

import sys


@dataclasses.dataclass
class AncestryInterval:
    """
    Records that the specified interval contains genetic material ancestral
    to the specified number of samples.
    """

    left: float
    right: float
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
    node_time: float = 0

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str((interval.left, interval.right, interval.ancestral_to))
            s += ", "
        if len(self.ancestry) > 0:
            s = s[:-2]
        return s + "]"

    @property
    def hull(self):
        return self.right - self.left

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

    def split(self, breakpoint, t):
        """
        Splits the ancestral material for this lineage at the specified
        breakpoint, and returns a second lineage with the ancestral
        material to the right.
        """
        left_ancestry = []
        right_ancestry = []
        for interval in self.ancestry:
            if interval.right <= breakpoint:
                left_ancestry.append(interval)
            elif interval.left >= breakpoint:
                right_ancestry.append(interval)
            else:
                assert interval.left < breakpoint < interval.right
                left_ancestry.append(dataclasses.replace(interval, right=breakpoint))
                right_ancestry.append(dataclasses.replace(interval, left=breakpoint))
        self.ancestry = left_ancestry
        return Lineage(self.node, right_ancestry, t)


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
    Inverse function of cumulative hazard function.

    T is mean time of nodes to last coalescenc event.
    """
    c = k * (k - 1) / 2
    d = 2 * T * rho
    return (-c - d + math.sqrt((c + d) ** 2 + 4 * x * rho)) / (2 * rho)


def draw_event_time(num_lineages, rho, rng, T=0):
    """
    Given expected coalescence rate (k choose 2) + (2*(t + T)*rho),
    draw single random value t from the non-homogeneous
    exponential distribution
    rho is the total overlap of all combinations expressed
    in recombination rate units.

    Inverse sampling formula for non-homogeneous exponential
    given rate as described above.
    T is the (mean) time to the last coalescence event.
    """
    if rho == 0.0:
        return rng.expovariate(math.comb(num_lineages, 2))
    else:
        # draw random value from exponential with rate 1
        s = rng.expovariate(1)
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
        elif a.ancestry[i].left >= b.ancestry[j].right:
            j += 1
        else:
            left = max(a.ancestry[i].left, b.ancestry[j].left)
            right = min(a.ancestry[i].right, b.ancestry[j].right)
            overlap.append(
                AncestryInterval(
                    left, right, a.ancestry[i].ancestral_to + b.ancestry[j].ancestral_to
                )
            )
            overlap_length += right - left
            if a.ancestry[i].right < b.ancestry[j].right:
                i += 1
            else:
                j += 1
    return (overlap, overlap_length)


def generate_segments(intervals, rho, T, node_times, rng):
    split_intervals = []
    temp = []
    total_branch_length = sum(T - t for t in node_times)

    for interval in intervals:
        expected_num_breakpoints = rho / 2 * interval.span * total_branch_length
        num_breakpoints = rng.poisson(expected_num_breakpoints)
        if num_breakpoints == 0:  # if else clause not needed here
            temp.append(interval)
        else:
            breakpoints = interval.span * rng.random(num_breakpoints) + interval.left
            breakpoints.sort()
            left = interval.left
            for bp in breakpoints:
                right = bp
                assert left < right, "interval edges not strictly ascending."
                temp.append(AncestryInterval(left, right, interval.ancestral_to))
                split_intervals.append(temp)
                temp = []
                left = right
            right = interval.right
            temp.append(AncestryInterval(left, right, interval.ancestral_to))
    split_intervals.append(temp)

    return split_intervals


def pick_segment(intervals, rho, T, node_times, rng):
    all_segments = generate_segments(intervals, rho, T, node_times, rng)
    idx = rng.integers(len(all_segments))
    return all_segments[idx]


def remove_segment(current_lineage, to_remove):
    """
    Yields the AncestralIntervals of current_lineage minus the
    intersection with to_remove AncestralIntervals. The latter
    list of intervals should always be contained within the former.
    """
    n = len(current_lineage.ancestry)
    m = len(to_remove)
    i = j = 0
    left = current_lineage.ancestry[i].left
    while i < n and j < m:
        if current_lineage.ancestry[i].right <= to_remove[j].left:
            yield current_lineage.ancestry[i]
        else:
            left = current_lineage.ancestry[i].left
            while j < m and to_remove[j].right <= current_lineage.ancestry[i].right:
                right = to_remove[j].left
                if left != right:
                    yield AncestryInterval(
                        left, right, current_lineage.ancestry[i].ancestral_to
                    )
                left = to_remove[j].right
                j += 1
            right = current_lineage.ancestry[i].right
            if left != right:
                yield AncestryInterval(
                    left, right, current_lineage.ancestry[i].ancestral_to
                )
        i += 1

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
    """
    Maps a pair of indices to a unique integer.
    """
    return int((sorted_pair[0]) + sorted_pair[1] * (sorted_pair[1] - 1) / 2)


def reverse_combinadic_map(idx, k=2):
    """
    Maps a unique index to a unique pair.
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


def update_total_overlap(ancestry, n):
    total = 0
    for interval in ancestry:
        total += (n - interval.ancestral_to + 1) * interval.span
    return total


def update_total_overlap_brute_force(lineages):
    total = 0
    for a, b in itertools.combinations(lineages, 2):
        _, overlap_length = intersect_lineages(a, b)
        total += overlap_length
    return total


def sample_pairwise_rates(lineages, t, rho, rng):
    pairwise_rates = np.zeros(math.comb(len(lineages), 2), dtype=np.float64)
    for a, b in itertools.combinations(range(len(lineages)), 2):
        _, overlap_length = intersect_lineages(lineages[a], lineages[b])
        overlapping = int(overlap_length > 0)
        rate = (
            1
            + (2 * t - (lineages[a].node_time + lineages[b].node_time))
            * overlap_length
            * rho
            / 2
        ) * overlapping
        pairwise_rates[combinadic_map((a, b))] = rate

        # total_overlap_test += overlap_length
    # assert total_overlap == total_overlap_test, 'total_overlap wrong'

    # draw random pair based on rates
    selected_idx = rng.choices(range(pairwise_rates.size), weights=pairwise_rates)[0]
    a, b = reverse_combinadic_map(selected_idx)
    overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    return a, b, overlap, overlap_length


def sample_pairwise_times(lineages, rng, time_last_event, rho):
    num_lineages = len(lineages)
    pairwise_times = np.zeros(math.comb(num_lineages, 2), dtype=np.float64)
    for a, b in itertools.combinations(range(num_lineages), 2):
        _, overlap_length = intersect_lineages(lineages[a], lineages[b])
        if overlap_length > 0:
            T = 2 * time_last_event - (lineages[a].node_time + lineages[b].node_time)
            new_event_time = draw_event_time(2, rho * overlap_length, rng, T / 2)
            pairwise_times[combinadic_map((a, b))] = new_event_time

    # pick pair with smallest new_event_time
    non_zero_times = np.nonzero(pairwise_times)[0]
    selected_idx = non_zero_times[np.argmin(pairwise_times[non_zero_times])]
    a, b = reverse_combinadic_map(selected_idx)
    overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    return a, b, overlap, overlap_length, pairwise_times[selected_idx]


def sample_rejection(lineages, rng):

    overlap_length = 0
    while overlap_length == 0:
        a, b = rng.sample(range(len(lineages)), k=2)
        overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    return a, b, overlap, overlap_length


def sim_yaca(n, rho, L, seed=None, rejection=True, verbose=False, expectation=True):
    rng = random.Random(seed)
    rng_numpy = np.random.default_rng(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    nodes = []
    total_overlap = L * n * (n - 1) / 2
    t = 0

    for _ in range(n):
        lineages.append(Lineage(len(nodes), [AncestryInterval(0, L, 1)], t))
        nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

    if verbose:
        check_progress(lineages, total_overlap)

    while total_overlap > 0:
        # draw new event time and sample lineages
        # either based on mean time to last event (expectation)
        # or based on pairwise rates
        if expectation:
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
                    lineages, t, rho, rng
                )

        else:
            a, b, overlap, overlap_length, new_event_time = sample_pairwise_times(
                lineages, rng, t, rho
            )
            t += new_event_time

        if verbose:
            print("coalescing lineages:", lineages[a].node, lineages[b].node)
        node_times = (lineages[a].node_time, lineages[b].node_time)
        coalesced_segment = pick_segment(overlap, rho, t, node_times, rng_numpy)
        c = Lineage(len(nodes), coalesced_segment, t)
        for interval in coalesced_segment:
            for lineage in lineages[a], lineages[b]:
                tables.edges.add_row(
                    interval.left, interval.right, c.node, lineage.node
                )

        nodes.append(Node(time=t))

        # remove interval from old lineage
        to_delete = []
        for lineage_idx in a, b:
            updated_ancestry = list(remove_segment(lineages[lineage_idx], c.ancestry))
            lineages[lineage_idx].ancestry = updated_ancestry
            if len(updated_ancestry) == 0:
                to_delete.append(lineage_idx)
        for lineage_idx in sorted(to_delete, reverse=True):
            del lineages[lineage_idx]

        # filter out intervals that are ancestral to all samples
        c.ancestry = [segment for segment in c.ancestry if segment.ancestral_to < n]
        if len(c.ancestry) > 0:
            lineages.append(c)

        # update total_overlap
        total_overlap = update_total_overlap_brute_force(lineages)
        if verbose:
            check_progress(lineages, total_overlap)

    assert total_overlap == 0, "total_overlap less than 0!"
    if len(lineages) > 0:
        print(seed)
    assert len(lineages) == 0, "Not all segments are ancestral to n samples."

    for node in nodes:
        tables.nodes.add_row(flags=node.flags, time=node.time, metadata=node.metadata)
    tables.sort()
    tables.edges.squash()
    return tables.tree_sequence()


def check_progress(lineages, total_overlap):
    print("total_overlap", total_overlap)
    print("num_lineages", len(lineages))
    for lin in lineages:
        print(lin)


def merge_lineages_test(lineages):
    intervals = []
    for lin in lineages:
        intervals += lin.ancestry
    intervals.sort(key=lambda x: x.left)
    current = intervals[0].left
    if current != 0:
        return False
    current = intervals[0].right
    for i in range(1, len(intervals)):
        if intervals[i].left != current:
            return False
        current = intervals[i].right
        i += 1

    return True
