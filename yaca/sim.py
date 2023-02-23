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

    def split(self, breakpoint):
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
        return Lineage(self.node, right_ancestry, self.node_time)


@dataclasses.dataclass
class Node:
    time: float
    flags: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)


def inverse_expectation_function(x, rho, k):
    c = k * (k - 1) / 2
    return (-c + math.sqrt(c**2 + 4 * x * rho)) / (2 * rho)


def inverse_expectation_function_extended(x, rho, c, T):
    """
    Inverse function of cumulative hazard function.
    c is the number of lineages that overlap
    T is weighted time of nodes to last coalescence event.
    Integral from 0 to t
    """
    d = T * rho
    t1 = 2 * c + d
    return (-t1 + math.sqrt(t1**2 + 8 * x * rho)) / (2 * rho)


def inverse_expectation_function_from_L(x, rho, c, T, L):
    """
    Inverse function of cumulative hazard function.
    c is the number of lineages that overlap
    T is weighted time of nodes to last coalescence event.
    L is time of last event: integral from L to L+t
    """
    d = (2 * L + T) * rho
    return (-d - 2 * c + math.sqrt((4 * c + d) * d + 4 * c**2 + 8 * x * rho)) / (
        2 * rho
    )


def draw_event_time(num_pairs_overlap, rho, rng, T=0):
    """
    Given expected coalescence rate num_pairs_overlap + (2*t + T) * rho / 2),
    draw single random value t from the non-homogeneous
    exponential distribution
    rho is the total overlap of all combinations expressed
    in recombination rate units.

    Inverse sampling formula for non-homogeneous exponential
    given rate as described above.
    T is the (weighted mean) time to the last coalescence event.
    """
    if rho == 0.0:
        return rng.expovariate(num_pairs_overlap)
    else:
        s = rng.expovariate(1)
        return inverse_expectation_function_extended(s, rho, num_pairs_overlap, T)


def coal_rate(c, rho, t, T, p):
    return (c + (2 * t + T) * rho / 2) * p


def draw_event_time_downsample(c, rho, rng, T=0, start_time=0, jump=0.1, p=1):
    """
    Algorithm to draw the first interevent time for a
    non-homogeneous poisson process. Algorithm adapted
    from Introduction to Probability Models by Sheldon Ross.

    start_time = time between oldest node and last event.
    """
    upper_t_interval = jump + start_time
    sup_rate = coal_rate(c, rho, upper_t_interval, T, p)
    new_time = start_time
    w = rng.expovariate(sup_rate)

    while True:
        if new_time + w < upper_t_interval:
            new_time += w
            u = rng.uniform(0, 1)
            if u < coal_rate(c, rho, new_time, T, p) / sup_rate:
                return new_time
            w = rng.expovariate(sup_rate)
        else:
            adjust_w = w - upper_t_interval + new_time
            new_time = upper_t_interval
            upper_t_interval += jump
            old_sup_rate = sup_rate
            sup_rate = coal_rate(c, rho, upper_t_interval, T, p)
            w = adjust_w * old_sup_rate / sup_rate


def expected_fraction_observed_rec_events(n):
    a_n = 0
    for i in range(1, n):
        a_n += 1 / i
    return 1 - 2 / (3 * a_n) * (1 - 1 / n)


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


def update_total_overlap_brute_force(lineages, last_event):
    total = 0
    pairs_count = 0
    overlap_weighted_node_times = 0
    pairwise_overlap_counter = np.zeros(math.comb(len(lineages), 2), dtype=np.int64)
    for a, b in itertools.combinations(range(len(lineages)), 2):
        _, overlap_length = intersect_lineages(lineages[a], lineages[b])
        total += overlap_length
        if overlap_length > 0:
            pairs_count += 1
            overlap_weighted_node_times += overlap_length * sum(
                last_event - n.node_time for n in (lineages[a], lineages[b])
            )
            pairwise_overlap_counter[a] += 1
            pairwise_overlap_counter[b] += 1
    if total > 0:
        overlap_weighted_node_times /= total
    else:
        overlap_weighted_node_times = math.inf

    return total, overlap_weighted_node_times, pairs_count, pairwise_overlap_counter


def sample_pairwise_rates(lineages, t, rho, rng):
    pairwise_rates = np.zeros(math.comb(len(lineages), 2), dtype=np.float64)
    for a, b in itertools.combinations(range(len(lineages)), 2):
        _, overlap_length = intersect_lineages(lineages[a], lineages[b])
        overlapping = int(overlap_length > 0)
        rate = (
            1
            + (2 * t - sum(n.node_time for n in (lineages[a], lineages[b])))
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


def sample_pairwise_times(lineages, rng, time_last_event, rho, p=1):
    num_lineages = len(lineages)
    pairwise_times = np.zeros(math.comb(num_lineages, 2), dtype=np.float64)

    for idx in range(len(pairwise_times)):
        a, b = list(reverse_combinadic_map(idx))
        _, overlap_length = intersect_lineages(lineages[a], lineages[b])
        if overlap_length > 0:
            node_time_diff = abs(lineages[a].node_time - lineages[b].node_time)
            oldest_node = max(lineages[a].node_time, lineages[b].node_time)
            start_time_exp_process = time_last_event - oldest_node
            new_event_time = draw_event_time_downsample(
                1,
                rho * overlap_length,
                rng,
                T=node_time_diff,
                start_time=start_time_exp_process,
                p=p,
            )
            new_event_time -= start_time_exp_process
            pairwise_times[idx] = new_event_time

    # pick pair with smallest new_event_time
    non_zero_times = np.nonzero(pairwise_times)[0]
    if len(non_zero_times) == 0:
        return (-1, -1), math.inf
    selected_idx = non_zero_times[np.argmin(pairwise_times[non_zero_times])]

    return tuple(reverse_combinadic_map(selected_idx)), pairwise_times[selected_idx]


def sample_rejection(lineages, rng):

    overlap_length = 0
    while overlap_length == 0:
        a, b = rng.sample(range(len(lineages)), k=2)
        overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    return a, b, overlap, overlap_length

def process_lineage_pair_overlap_only(lineages, tables, idxs, parent_node, t, rng_numpy, n, rho):
    a, b = idxs
    overlap, overlap_length = intersect_lineages(lineages[a], lineages[b])

    #if verbose:
    #    print("coalescing lineages:", lineages[a].node, lineages[b].node)
    node_times = (lineages[a].node_time, lineages[b].node_time)
    coalesced_segment = pick_segment(
        overlap, rho, t, node_times, rng_numpy
    )
    c = Lineage(parent_node, coalesced_segment, t)
    for interval in coalesced_segment:
        for lineage in lineages[a], lineages[b]:
            tables.edges.add_row(
                interval.left, interval.right, c.node, lineage.node
            )

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


def sim_yaca(n, rho, L, seed=None, verbose=False, union=True, rec_adj=True):
    rng = random.Random(seed)
    rng_numpy = np.random.default_rng(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    nodes = []
    t = 0

    for _ in range(n):
        lineages.append(Lineage(len(nodes), [AncestryInterval(0, L, 1)], t))
        nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

    if verbose:
        check_progress(lineages)
    
    while not fully_coalesced(lineages, n):
        # draw new event time and sample lineages
        if rec_adj:
            rec_rate_adj = expected_fraction_observed_rec_events(len(lineages))
        else:
            rec_rate_adj = 1
        (a, b), new_event_time = sample_pairwise_times(
            lineages, rng, t, rho * rec_rate_adj
        )
        assert new_event_time < math.inf, "Infinite waiting time until next event"
        t += new_event_time

        parent_node = len(nodes)
        nodes.append(Node(time=t))
        if union:
            process_lineage_pair(lineages, tables, (a, b), parent_node, t, rng_numpy, n, rho * rec_rate_adj)        
        else:
            process_lineage_pair_overlap_only(lineages, tables, (a, b), parent_node, t, rng_numpy, n, rho * rec_rate_adj)
        
        if verbose:
            check_progress(lineages)

    assert len(lineages) == 0, "Not all segments are ancestral to n samples."

    for node in nodes:
        tables.nodes.add_row(flags=node.flags, time=node.time, metadata=node.metadata)
    tables.sort()
    tables.edges.squash()
    return tables.tree_sequence()


@dataclasses.dataclass
class Simulator:
    samples: int
    sequence_length: float
    rho: float
    seed: int = None
    rec_adj: bool = True

    def __post_init__(self):
        self.nodes = []
        self.lineages = []
        self.tables = tskit.TableCollection(self.sequence_length)
        self.tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
        self.rng = random.Random(self.seed)
        self.rng_numpy = np.random.default_rng(self.seed)
        self.time = 0
        self.model = "yaca"

        for _ in range(self.samples):
            self.lineages.append(
                Lineage(
                    len(self.nodes),
                    [AncestryInterval(0, self.sequence_length, 1)],
                    self.time,
                )
            )
            self.nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

    @property
    def num_ancestors(self):
        return len(self.lineages)

    @property
    def ancestors(self):
        return self.lineages

    @property
    def num_nodes(self):
        return len(self.nodes)

    def run(self):
        ret = self._run_until(math.inf)
        self.finalise_tables()
        return self.tables.tree_sequence()

    def _run_until(self, end_time):
        while self.time < end_time:
            if not fully_coalesced(self.lineages, self.samples):
                self._step(end_time)
            else:
                return 1
        return 0

    def _step(self, end_time):
        if self.rec_adj:
            rec_rate_adj = expected_fraction_observed_rec_events(len(self.lineages))
        else:
            rec_rate_adj = 1
        (a, b), new_event_time = sample_pairwise_times(
            self.lineages, self.rng, self.time, self.rho * rec_rate_adj
        )
        assert new_event_time < math.inf, "Infinite waiting time until next event"
        self.time += new_event_time
        if self.time > end_time:
            self.time = end_time
        else:
            overlap, overlap_length = intersect_lineages(
                self.lineages[a], self.lineages[b]
            )

            node_times = (self.lineages[a].node_time, self.lineages[b].node_time)
            coalesced_segment = pick_segment(
                overlap, self.rho * rec_rate_adj, self.time, node_times, self.rng_numpy
            )
            c = Lineage(len(self.nodes), coalesced_segment, self.time)
            for interval in coalesced_segment:
                for lineage in self.lineages[a], self.lineages[b]:
                    self.tables.edges.add_row(
                        interval.left, interval.right, c.node, lineage.node
                    )

            self.nodes.append(Node(time=self.time))

            # remove interval from old lineage
            to_delete = []
            for lineage_idx in a, b:
                updated_ancestry = list(
                    remove_segment(self.lineages[lineage_idx], c.ancestry)
                )
                self.lineages[lineage_idx].ancestry = updated_ancestry
                if len(updated_ancestry) == 0:
                    to_delete.append(lineage_idx)
            for lineage_idx in sorted(to_delete, reverse=True):
                del self.lineages[lineage_idx]

            # filter out intervals that are ancestral to all samples
            c.ancestry = [
                segment for segment in c.ancestry if segment.ancestral_to < self.samples
            ]
            if len(c.ancestry) > 0:
                self.lineages.append(c)

    def finalise_tables(self):
        for node in self.nodes:
            self.tables.nodes.add_row(
                flags=node.flags, time=node.time, metadata=node.metadata
            )
        self.tables.sort()
        self.tables.edges.squash()


def check_progress(lineages):
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


################### recording unary nodes #####################


class Segment:
    """
    Doubly linked list implementation
    """

    def __init__(self):
        self.left = 0
        self.right = math.inf
        self.next = None
        self.prev = None
        self.mark = 0
        self.is_bp = 0
        self.ancestral_to = np.zeros(2, dtype=np.uint64)
        self.internal_bp = False

    @staticmethod
    def show_chain(seg):
        s = ""
        while seg is not None:
            s += (
                f"[{seg.left} ({seg.is_bp}), {seg.right}: {np.sum(seg.ancestral_to)}], "
            )
            seg = seg.next
        return s[:-2]

    def __lt__(self, other):
        return (self.left, self.right) < (other.left, other.right)


class SegmentTracker:
    """
    Keeps track of the absence and presence of particular segments in n overlapping lineages
    """

    def __init__(self, L):
        self.L = L
        self.chain = self._make_segment(0, L, None)
        self.overlap_count = 0

    @property
    def state(self):
        return Segment.show_chain(self.chain)

    def _split(self, seg, bp, is_bp=0):
        right = self._make_segment(bp, seg.right, seg.next, is_bp, seg.ancestral_to)
        if seg.next is not None:
            right.next = seg.next
            seg.next.prev = right
        right.prev = seg
        seg.next = right
        seg.right = bp

    def _make_segment(self, left, right, next_seg, is_bp=0, ancestral_to=None):
        seg = Segment()
        seg.left = left
        seg.right = right
        seg.next = next_seg
        seg.is_bp = is_bp
        if isinstance(ancestral_to, np.ndarray):
            seg.ancestral_to = ancestral_to.copy()
        return seg

    def increment_interval(self, left, right, lin, bp_flag, ancestral_to):
        """
        if breakpoint than breakpoint is left
        """
        curr_interval = self.chain
        # add in ancestral_to
        while left < right:
            if curr_interval.left == left:
                curr_interval.is_bp += bp_flag * 2**lin
                bp_flag = 0
                if curr_interval.right <= right:
                    curr_interval.ancestral_to[lin] = ancestral_to
                    left = curr_interval.right
                    curr_interval = curr_interval.next
                else:
                    self._split(curr_interval, right)
                    curr_interval.ancestral_to[lin] = ancestral_to
                    break
            else:
                if curr_interval.right <= left:  # code changed here from < to <=
                    curr_interval = curr_interval.next
                else:
                    self._split(curr_interval, left, bp_flag * 2**lin)
                    bp_flag = 0
                    curr_interval = curr_interval.next

    def remove_interval(self, seg):
        seg.prev.next = seg.next
        seg.next.prev = seg.prev

    def count_overlapping_segments(self):
        count = 0
        seg = self.chain
        while seg is not None:
            if (seg.is_bp > 0) and (np.all(seg.ancestral_to > 0)):
                count += 1
            seg = seg.next
        self.overlap_count = count + 1


# process a pair of overlapping lineages to generate a coalescence event


def record_edges(segment, child_node, parent_node, tables, bound, idx, reverse=True):
    curr_interval = segment
    if curr_interval is not None:
        bp = curr_interval.is_bp
    else:
        bp = 0
    if reverse:
        curr_interval = curr_interval.prev

    while curr_interval is not None:
        if bp == idx + 1:
            break
        if curr_interval.ancestral_to[idx] > 0:
            tables.edges.add_row(
                curr_interval.left, curr_interval.right, parent_node, child_node
            )
            assert curr_interval.mark == 0, "interval already marked"
            curr_interval.mark = curr_interval.ancestral_to[idx]
            curr_interval.ancestral_to[idx] = 0

        if reverse:
            bp = curr_interval.is_bp
            curr_interval = curr_interval.prev
        else:
            curr_interval = curr_interval.next
            if curr_interval is not None:
                bp = curr_interval.is_bp

    return curr_interval


def record_coalescence(segment, child_nodes, parent_node, tables, n):
    # mark the correct segment as starting with an internal breakpoint
    curr_interval = segment
    overlap_bp = False
    while not overlap_bp:
        sum_ancestral_to = np.sum(curr_interval.ancestral_to)
        overlapping = sum_ancestral_to > curr_interval.ancestral_to[0]
        if sum_ancestral_to < n:
            curr_interval.mark = sum_ancestral_to
        for lin in range(2):
            if curr_interval.ancestral_to[lin] > 0:
                tables.edges.add_row(
                    curr_interval.left,
                    curr_interval.right,
                    parent_node,
                    child_nodes[lin],
                )
                curr_interval.ancestral_to[lin] = 0
        curr_interval = curr_interval.next
        if curr_interval is not None:
            if curr_interval.is_bp > 0:
                if overlapping:
                    overlap_bp = True
                else:
                    curr_interval.internal_bp = True
        else:
            break

    return curr_interval


def collect_marked_segments(segment, collect):
    empty = True
    temp = []
    curr_interval = segment

    while curr_interval is not None:
        if curr_interval.mark > 0:
            if curr_interval.internal_bp:
                if len(temp) > 0:
                    collect.append(temp)
                    temp = []
                    empty = True
            
            sum_ancestral_to = curr_interval.mark
            if not empty:
                if (
                    temp[-1].right == curr_interval.left
                    and temp[-1].ancestral_to == sum_ancestral_to
                ):
                    temp[-1].right = curr_interval.right
                else:
                    temp.append(
                        AncestryInterval(
                            curr_interval.left, curr_interval.right, sum_ancestral_to
                        )
                    )
                    empty = False
            else:
                temp.append(
                    AncestryInterval(
                        curr_interval.left, curr_interval.right, sum_ancestral_to
                    )
                )
                empty = False
        curr_interval = curr_interval.next
    
    if len(temp) > 0:
        collect.append(temp)

    return empty


def collect_ancestral_segments(segment, collect, bound, idx):
    empty = len(collect) == 0
    curr_interval = segment

    while curr_interval is not None and curr_interval.left < bound:
        if curr_interval.ancestral_to[idx] > 0:
            if not empty:
                if (
                    collect[-1].right == curr_interval.left
                    and collect[-1].ancestral_to == curr_interval.ancestral_to[idx]
                ):
                    collect[-1].right = curr_interval.right
                else:
                    collect.append(
                        AncestryInterval(
                            curr_interval.left,
                            curr_interval.right,
                            curr_interval.ancestral_to[idx],
                        )
                    )
                    empty = False
            else:
                collect.append(
                    AncestryInterval(
                        curr_interval.left,
                        curr_interval.right,
                        curr_interval.ancestral_to[idx],
                    )
                )
                empty = False
        curr_interval = curr_interval.next

    return empty


def generate_breakpoints(interval, rho, t, node_time, rng):
    total_branch_length = t - node_time
    expected_num_breakpoints = rho / 2 * interval.span * total_branch_length
    num_breakpoints = rng.poisson(expected_num_breakpoints)
    if num_breakpoints == 0:
        return []
    else:
        breakpoints = interval.span * rng.random(num_breakpoints) + interval.left
        breakpoints.sort()

    return breakpoints


def init_segment_tracker(lineages, rng, rho, t, idxs):
    # given two overlapping lineages with idx a and b
    L = max(lineages[i].right for i in idxs)
    l = min(lineages[i].left for i in idxs)
    S = SegmentTracker(L)
    S.chain.left = l

    for i, lin_idx in enumerate(idxs):
        for segment in lineages[lin_idx].ancestry:
            left = segment.left
            bp_flag = 0
            for breakpoint in generate_breakpoints(
                segment, rho, t, lineages[lin_idx].node_time, rng
            ):
                right = breakpoint
                S.increment_interval(left, right, i, bp_flag, segment.ancestral_to)
                left = right
                bp_flag = 1
            right = segment.right
            S.increment_interval(left, right, i, bp_flag, segment.ancestral_to)

    S.count_overlapping_segments()

    return S


def process_lineage_pair(lineages, tables, idxs, parent_node, t, rng, n, rho, picked_idx=-1):
    S = init_segment_tracker(lineages, rng, rho, t, idxs)
    if picked_idx == -1:
        picked_idx = rng.integers(S.overlap_count)
    # go to picked segment
    interval_count = 0
    curr_interval = S.chain
    # move to first overlapping interval
    while np.any(curr_interval.ancestral_to == 0):
        curr_interval = curr_interval.next
    # move to overlapping interval picked to coalesce
    while interval_count < picked_idx:
        if curr_interval.next.is_bp > 0 and np.all(curr_interval.ancestral_to > 0):
            interval_count += 1
        curr_interval = curr_interval.next
    child_nodes = [lineages[idx].node for idx in idxs]
    # register edges on the left of the segment and update ancestral_to
    for i, lin in enumerate(idxs):
        bound = lineages[lin].left
        record_edges(
            curr_interval, child_nodes[i], parent_node, tables, bound, i, reverse=True
        )
    # register coalescing segment and make new lineage and update ancestral_to
    curr_interval = record_coalescence(
        curr_interval, child_nodes, parent_node, tables, n
    )
    for i, lin in enumerate(idxs):
        bound = lineages[lin].right
        record_edges(
            curr_interval, child_nodes[i], parent_node, tables, bound, i, reverse=False
        )

    new_lineages = []
    collect_marked_segments(S.chain, new_lineages)
    # update the ancestry of lineages in idxs
    delete_stack = []
    for idx, lin in enumerate(idxs):
        collect = []
        to_delete = collect_ancestral_segments(
            S.chain, collect, lineages[lin].right, idx
        )
        lineages[lin].ancestry = collect
        if to_delete:
            delete_stack.append(lin)
    for idx in sorted(delete_stack, reverse=True):
        del lineages[idx]
    if len(new_lineages) > 0:
        for lineage in new_lineages:
            lineages.append(Lineage(parent_node, lineage, t))
