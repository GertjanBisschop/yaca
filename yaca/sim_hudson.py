"""
Code adapted from https://github.com/tskit-dev/what-is-an-arg-paper
"""

from __future__ import annotations

import collections
import random
import math
import dataclasses
from typing import List
from typing import Any

import numpy as np
import tskit

from yaca import sim

NODE_IS_RECOMB = 1 << 1


@dataclasses.dataclass
class MappingSegment:
    left: int
    right: int
    value: Any = None


def overlapping_segments(segments):
    """
    Returns an iterator over the (left, right, X) tuples describing the
    distinct overlapping segments in the specified set.
    """
    S = sorted(segments, key=lambda x: x.left)
    n = len(S)
    # Insert a sentinel at the end for convenience.
    S.append(MappingSegment(math.inf, 0))
    right = S[0].left
    X = []
    j = 0
    while j < n:
        # Remove any elements of X with right <= left
        left = right
        X = [x for x in X if x.right > left]
        if len(X) == 0:
            left = S[j].left
        while j < n and S[j].left == left:
            X.append(S[j])
            j += 1
        j -= 1
        right = min(x.right for x in X)
        right = min(right, S[j + 1].left)
        yield left, right, X
        j += 1

    while len(X) > 0:
        left = right
        X = [x for x in X if x.right > left]
        if len(X) > 0:
            right = min(x.right for x in X)
            yield left, right, X


def merge_ancestry(lineages):
    """
    Return an iterator over the ancestral material for the specified lineages.
    For each distinct interval at which ancestral material exists, we return
    the AncestryInterval and the corresponding list of lineages.
    """
    # See note above on the implementation - this could be done more cleanly.
    segments = []
    for lineage in lineages:
        for interval in lineage.ancestry:
            segments.append(
                MappingSegment(interval.left, interval.right, (lineage, interval))
            )

    for left, right, U in overlapping_segments(segments):
        max_ancestral_to = max(u.value[1].ancestral_to for u in U)
        ancestral_to = sum(u.value[1].ancestral_to for u in U)
        if max_ancestral_to < ancestral_to:
            return True
        interval = sim.AncestryInterval(left, right, ancestral_to)
        yield interval, [u.value[0] for u in U]
    return False


def time_to_next_coalescent_event(lineages, rho, t, seed=None):
    rng = random.Random(seed)
    t_inc = 0
    t_re = t_inc
    nodes = []

    # loop until the last event is a coalescence event
    # while t_inc == t_re:
    coalescence = False
    while not coalescence:

        # working with continuous genome!
        lineage_links = [lineage.hull for lineage in lineages]
        total_links = sum(lineage_links)
        re_rate = total_links * rho

        t_re = math.inf if re_rate == 0 else rng.expovariate(re_rate)
        k = len(lineages)
        ca_rate = k * (k - 1) / 2
        t_ca = rng.expovariate(ca_rate)
        t_inc = min(t_re, t_ca)
        t += t_inc

        if t_inc == t_re:
            left_lineage = rng.choices(lineages, weights=lineage_links)[0]
            breakpoint = left_lineage.hull * rng.random() + left_lineage.left
            assert left_lineage.left < breakpoint < left_lineage.right
            right_lineage = left_lineage.split(breakpoint, t)
            lineages.append(right_lineage)
            child = left_lineage.node

            for lineage in left_lineage, right_lineage:
                lineage.node = len(nodes)
                nodes.append(sim.Node(time=t))

        else:
            a = lineages.pop(rng.randrange(len(lineages)))
            b = lineages.pop(rng.randrange(len(lineages)))
            c = sim.Lineage(len(nodes), [], t)

            ma_iter = merge_ancestry([a, b])
            while True:
                try:
                    interval, intersecting_lineages = next(ma_iter)
                    c.ancestry.append(interval)
                except StopIteration as ex:
                    coalescence = ex
                    break

            nodes.append(sim.Node(time=t))
            lineages.append(c)

    return (t, a.node, b.node)
