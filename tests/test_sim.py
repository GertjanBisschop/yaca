import pytest
import random

import yaca.sim as sim


class TestIntersect:
    def test_intersect_segment(self):
        a = sim.Lineage(
            0,
            [
                sim.AncestryInterval(1, 20, 0),
                sim.AncestryInterval(25, 29, 0),
                sim.AncestryInterval(32, 40, 0),
            ],
        )
        b = sim.Lineage(
            1,
            [
                sim.AncestryInterval(0, 15, 0),
                sim.AncestryInterval(17, 30, 0),
                sim.AncestryInterval(32, 40, 0),
            ],
        )
        exp = [
            sim.AncestryInterval(1, 15, 0),
            sim.AncestryInterval(17, 20, 0),
            sim.AncestryInterval(25, 29, 0),
            sim.AncestryInterval(32, 40, 0),
        ]
        overlap, overlap_length = list(sim.intersect_lineages(a, b))
        assert overlap == exp
        assert overlap_length == sum(el.span for el in exp)

    @pytest.mark.parametrize(
        "breakpoints, exp",
        [
            ([5, 8], [sim.AncestryInterval(5, 6, 0), sim.AncestryInterval(10, 12, 0)]),
            ([5, 9], [sim.AncestryInterval(5, 6, 0), sim.AncestryInterval(10, 13, 0)]),
            ([5, 10], [sim.AncestryInterval(5, 6, 0), sim.AncestryInterval(10, 14, 0)]),
            ([5, 11], [sim.AncestryInterval(5, 6, 0), sim.AncestryInterval(10, 15, 0)]),
            (
                [5, 12],
                [
                    sim.AncestryInterval(5, 6, 0),
                    sim.AncestryInterval(10, 15, 0),
                    sim.AncestryInterval(20, 21, 0),
                ],
            ),
            (
                [0, 12],
                [
                    sim.AncestryInterval(0, 6, 0),
                    sim.AncestryInterval(10, 15, 0),
                    sim.AncestryInterval(20, 21, 0),
                ],
            ),
            ([11, 12], [sim.AncestryInterval(20, 21, 0)]),
            (
                [0, 15],
                [
                    sim.AncestryInterval(0, 6, 0),
                    sim.AncestryInterval(10, 15, 0),
                    sim.AncestryInterval(20, 24, 0),
                ],
            ),
            (
                [0, 31],
                [
                    sim.AncestryInterval(0, 6, 0),
                    sim.AncestryInterval(10, 15, 0),
                    sim.AncestryInterval(20, 40, 0),
                ],
            ),
        ],
    )
    def test_merge_segment(self, breakpoints, exp):
        test_intervals = [
            sim.AncestryInterval(0, 6, 0),
            sim.AncestryInterval(10, 15, 0),
            sim.AncestryInterval(20, 40, 0),
        ]
        result = list(sim.merge_segment(test_intervals, breakpoints))
        assert result == exp

    def test_remove_segment(self):
        lineage = sim.Lineage(
            0,
            [
                sim.AncestryInterval(10, 20, 3),
                sim.AncestryInterval(23, 30, 2),
                sim.AncestryInterval(35, 50, 1),
            ],
            0,
        )
        to_remove = [
            sim.AncestryInterval(11, 15, 4),
            sim.AncestryInterval(17, 20, 3),
            sim.AncestryInterval(23, 30, 1),
        ]
        expected = [
            sim.AncestryInterval(10, 11, 3),
            sim.AncestryInterval(15, 17, 3),
            sim.AncestryInterval(35, 50, 1),
        ]
        test_result = list(sim.remove_segment(lineage, to_remove))
        assert len(test_result) == len(expected)
        for test, exp in zip(test_result, expected):
            assert test == exp

    def test_pick_breakpoints(self):
        rng = random.Random(42)
        T = 10
        rho = 1
        overlap = [
            sim.AncestryInterval(11, 15, 4),
            sim.AncestryInterval(17, 20, 3),
            sim.AncestryInterval(23, 30, 1),
        ]
        total_overlap = sum(e.span for e in overlap)
        node_times = (0, 0)
        breakpoints = sim.pick_breakpoints(
            overlap, total_overlap, rho, T, node_times, rng
        )
        print(breakpoints)
        assert False
