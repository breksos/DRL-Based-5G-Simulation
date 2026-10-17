import numpy as np
import pytest

from drl5g.metrics import jain_index, per_ue_throughput, scheduling_objective


@pytest.mark.parametrize(
    ("throughputs", "expected"),
    [([5.0, 5.0, 5.0, 5.0], 1.0), ([8.0, 0.0, 0.0, 0.0], 0.25), ([0.0, 0.0], 0.0)],
)
def test_jain_index(throughputs, expected):
    assert jain_index(np.array(throughputs)) == pytest.approx(expected)


def test_per_ue_throughput_counts_rbs():
    capacity = np.array([1.0, 2.0, 3.0])
    tp = per_ue_throughput(np.array([0, 2, 2, 1, 2]), capacity)
    np.testing.assert_allclose(tp, [1.0, 2.0, 9.0])


@pytest.mark.parametrize("bad", [[0, 3], [-1, 0]])
def test_per_ue_throughput_rejects_unknown_ues(bad):
    with pytest.raises(ValueError):
        per_ue_throughput(np.array(bad), np.ones(3))


def test_objective_saturates_throughput_term():
    equal = np.full(4, 25e6)  # 100 Mbit/s total, perfectly fair
    assert scheduling_objective(equal, 0.5, 0.5, throughput_norm_mbps=50.0) == pytest.approx(1.0)
    assert scheduling_objective(equal / 4, 0.5, 0.5, throughput_norm_mbps=50.0) == pytest.approx(
        0.75
    )


def test_objective_of_nothing_is_zero():
    assert scheduling_objective(np.zeros(5), 0.5, 0.5, 50.0) == 0.0


def test_starvation_penalty_counts_unserved_ues():
    tp = np.array([10e6, 10e6, 0.0, 0.0])
    base = scheduling_objective(tp, 0.5, 0.5, 50.0)
    assert scheduling_objective(tp, 0.5, 0.5, 50.0, starvation_penalty=1.0) == pytest.approx(
        base - 0.5
    )
