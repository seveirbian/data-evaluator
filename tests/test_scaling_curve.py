from src.scaling_curve_evaluator.scaling_curve import _compute_steps


def test_compute_steps_bounds():
    steps = _compute_steps(n_total=30, num_points=10)
    assert steps[0] == 1
    assert steps[-1] == 30
    assert all(isinstance(s, int) for s in steps)
    assert len(steps) <= 10


def test_compute_steps_deduplication():
    # 数据量少时去重后应不超过 n_total 个点
    steps = _compute_steps(n_total=3, num_points=20)
    assert steps == [1, 2, 3]
