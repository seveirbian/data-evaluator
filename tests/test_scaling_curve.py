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


import torch
from unittest.mock import patch
from src.scaling_curve_evaluator.scaling_curve import ScalingCurveGenerator


def _make_obs(n: int) -> list[dict]:
    return [{"cam": torch.zeros(3, 4, 4)} for _ in range(n)]


@patch("src.scaling_curve_evaluator.scaling_curve.PolicyEmbeddingExtractor")
@patch("src.scaling_curve_evaluator.scaling_curve.LeRobotDatasetLoader")
def test_generate_returns_correct_structure(MockLoader, MockExtractor):
    # train: 5 episodes, eval: 2 episodes
    MockLoader.return_value.camera_keys = ["cam"]
    MockLoader.return_value.get_initial_observations.side_effect = [
        _make_obs(5),  # train
        _make_obs(2),  # eval
    ]
    MockExtractor.return_value.extract_per_camera.return_value = {
        "cam": torch.zeros(128)
    }

    gen = ScalingCurveGenerator("p", "t", "e", "m", num_points=3)
    results = gen.generate()

    assert isinstance(results, list)
    assert len(results) >= 1
    n, score = results[0]
    assert isinstance(n, int)
    assert isinstance(score, float)
    assert results[-1][0] == 5  # 最后一点必须是 n_total
