from src.scaling_curve.scaling_curve import _compute_steps


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
from src.scaling_curve.scaling_curve import ScalingCurveGenerator


def _make_obs(n: int) -> list[dict]:
    return [{"cam": torch.zeros(3, 4, 4)} for _ in range(n)]


@patch("src.scaling_curve.scaling_curve.PolicyEmbeddingExtractor")
@patch("src.scaling_curve.scaling_curve.LeRobotDatasetLoader")
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


import pytest


def _generator_with_results(results):
    """Bypass __init__ and inject results directly."""
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = results
    return gen


def test_plot_raises_if_generate_not_called():
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = None
    with pytest.raises(RuntimeError, match="generate"):
        gen.plot()


def test_plot_saves_file(tmp_path):
    gen = _generator_with_results([(1, 0.4), (5, 0.7), (10, 1.0)])
    save_path = tmp_path / "subdir" / "curve.png"
    gen.plot(save_path=str(save_path))
    assert save_path.exists()


from src.scaling_curve.scaling_curve import _infer_labels


def test_infer_labels_no_collision():
    curves = [
        {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "m"},
        {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch2", "hook_module": "m"},
    ]
    labels = _infer_labels(curves)
    assert labels == ["batch1", "batch2"]


def test_infer_labels_collision():
    curves = [
        {"policy_dir": "policy/act", "train_data_dir": "data/train/batch1", "hook_module": "m"},
        {"policy_dir": "policy/pi0", "train_data_dir": "data/train/batch1", "hook_module": "m"},
    ]
    labels = _infer_labels(curves)
    assert labels == ["batch1/act", "batch1/pi0"]


from src.scaling_curve.scaling_curve import MultiScalingCurveGenerator


def test_multi_plotter_empty_curves_raises():
    with pytest.raises(ValueError, match="curves"):
        MultiScalingCurveGenerator(eval_data_dir="e", curves=[])


@patch("src.scaling_curve.scaling_curve.ScalingCurveGenerator")
def test_generate_all_calls_each_generator(MockGen):
    MockGen.return_value.generate.return_value = [(1, 0.5), (5, 0.9)]
    plotter = MultiScalingCurveGenerator(
        eval_data_dir="e",
        curves=[
            {"policy_dir": "p1", "train_data_dir": "t1", "hook_module": "m"},
            {"policy_dir": "p2", "train_data_dir": "t2", "hook_module": "m"},
        ],
    )
    plotter.generate_all()
    assert MockGen.return_value.generate.call_count == 2


def _multi_plotter_with_results(results_list, labels):
    """Bypass __init__, inject generators with results directly."""
    plotter = object.__new__(MultiScalingCurveGenerator)
    plotter._labels = labels
    plotter._generators = []
    for results in results_list:
        gen = object.__new__(ScalingCurveGenerator)
        gen._results = results
        plotter._generators.append(gen)
    return plotter


def test_multi_plot_raises_if_generate_not_called():
    plotter = object.__new__(MultiScalingCurveGenerator)
    plotter._labels = ["a"]
    gen = object.__new__(ScalingCurveGenerator)
    gen._results = None
    plotter._generators = [gen]
    with pytest.raises(RuntimeError, match="generate"):
        plotter.plot()


def test_multi_plot_saves_file(tmp_path):
    plotter = _multi_plotter_with_results(
        results_list=[
            [(1, 0.4), (5, 0.7), (10, 1.0)],
            [(1, 0.3), (5, 0.6), (10, 0.9)],
        ],
        labels=["batch1", "batch2"],
    )
    save_path = tmp_path / "multi_curve.png"
    plotter.plot(save_path=str(save_path))
    assert save_path.exists()


from src.scaling_curve._similarity import top_k_train_matches


def test_top_k_matches_orders_by_score_descending():
    # 2 eval episodes × 4 train episodes
    sim = torch.tensor([
        [0.1, 0.9, 0.3, 0.5],   # eval 0: best train = 1, then 3, then 2
        [0.8, 0.2, 0.7, 0.6],   # eval 1: best train = 0, then 2, then 3
    ])
    c_min, c_max = 0.0, 1.0  # 归一化为恒等变换,便于手算
    result = top_k_train_matches(sim, c_min, c_max, k=3)

    assert isinstance(result, list)
    assert [r["eval_id"] for r in result] == [0, 1]

    ev0 = result[0]["top_k"]
    assert [m["train_id"] for m in ev0] == [1, 3, 2]
    assert ev0[0]["score"] == pytest.approx(0.9)
    assert ev0[1]["score"] == pytest.approx(0.5)
    assert ev0[2]["score"] == pytest.approx(0.3)

    ev1 = result[1]["top_k"]
    assert [m["train_id"] for m in ev1] == [0, 2, 3]


def test_top_k_matches_clamps_k_to_n_train():
    sim = torch.tensor([[0.2, 0.5]])  # 1 eval × 2 train
    result = top_k_train_matches(sim, 0.0, 1.0, k=5)
    assert len(result[0]["top_k"]) == 2  # clamp 到 N_train=2
    assert [m["train_id"] for m in result[0]["top_k"]] == [1, 0]


def test_top_k_matches_normalizes_scores():
    sim = torch.tensor([[0.4, 0.6]])  # 1 eval × 2 train
    # 归一化: (raw - 0.4) / (0.6 - 0.4)  → 0.6→1.0, 0.4→0.0
    result = top_k_train_matches(sim, 0.4, 0.6, k=2)
    scores = [m["score"] for m in result[0]["top_k"]]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)


def test_top_k_matches_degenerate_range_returns_one():
    sim = torch.tensor([[0.5, 0.5, 0.5]])  # denom < 1e-8
    result = top_k_train_matches(sim, 0.5, 0.5, k=2)
    for m in result[0]["top_k"]:
        assert m["score"] == 1.0


def test_top_k_matches_clips_below_range_scores_to_zero():
    # c_min=0.5 (per-eval max), but rank-2 raw value 0.2 < c_min → must clip to 0.0
    sim = torch.tensor([[0.5, 0.2]])
    result = top_k_train_matches(sim, 0.5, 1.0, k=2)
    scores = [m["score"] for m in result[0]["top_k"]]
    assert scores[0] == pytest.approx(0.0)  # (0.5-0.5)/0.5 = 0.0
    assert scores[1] == 0.0                  # (0.2-0.5)/0.5 = -0.6 → clipped to 0.0
