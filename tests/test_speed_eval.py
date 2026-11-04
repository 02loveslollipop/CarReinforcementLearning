import numpy as np

from training.dqn import speed_eval
from training.dqn.trainer import train


def test_seeded_suites_are_disjoint_and_reproducible():
    assert not set(speed_eval.TUNE_SEEDS) & set(speed_eval.HOLDOUT_SEEDS)
    first = speed_eval.build_tracks([1000, 1001])
    second = speed_eval.build_tracks([1000, 1001])
    for (seed_a, track_a), (seed_b, track_b) in zip(first, second):
        assert seed_a == seed_b
        assert np.array_equal(track_a.center, track_b.center)
    assert not np.array_equal(first[0][1].center, first[1][1].center)


def test_speed_report_has_reliability_and_completed_lap_metrics(tmp_path):
    checkpoint = train({"training": {"seed": 4, "replay_capacity": 50, "batch_size": 8,
                                     "learning_starts": 8, "hidden_sizes": [8, 8],
                                     "checkpoint_every": 0}}, tmp_path / "model", 0)
    report = speed_eval.evaluate_suite(checkpoint, seeds=(1000,), suite="smoke")
    summary = report["summary"]
    assert summary["episodes"] == len(speed_eval.START_FRACTIONS)
    assert 0.0 <= summary["lap_completion_rate"] <= 1.0
    assert "completed_lap_speed_mps" in summary
    assert len(report["results"]) == summary["episodes"]
    if summary["completed_laps"]:
        assert summary["completed_lap_speed_mps"] is not None
        assert summary["completed_lap_time_s"] is not None
    comparison = speed_eval.compare_reports(report, report)
    assert comparison["paired_episodes"] == summary["episodes"]
    assert all(value == 0 for value in comparison["summary_delta"].values() if value is not None)
