import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from physical_seed import DT, STEPS, Experiment, controls, offsets, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    knots = np.full((3, 23), .2)
    reference = offsets(knots)
    np.testing.assert_allclose(reference[49], .2)
    np.testing.assert_allclose(reference[-1], 0)
    targets = controls(np.zeros(23), np.zeros((STEPS, 23)), knots, np.tile([-.1, .1], (23, 1)))
    assert np.max(np.abs(targets)) <= .1
    assert np.max(np.abs(np.diff(np.vstack((np.zeros(23), targets)), axis=0))) <= .01000000001
    assert rank({"safe": True, "violation": 0, "task_loss": 100}) < rank({"safe": False, "violation": .01, "task_loss": 0})
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] in ("completed", "budget_limited") and len(report["runs"]) == 4
        assert report["source_sha256"] == hashlib.sha256(Path(__file__).with_name("physical_seed.py").read_bytes()).hexdigest()
        for run in report["runs"]:
            entry = {"model": run["model"], "model_sha256": run["model_sha256"],
                     "anchor": {"physical_state": run["initial_physics_state"], "previous_ctrl": run["initial_ctrl"]}}
            experiment = Experiment(entry)
            saved = np.load(run["trajectory"], allow_pickle=False)
            assert saved["ctrl"].shape == (STEPS, 23) and saved["dt"] == DT
            assert np.max(np.abs(np.diff(np.vstack((run["initial_ctrl"], saved["ctrl"])), axis=0))) <= .01000000001
            summary, trace, qpos, qvel = experiment.rollout(saved["ctrl"], detailed=True)
            assert summary == run["best"]
            np.testing.assert_allclose(qpos, saved["qpos"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(qvel, saved["qvel"], atol=1e-10, rtol=0)
            assert run["evaluations"] <= 97
        for start in (0, 2):
            np.testing.assert_array_equal(report["runs"][start]["initial_physics_state"], report["runs"][start + 1]["initial_physics_state"])
    print("PASS: smooth knots, bounds/rate limit, safety-first ranking; supplied reports replay exactly")


if __name__ == "__main__":
    main()
