import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["reference_completed"]
    assert report["route"] == "step" and report["controller"] == "PredictivePositionController"
    assert report["replay_passed"] and not report["full_transition_validated"]
    for name, expected in report["sources"].items():
        assert hashlib.sha256(report["source_snapshots"][name].encode()).hexdigest() == expected
    trajectory = ROOT / report["trajectory"]
    assert digest(trajectory) == report["trajectory_sha256"]
    task = StandTransitionTask()
    initial = task.make_states([0])[0]
    with np.load(trajectory) as saved:
        np.testing.assert_array_equal(saved["qpos"][0], initial["qpos"])
        samples = saved["samples"]
        assert len(samples) > 10000 and np.all(saved["phase"] == 0)
        assert np.all((samples[:, 3] >= .16) & (samples[:, 3] <= .95))
        assert np.all(samples[:, 12] <= 1.)
        assert np.all(samples[:, 13] < .95)
        assert np.all(samples[:, 14] <= .001)
        assert np.max(saved["penetrations"][:, 0]) <= .005
        assert np.max(saved["penetrations"][:, 1]) <= .001
        assert np.max(samples[:, 10]) >= .03
        assert np.all(samples[-500:, 6:8] > 20.)
        assert np.linalg.norm(saved["qvel"][-1][:3]) < .05
        assert np.linalg.norm(saved["qvel"][-1][3:6]) < .1
        actual, reference = mujoco.MjData(task.model), mujoco.MjData(task.model)
        actual.qpos[:] = saved["qpos"][-1]
        reference.qpos[:] = report["online_reference_nodes"][-1]["qpos"]
        mujoco.mj_forward(task.model, actual)
        mujoco.mj_forward(task.model, reference)
        feet = task.supports[2:]
        errors = np.linalg.norm(actual.geom_xpos[feet] - reference.geom_xpos[feet], axis=1)
        assert np.all(errors < .01)
        angles = []
        for geom in feet:
            difference = reference.geom_xmat[geom].reshape(3, 3).T @ actual.geom_xmat[geom].reshape(3, 3)
            angles.append(float(np.arccos(np.clip((np.trace(difference) - 1.) / 2., -1., 1.))))
        assert max(angles) < .05
        assert report["contact_events"]
        for event in report["contact_events"]:
            before = samples[samples[:, 0] <= event["time_s"] + 1e-9]
            assert np.all(before[-21::10, 6] > 5.)
        assert all(step["prediction_velocity_error"] < 1e-10 for step in report["steps"])
        print(json.dumps({"step_passed": True, "duration_s": float(samples[-1, 0]),
                          "foot_position_error_m": errors.tolist(), "foot_angle_error_rad": angles,
                          "full_transition_validated": False}))


if __name__ == "__main__":
    main()
