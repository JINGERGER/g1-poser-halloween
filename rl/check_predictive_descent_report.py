import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from artifact_utils import digest
from check_predictive_descent import descent_gates
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["passed"] and report["replay_passed"]
    assert not report["full_transition_validated"]
    for name, expected in report["sources"].items():
        assert hashlib.sha256(report["source_snapshots"][name].encode()).hexdigest() == expected
    prefix = report["prefix"]
    assert digest(ROOT / prefix["report"]) == prefix["report_sha256"]
    source = json.loads((ROOT / prefix["report"]).read_text())
    assert source["passed"] and source["requested_sides"] == ["right", "left"]
    assert digest(ROOT / prefix["trajectory"]) == prefix["trajectory_sha256"] == source["trajectory_sha256"]
    assert digest(ROOT / report["trajectory"]) == report["trajectory_sha256"]
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    segment = report["descent"]
    with np.load(ROOT / report["trajectory"]) as saved, np.load(ROOT / prefix["trajectory"]) as original:
        samples, positions, velocities = saved["samples"], saved["qpos"], saved["qvel"]
        penetrations = saved["penetrations"]
        start, end = segment["start_index"], segment["end_index"]
        assert start == prefix["end_index"] == len(original["ctrl"])
        assert end == len(samples) == len(saved["ctrl"]) == len(positions) - 1
        np.testing.assert_array_equal(positions[0], state["qpos"])
        np.testing.assert_array_equal(velocities[0], state["qvel"])
        for key in ("qpos", "qvel", "ctrl", "samples", "penetrations", "phase"):
            np.testing.assert_allclose(saved[key][:len(original[key])], original[key], atol=1e-10, rtol=0.)
        assert np.all(saved["phase"] == 0)
        np.testing.assert_allclose(samples[:, 0], np.arange(1, end + 1) * .002, atol=1e-9, rtol=0.)
        assert abs(segment["start_time_s"] - start * .002) < 1e-9
        assert abs(segment["end_time_s"] - end * .002) < 1e-9
        assert all(node["passed"] for node in segment["nodes"])
        assert all(descent_gates(task, segment, saved).values())
        steps = segment["steps"]
        assert len(steps) * 10 == end - start
        assert sum(record["phase"] == 1. for record in steps) >= 200
        for index, record in enumerate(steps):
            assert abs(record["time_s"] - (start + index * 10) * .002) < 1e-9
            assert record["prediction_velocity_error"] < 1e-10
            assert record["normalized_constraint_residual"] <= 1e-4
            chosen = next(candidate for candidate in record["nonlinear_search"] if candidate["fraction"] == record["selected_fraction"])
            assert chosen["cost"] == min(candidate["cost"] for candidate in record["nonlinear_search"])
        for index, control in enumerate(saved["ctrl"]):
            assert not task.data.xfrc_applied.any() and not task.data.qfrc_applied.any()
            assert np.all(control >= task.model.actuator_ctrlrange[:, 0])
            assert np.all(control <= task.model.actuator_ctrlrange[:, 1])
            task.data.ctrl[:] = control
            mujoco.mj_step(task.model, task.data)
            mujoco.mj_forward(task.model, task.data)
            np.testing.assert_allclose(task.data.qpos, positions[index + 1], atol=1e-10, rtol=0.)
            np.testing.assert_allclose(task.data.qvel, velocities[index + 1], atol=1e-10, rtol=0.)
            measured, gates, penetration = task.measure()
            assert all(gates.values())
            np.testing.assert_allclose(measured, samples[index], atol=1e-9, rtol=0.)
            np.testing.assert_allclose(penetration, penetrations[index], atol=1e-9, rtol=0.)
    print(json.dumps({"passed": True, "duration_s": report["duration_s"],
                      "base_drop_m": segment["actual_base_drop_m"], "com_drop_m": segment["actual_com_drop_m"],
                      "full_transition_validated": False}))


if __name__ == "__main__":
    main()
