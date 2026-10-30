import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from artifact_utils import digest
from check_predictive_step_chain import reference_gates, segment_gates
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["passed"]
    assert report["replay_passed"] and not report["full_transition_validated"]
    for name, expected in report["sources"].items():
        assert hashlib.sha256(report["source_snapshots"][name].encode()).hexdigest() == expected
    trajectory = ROOT / report["trajectory"]
    assert digest(trajectory) == report["trajectory_sha256"]
    task = StandTransitionTask()
    initial = task.make_states([0])[0]
    with np.load(trajectory) as saved:
        samples, positions, velocities = saved["samples"], saved["qpos"], saved["qvel"]
        penetrations = saved["penetrations"]
        np.testing.assert_array_equal(positions[0], initial["qpos"])
        np.testing.assert_array_equal(velocities[0], np.zeros(task.model.nv))
        assert len(positions) == len(samples) + 1 == len(saved["ctrl"]) + 1
        assert np.all(saved["phase"] == 0)
        np.testing.assert_allclose(samples[:, 0], np.arange(1, len(samples) + 1) * .002, atol=1e-9, rtol=0)
        assert np.all((samples[:, 3] >= .16) & (samples[:, 3] <= .95))
        assert np.all(samples[:, 12] <= 1.)
        assert np.all(samples[:, 13] < .95)
        assert np.all(samples[:, 14] <= .001)
        assert np.max(saved["penetrations"][:, 0]) <= .005
        assert np.max(saved["penetrations"][:, 1]) <= .001
        assert len(report["segments"]) == len(report["requested_sides"])
        previous_end = 250
        if "prefix" in report:
            prefix = report["prefix"]
            assert digest(ROOT / prefix["report"]) == prefix["report_sha256"]
            assert digest(ROOT / prefix["trajectory"]) == prefix["trajectory_sha256"]
            previous_end = prefix["end_index"]
            with np.load(ROOT / prefix["trajectory"]) as original:
                assert len(original["ctrl"]) == previous_end
                for name in ("qpos", "qvel", "ctrl", "phase", "samples", "penetrations"):
                    np.testing.assert_allclose(saved[name][:len(original[name])], original[name], atol=1e-10, rtol=0.)
        for requested, segment in zip(report["requested_sides"], report["segments"], strict=True):
            assert segment["side"] == ("left", "right").index(requested)
            assert segment["start_index"] == previous_end
            assert segment["passed"] and all(segment["planner_gates"].values())
            assert all(reference_gates(segment["nodes"], segment.get("adaptive_height", False)).values())
            if segment.get("bounded_reference", False):
                poses = np.array([node["qpos"] for node in segment["nodes"]])
                np.testing.assert_allclose(poses[0], positions[previous_end], atol=1e-10, rtol=0.)
                assert np.max(np.abs(np.diff(poses[:, task.addresses], axis=0))) <= .120001
            if segment.get("balance_region", False):
                limits = np.minimum(-task.model.jnt_actfrcrange[task.joints, 0], task.model.jnt_actfrcrange[task.joints, 1])
                assert all(node["balance_region"] for node in segment["nodes"])
                assert all(np.max(np.abs(node["torque_nm"]) / limits) <= .700001 for node in segment["nodes"])
            if segment.get("adaptive_height", False):
                assert all(node["support_margin_m"] >= .005 for node in segment["nodes"])
            np.testing.assert_array_equal(segment["start_qpos"], positions[previous_end])
            np.testing.assert_array_equal(segment["end_qpos"], positions[segment["end_index"]])
            assert abs(segment["start_time_s"] - previous_end * .002) < 1e-9
            assert abs(segment["end_time_s"] - segment["end_index"] * .002) < 1e-9
            assert all(segment_gates(task, segment, samples, positions, velocities).values())
            steps = segment["steps"]
            assert len(steps) * 10 == segment["end_index"] - previous_end
            assert sum(record["phase"] == 1. for record in steps) >= 200
            assert all(record["prediction_velocity_error"] < 1e-10 for record in steps)
            assert all(record["normalized_constraint_residual"] <= 1e-4 for record in steps)
            if report.get("nonlinear_search", False):
                for record in steps:
                    candidates = record["nonlinear_search"]
                    chosen = next(candidate for candidate in candidates if candidate["fraction"] == record["selected_fraction"])
                    assert chosen["cost"] == min(candidate["cost"] for candidate in candidates)
            for event in segment["contact_events"]:
                before = samples[samples[:, 0] <= event["time_s"] + 1e-9]
                assert np.all(before[-21::10, 6 + segment["side"]] > event["required_load_n"])
            confirmation = segment["contact_events"][0]["time_s"]
            assert all(record["phase"] <= .9 for record in steps if record["time_s"] < confirmation - 1e-9)
            previous_end = segment["end_index"]
        assert previous_end == len(samples)
        task.data = mujoco.MjData(task.model)
        task.data.qpos[:], task.data.qvel[:], task.data.ctrl[:] = positions[0], velocities[0], task.target
        mujoco.mj_forward(task.model, task.data)
        for index, control in enumerate(saved["ctrl"]):
            task.data.ctrl[:] = control
            mujoco.mj_step(task.model, task.data)
            mujoco.mj_forward(task.model, task.data)
            np.testing.assert_allclose(task.data.qpos, positions[index + 1], atol=1e-10, rtol=0.)
            np.testing.assert_allclose(task.data.qvel, velocities[index + 1], atol=1e-10, rtol=0.)
            measured, gates, penetration = task.measure()
            assert all(gates.values())
            np.testing.assert_allclose(measured, samples[index], atol=1e-9, rtol=0.)
            np.testing.assert_allclose(penetration, penetrations[index], atol=1e-9, rtol=0.)
    print(json.dumps({"passed": True, "sides": report["requested_sides"],
                      "duration_s": report["duration_s"], "full_transition_validated": False}))


if __name__ == "__main__":
    main()
