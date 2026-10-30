import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from load_transfer import physical_metrics
from contact_pose_path import ContactPosePath
from local_shift_path import schedule
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert digest(report["geometry_report"]) == report["geometry_report_sha256"]
    assert digest(report["baseline_report"]) == report["baseline_report_sha256"]
    baseline = json.loads(Path(report["baseline_report"]).read_text())
    old_hold = next(run for run in baseline["runs"] if run["branch"] == "hold")
    new_hold = next(run for run in report["runs"] if run["branch"] == "hold")
    with np.load(old_hold["trajectory"]) as old, np.load(new_hold["trajectory"]) as new:
        for key in ("qpos", "qvel", "ctrl"):
            np.testing.assert_allclose(new[key], old[key], atol=1e-9, rtol=0)
    geometry = json.loads(Path(report["geometry_report"]).read_text())
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    source = json.loads(Path(geometry["source_report"]).read_text())
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["model"]) == source["model_sha256"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    planner = ContactPosePath(model, geometry["initial"]["qpos"], geometry["geometric_target"]["shift_xy_m"])
    for branch in ("hold", "shift_then_lift", "joint_shift_lift"):
        assert schedule(branch, 0) == (0., 0.)
        assert schedule(branch, 6) == ((0., 0.) if branch == "hold" else (1., 1.))
    for values in (planner.origin, planner.origin + .0001):
        analytic = planner.jacobian(values)
        numeric = np.zeros_like(analytic)
        for index in range(len(values)):
            offset = np.zeros_like(values)
            offset[index] = 1e-6
            numeric[:, index] = (planner.residual(values + offset) - planner.residual(values - offset)) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-6)
    for path in report["paths"]:
        assert len(path["nodes"]) <= 10
        for node in path["nodes"]:
            assert node["nfev"] <= 100
            shift, lift = schedule(path["branch"], node["time_s"])
            planner.target_sites = planner.sites.copy()
            planner.target_sites[2, 2] += .022 * lift
            planner.target_com = planner.com + planner.shift * shift
            qpos = np.array(node["qpos"])
            values = np.r_[qpos[:3], qpos[7:]]
            analytic = planner.jacobian(values)
            numeric = np.zeros_like(analytic)
            for index in range(len(values)):
                offset = np.zeros_like(values)
                offset[index] = 1e-6
                numeric[:, index] = (planner.residual(values + offset) - planner.residual(values - offset)) / 2e-6
            np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-6)
            assert np.all(values >= planner.lower) and np.all(values <= planner.upper)
            np.testing.assert_array_equal(qpos[3:7], planner.initial[3:7])
            residual = planner.residual(values)
            np.testing.assert_allclose(np.max(np.linalg.norm(residual[:12].reshape(4, 3) * .001, axis=1)), node["site_error_m"], atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(residual[12:14] * .001), node["com_error_m"], atol=1e-12)
            metrics = physical_metrics(model, planner.data)
            np.testing.assert_allclose(metrics["gaps_m"], node["gaps_m"], atol=1e-12)
            current = planner.data.geom_xmat[planner.foot_id].reshape(3, 3)
            angle = float(np.arccos(np.clip((np.trace(planner.rotation.T @ current) - 1) / 2, -1, 1)))
            np.testing.assert_allclose(angle, node["right_foot_orientation_error_rad"], atol=1e-12)
            drift = abs(metrics["gaps_m"][3] - planner.initial_gap)
            np.testing.assert_allclose(drift, node["right_foot_gap_drift_m"], atol=1e-12)
            if node["passed"]:
                assert angle <= .001 and drift <= .0001
            if not node["passed"]:
                assert not all(metrics["floor_contacts"][limb] for limb in (0, 1, 3)) or node["site_error_m"] > .001 or node["com_error_m"] > .001 or node["penetration_m"] > .005 or node["unexpected_penetrations"] or (node["time_s"] == 5 and node["gaps_m"][2] < .015) or angle > .001 or drift > .0001
        if not path["passed"]:
            run = next(run for run in report["runs"] if run["branch"] == path["branch"])
            assert not run["executed"] and run["reason"] == "geometry_gate_failed"
    for run in report["runs"]:
        if not run["executed"]:
            continue
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["initial_physics_state"], source["initial_physics_state"])
            np.testing.assert_array_equal(saved["initial_ctrl"], source["initial_ctrl"])
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0]) and np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            for index, control in enumerate(saved["ctrl"]):
                data.ctrl[:] = control
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                metrics = physical_metrics(model, data)
                peak = max(peak, metrics["penetration"]["depth_m"])
                np.testing.assert_allclose(metrics["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12)
    print("PASS: source hashes, schedule, analytic Jacobian, bounded geometry, contact rejection, paired state, control bounds and native replay")


if __name__ == "__main__":
    main()
