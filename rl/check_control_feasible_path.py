import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from control_feasible_path import ControlFeasiblePath
from local_shift_path import schedule
from load_transfer import physical_metrics
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert len(report["paths"]) == 4
    assert digest(report["geometry_comparison"]) == report["geometry_comparison_sha256"]
    assert digest(report["baseline_report"]) == report["baseline_report_sha256"]
    baseline = json.loads(Path(report["baseline_report"]).read_text())
    assert digest(baseline["geometry_report"]) == baseline["geometry_report_sha256"]
    geometry = json.loads(Path(baseline["geometry_report"]).read_text())
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    source = json.loads(Path(geometry["source_report"]).read_text())
    assert digest(source["model"]) == source["model_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    model = mujoco.MjModel.from_xml_path(source["model"])
    bias = np.array(source["initial_ctrl"]) - np.array(geometry["initial"]["qpos"])[7:]
    violations = []
    for path in report["paths"]:
        planner = ControlFeasiblePath(model, geometry["initial"]["qpos"], geometry["geometric_target"]["shift_xy_m"], path["rotating"], source["initial_ctrl"])
        assert len(path["nodes"]) <= 10
        for node in path["nodes"]:
            assert node["nfev"] <= 100
            values = np.array(node["values"])
            assert np.all(values >= planner.lower) and np.all(values <= planner.upper)
            if path["rotating"]:
                assert np.linalg.norm(values[3:6]) <= .15 + 1e-10
            measured = planner.inspect(values, node["time_s"], path["branch"])
            assert measured["passed"] == node["passed"]
            assert measured["stance"] == ([0, 1, 2, 3] if schedule(path["branch"], node["time_s"])[1] == 0 else [0, 1, 3])
            np.testing.assert_allclose(measured["qpos"], node["qpos"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(planner.contact_values()[1], measured["gaps_m"], atol=1e-10, rtol=0)
            analytic = planner.jacobian(values)
            numeric = np.zeros_like(analytic)
            for column in range(len(values)):
                offset = np.zeros_like(values)
                offset[column] = 1e-6
                numeric[:, column] = (planner.residual(values + offset) - planner.residual(values - offset)) / 2e-6
            np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-5)
        assert path["passed"] == (len(path["nodes"]) == 10 and all(node["passed"] for node in path["nodes"]))
        name = f'{path["branch"]}_rotation{int(path["rotating"])}'
        run = next(run for run in report["runs"] if run["branch"] == name)
        targets = np.array([node["qpos"][7:] for node in path["nodes"]]) + bias
        assert np.all(targets >= model.actuator_ctrlrange[:, 0]) and np.all(targets <= model.actuator_ctrlrange[:, 1])
        excess = np.maximum(model.actuator_ctrlrange[:, 0] - targets, targets - model.actuator_ctrlrange[:, 1])
        if run.get("reason") == "biased_targets_exceed_ctrlrange":
            assert path["passed"] and not run["executed"] and np.any(excess > 0)
            entries = []
            for joint in np.flatnonzero(np.max(excess, axis=0) > 0):
                entries.append({"joint": model.joint(int(model.actuator_trnid[joint, 0])).name,
                                "max_excess_rad": float(np.max(excess[:, joint]))})
            violations.append({"branch": name, "violations": entries})
        elif not path["passed"]:
            assert not run["executed"]
    for run in report["runs"]:
        if not run["executed"]:
            continue
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["initial_physics_state"], source["initial_physics_state"])
            np.testing.assert_array_equal(saved["initial_ctrl"], source["initial_ctrl"])
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0]) and np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            rows = []
            for index, control in enumerate(saved["ctrl"]):
                data.ctrl[:] = control
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                row = physical_metrics(model, data)
                row["elapsed_s"] = (index + 1) * .02
                rows.append(row)
                peak = max(peak, row["penetration"]["depth_m"])
                np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12, rtol=0)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            safe = bool(len(rows) == 300 and peak <= .005 and all(row["other_total_n"] <= 1 and .16 <= row["base_height_m"] <= .65 for row in rows)
                        and slips and np.mean(slips) <= .02 and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            assert run["physical_safety_passed"] == safe
            supported = all(all(row["loads_n"][limb] > 5 and row["floor_contacts"][limb] for limb in (0, 1, 3)) for row in rows[-50:])
            unloaded = bool(safe and supported and all(row["loads_n"][2] <= 5 for row in rows[-50:]))
            assert run["unload_passed"] == unloaded
            phase = True
            if run["branch"] != "hold":
                branch = run["branch"].rsplit("_rotation", 1)[0]
                phase = all(all(row["floor_contacts"][limb] for limb in ((0, 1, 2, 3) if schedule(branch, row["elapsed_s"])[1] == 0 else (0, 1, 3))) for row in rows)
                assert run["contact_phase_passed"] == phase
            lifted = bool(unloaded and phase and all(row["loads_n"][2] <= 1 and row["gaps_m"][2] >= .015 and not row["floor_contacts"][2] for row in rows[-50:]))
            assert run["lift_passed"] == lifted
            if run["branch"] == "hold":
                old = next(old for old in baseline["runs"] if old["branch"] == "hold")
                with np.load(old["trajectory"], allow_pickle=False) as original:
                    for key in ("ctrl", "qpos", "qvel"):
                        np.testing.assert_allclose(saved[key], original[key], atol=1e-9, rtol=0)
    print(json.dumps(violations, indent=2))
    print("PASS: source chain, contact modes, analytic gaps and Jacobians at every node, budgets, limits, rejected controls and native replay")


if __name__ == "__main__":
    main()
