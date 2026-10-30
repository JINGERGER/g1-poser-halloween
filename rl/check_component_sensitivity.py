import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from load_transfer import physical_metrics
from model_validity_audit import PROFILES, configure, effective_contacts
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert report["mujoco"] == mujoco.__version__
    assert report["protocol"]["profiles"] == [list(profile) for profile in PROFILES]
    assert report["protocol"]["reoptimization"] is False
    for key in ("source_report", "source_trajectory", "model"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for path, expected in report["mesh_sha256"].items():
        assert digest(path) == expected
    assert [run["profile"] for run in report["runs"]] == [profile[0] for profile in PROFILES]
    with np.load(report["source_trajectory"], allow_pickle=False) as original:
        for profile, run in zip(PROFILES, report["runs"]):
            model = configure(Path(report["model"]), profile)
            assert digest(run["trajectory"]) == run["trajectory_sha256"]
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, original["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = original["initial_ctrl"]
            mujoco.mj_forward(model, data)
            assert effective_contacts(model, data) == run["initial_effective_contacts"]
            initial_com = data.subtree_com[1, :2].copy()
            np.testing.assert_allclose(initial_com, report["initial_com_xy_m"], atol=1e-12, rtol=0)
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            rows, velocities, errors = [], [], []
            with np.load(run["trajectory"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["initial_physics_state"], original["initial_physics_state"])
                np.testing.assert_array_equal(saved["initial_ctrl"], original["initial_ctrl"])
                np.testing.assert_array_equal(saved["ctrl"], original["ctrl"][:len(saved["ctrl"])])
                np.testing.assert_array_equal(saved["qpos"][0], data.qpos)
                np.testing.assert_array_equal(saved["qvel"][0], data.qvel)
                for index, ctrl in enumerate(saved["ctrl"]):
                    data.ctrl[:] = ctrl
                    for substep in range(round(.02 / model.opt.timestep)):
                        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                        mujoco.mj_step(model, data)
                        peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                    mujoco.mj_forward(model, data)
                    np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                    assert abs(data.time - (index + 1) * .02) < 1e-8
                    row = physical_metrics(model, data)
                    np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
                    peak = max(peak, row["penetration"]["depth_m"])
                    rows.append(row)
                    velocities.append(float(max(abs(data.qvel))))
                    errors.append(float(np.linalg.norm(data.subtree_com[1, :2] - initial_com)))
                if profile[0] == "nominal":
                    for key in ("qpos", "qvel", "ctrl"):
                        np.testing.assert_allclose(saved[key], original[key], atol=1e-9, rtol=0)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            safe = bool(len(rows) == len(original["ctrl"]) and peak <= .005 and slips and np.mean(slips) <= .02
                        and all(row["other_total_n"] <= 1 and .16 <= row["base_height_m"] <= .65 for row in rows)
                        and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            held = bool(safe and len(rows) == 150 and all(errors[index] <= .005 and velocities[index] <= .05
                        and all(load > 5 for load in rows[index]["loads_n"]) and all(rows[index]["floor_contacts"])
                        for index in range(max(0, len(rows) - 50), len(rows))))
            assert safe == run["historical_engineering_gate_passed"] and held == run["hold_gate_passed"]
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(errors[-1], run["final_com_error_m"], atol=1e-12, rtol=0)
    print("PASS: frozen initial state/controls, five profiles, independent gates and native replay of all saved trajectories")


if __name__ == "__main__":
    main()
