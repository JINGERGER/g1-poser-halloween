import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from component_initial_state import ContactGeometry
from joint_shift_equilibrium import ShiftEquilibrium
from load_transfer import physical_metrics
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    for key in ("source_report", "model", "nominal_manifest"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    candidate = json.loads(Path(report["source_report"]).read_text())
    model_path = Path(report["model"])
    assert digest(model_path.with_name("manifest.json")) == candidate["manifest_sha256"]
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(model_path.parent / "meshes" / name) == expected
    chain = candidate
    for unused in range(3):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    assert digest(chain["equilibrium_report"]) == chain["equilibrium_report_sha256"]
    old = json.loads(Path(chain["equilibrium_report"]).read_text())["solution"]
    expected_seeds = [json.loads(Path(report["nominal_manifest"]).read_text())["qpos"], old["qpos"]]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert [entry["seed"] for entry in report["seeds"]] == ["old_nominal", "old_equilibrium"]
    replayed = 0
    for entry, seed in zip(report["seeds"], expected_seeds):
        np.testing.assert_array_equal(entry["seed_qpos"], seed)
        geometry = entry["geometry"]
        values = np.array(geometry["values"])
        assert max(abs(values)) <= 1 + 1e-9
        measured = ContactGeometry(model, seed).evaluate(values)
        np.testing.assert_allclose(measured["qpos"], geometry["qpos"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(measured["gaps_m"], geometry["gaps_m"], atol=1e-12, rtol=0)
        assert geometry["passed"] == bool(min(measured["ineq"]) >= -1e-5)
        if not geometry["passed"]:
            assert entry["equilibrium"] is None and entry["hold"] is None
            continue
        ctrl = np.clip(np.array(geometry["qpos"])[7:] + np.array(old["ctrl"]) - np.array(old["qpos"])[7:],
                       model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        equilibrium = entry["equilibrium"]
        values = np.array(equilibrium["values"])
        assert np.max(abs(values[:29])) <= 1 + 1e-9
        assert np.all(values[29:] >= model.actuator_ctrlrange[:, 0]) and np.all(values[29:] <= model.actuator_ctrlrange[:, 1])
        measured = ShiftEquilibrium(model, geometry["qpos"], ctrl).evaluate(values)
        np.testing.assert_allclose(measured["qpos"], equilibrium["qpos"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(measured["ctrl"], equilibrium["ctrl"], atol=1e-12, rtol=0)
        passed = bool(np.linalg.norm(measured["qacc"][:3]) <= .01 and np.max(abs(measured["qacc"][3:])) <= .1
                      and min(measured["geometry"]) >= -1e-5)
        assert passed == equilibrium["passed"]
        if not passed:
            assert entry["hold"] is None
            continue
        hold = entry["hold"]
        assert digest(hold["trajectory"]) == hold["trajectory_sha256"]
        with np.load(hold["trajectory"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["qpos"][0], equilibrium["qpos"])
            assert not saved["qvel"][0].any()
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            target_com = data.subtree_com[1, :2].copy()
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            rows = []
            for index, control in enumerate(saved["ctrl"]):
                assert np.all(control >= model.actuator_ctrlrange[:, 0]) and np.all(control <= model.actuator_ctrlrange[:, 1])
                data.ctrl[:] = control
                for substep in range(round(.02 / model.opt.timestep)):
                    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                assert abs(data.time - (index + 1) * .02) < 1e-8
                row = physical_metrics(model, data)
                row.update(com_error_m=float(np.linalg.norm(data.subtree_com[1, :2] - target_com)), velocity_max=float(max(abs(data.qvel))))
                peak = max(peak, row["penetration"]["depth_m"])
                np.testing.assert_allclose(row["loads_n"], hold["trace"][index]["loads_n"], atol=1e-9, rtol=0)
                rows.append(row)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            safe = bool(peak <= .005 and all(row["other_total_n"] <= 1 and .16 <= row["base_height_m"] <= .65 for row in rows)
                        and slips and np.mean(slips) <= .02 and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            reached = bool(safe and len(rows) == 150 and all(row["com_error_m"] <= .005 and row["velocity_max"] <= .05
                           and all(load > 5 for load in row["loads_n"]) and all(row["floor_contacts"]) for row in rows[-50:]))
            assert hold["physical_safety_passed"] == safe and hold["reached"] == reached
            np.testing.assert_allclose(peak, hold["max_penetration_m"], atol=1e-12, rtol=0)
            replayed += len(rows)
    print(f"PASS: two fixed seeds, geometry/equilibrium gates, failed-seed isolation and {replayed} native motor-hold control frames")


if __name__ == "__main__":
    main()
