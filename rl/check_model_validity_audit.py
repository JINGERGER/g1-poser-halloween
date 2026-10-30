import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from load_transfer import physical_metrics
from model_validity_audit import PROFILES, configure, effective_contacts, inventory, model_diff
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    for key in ("model", "source_xml", "equilibrium_report"):
        assert digest(report[key]) == report[key + "_sha256"]
    for path, expected in report["mesh_sha256"].items():
        assert digest(path) == expected
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(Path(report["model"]).with_name("manifest.json")) == report["manifest_sha256"]
    equilibrium = json.loads(Path(report["equilibrium_report"]).read_text())
    nominal = mujoco.MjModel.from_xml_path(report["model"])
    reference = mujoco.MjModel.from_xml_path(report["source_xml"])
    assert inventory(nominal, equilibrium["solution"]["qpos"]) == report["inventory"]
    assert model_diff(reference, nominal) == report["source_snapshot_differences"]
    assert all(body["inertia_sanity"] for body in report["inventory"]["bodies"])
    assert len(report["runs"]) == 10
    for run in report["runs"]:
        assert digest(run["source_trajectory"]) == run["source_trajectory_sha256"]
        profile = next(profile for profile in PROFILES if profile[0] == run["profile"])
        model = configure(report["model"], profile)
        for field in ("body_mass", "body_inertia", "jnt_range", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "geom_size", "geom_friction", "geom_solimp"):
            np.testing.assert_array_equal(getattr(model, field), getattr(nominal, field))
        np.testing.assert_array_equal(model.geom_solref[:, 0], nominal.geom_solref[:, 0] * profile[3])
        with np.load(run["trajectory"], allow_pickle=False) as saved, np.load(run["source_trajectory"], allow_pickle=False) as original:
            np.testing.assert_array_equal(saved["initial_physics_state"], original["initial_physics_state"])
            np.testing.assert_array_equal(saved["initial_ctrl"], original["initial_ctrl"])
            np.testing.assert_array_equal(saved["ctrl"], original["ctrl"][:len(saved["ctrl"])])
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = saved["initial_ctrl"]
            mujoco.mj_forward(model, data)
            assert effective_contacts(model, data) == run["initial_effective_contacts"]
            peak = physical_metrics(model, data)["penetration"]["depth_m"]
            rows = []
            for index, ctrl in enumerate(saved["ctrl"]):
                data.ctrl[:] = ctrl
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                row = physical_metrics(model, data)
                peak = max(peak, row["penetration"]["depth_m"])
                np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
                rows.append(row)
            np.testing.assert_allclose(peak, run["max_penetration_m"], atol=1e-12)
            slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
            passed = bool(len(rows) == len(original["ctrl"]) and peak <= .005 and slips and np.mean(slips) <= .02
                          and all(row["other_total_n"] <= 1 and .16 <= row["base_height_m"] <= .65 for row in rows)
                          and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
            assert run["historical_engineering_gate_passed"] == passed
            if profile[0] == "nominal":
                np.testing.assert_allclose(saved["qpos"], original["qpos"], atol=1e-9, rtol=0)
                np.testing.assert_allclose(saved["qvel"], original["qvel"], atol=1e-9, rtol=0)
    print("PASS: model/mesh/source hashes, compiled inventory and differences, inertia sanity, isolated profiles, identical controls, all ten native replays and historical gate verdicts")


if __name__ == "__main__":
    main()
