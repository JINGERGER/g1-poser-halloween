import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from component_unload_ramps import AMPLITUDES, compare, guard
from load_transfer import physical_metrics, smooth
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    for key in ("source_report", "source_trajectory", "model"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    path = Path(report["model"])
    assert digest(path.with_name("manifest.json")) == report["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    assert report["protocol"]["amplitudes"] == list(AMPLITUDES)
    assert [run["amplitude"] for run in report["runs"]] == list(AMPLITUDES)
    assert report["protocol"]["optimization"] is False and report["protocol"]["restarts"] == 0
    chain = json.loads(Path(report["source_report"]).read_text())
    for unused in range(2):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    np.testing.assert_array_equal(chain["candidate_ctrl"], report["candidate_ctrl"])
    model = mujoco.MjModel.from_xml_path(str(path))
    replayed_steps = 0
    with np.load(report["source_trajectory"], allow_pickle=False) as initial:
        for run in report["runs"]:
            assert digest(run["trajectory"]) == run["trajectory_sha256"]
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, initial["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.ctrl[:] = initial["initial_ctrl"]
            mujoco.mj_forward(model, data)
            row = physical_metrics(model, data)
            assert row == run["initial_metrics"]
            peak = row["penetration"]["depth_m"]
            start_time = float(data.time)
            expected_ctrl = data.ctrl.copy()
            period = round(.02 / model.opt.timestep)
            failure = None
            with np.load(run["trajectory"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["initial_physics_state"], initial["initial_physics_state"])
                np.testing.assert_array_equal(saved["initial_ctrl"], initial["initial_ctrl"])
                np.testing.assert_array_equal(saved["qpos"][0], data.qpos)
                np.testing.assert_array_equal(saved["qvel"][0], data.qvel)
                assert float(saved["dt"]) == model.opt.timestep
                for index, control in enumerate(saved["ctrl"]):
                    if index % period == 0:
                        desired = initial["initial_ctrl"] + run["amplitude"] * smooth(((index // period + 1) * .02) / .6) * (
                            np.array(report["candidate_ctrl"]) - initial["initial_ctrl"])
                        expected_ctrl += np.clip(desired - expected_ctrl, -.01, .01)
                    np.testing.assert_array_equal(control, expected_ctrl)
                    assert np.all(control >= model.actuator_ctrlrange[:, 0]) and np.all(control <= model.actuator_ctrlrange[:, 1])
                    data.ctrl[:] = control
                    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                    mujoco.mj_step(model, data)
                    peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
                    mujoco.mj_forward(model, data)
                    np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
                    assert abs(data.time - start_time - (index + 1) * model.opt.timestep) < 1e-8
                    row = physical_metrics(model, data)
                    np.testing.assert_allclose(row["loads_n"], run["trace"][index]["loads_n"], atol=1e-9, rtol=0)
                    peak = max(peak, row["penetration"]["depth_m"])
                    failure = guard(model, data, row, peak)
                    if failure:
                        assert index == len(saved["ctrl"]) - 1
                assert failure == run["failure"]
                assert run["completed"] == (failure is None and len(saved["ctrl"]) == round(1 / model.opt.timestep))
                replayed_steps += len(saved["ctrl"])
            np.testing.assert_allclose(peak, run["peak_penetration_m"], atol=1e-12, rtol=0)
            for key, value in compare(run, report["runs"][0]).items():
                assert value == run[key]
    print(f"PASS: four paired initial states, exact slew-limited controls, {replayed_steps} native steps, first-violation stopping and paired reduction gates")


if __name__ == "__main__":
    main()
