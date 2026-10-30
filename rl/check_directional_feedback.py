import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from directional_feedback import DirectionalPredictor, hand_envelope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    initial = [86, 42, 47, 137]
    assert hand_envelope(initial, initial)
    assert hand_envelope([85, 43, 47, 137], initial)
    assert not hand_envelope([87, 42, 47, 137], initial)
    assert not hand_envelope([86, 41, 47, 137], initial)
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and [run["branch"] for run in report["runs"]] == ["hold", "feedback", "directional"]
        source_path = Path(report["baseline_report"])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["baseline_report_sha256"]
        old = json.loads(source_path.read_text())
        np.testing.assert_array_equal(report["initial_physics_state"], old["initial_physics_state"])
        np.testing.assert_array_equal(report["initial_ctrl"], old["initial_ctrl"])
        assert hashlib.sha256(Path(report["model"]).read_bytes()).hexdigest() == report["model_sha256"]
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        model = mujoco.MjModel.from_xml_path(report["model"])
        for run in report["runs"]:
            assert run["prediction_rollouts"] <= 1530 and len(run["plans"]) <= 30
            assert run["direction_guard_passed"] == bool(run["trace"] and all(hand_envelope(row["loads_n"], run["initial_loads_n"]) for row in run["trace"]))
            if run["branch"] != "directional":
                original = next(row for row in old["runs"] if row["branch"] == run["branch"])
                np.testing.assert_allclose(run["final_loads_n"], original["final_loads_n"], atol=1e-9, rtol=0)
            else:
                assert all(plan["qp_iterations"] <= 100 for plan in run["plans"])
            saved = np.load(run["trajectory"], allow_pickle=False)
            np.testing.assert_array_equal(saved["initial_physics_state"], report["initial_physics_state"])
            np.testing.assert_array_equal(saved["initial_ctrl"], report["initial_ctrl"])
            assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0)), initial=0) <= .010000001
            assert np.all(saved["ctrl"] >= model.actuator_ctrlrange[:, 0]) and np.all(saved["ctrl"] <= model.actuator_ctrlrange[:, 1])
            data = mujoco.MjData(model)
            mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            for index, control in enumerate(saved["ctrl"]):
                data.ctrl[:] = control
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                for substep in range(round(.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
            if run["trace"]:
                assert run["max_penetration_m"] >= max(row["penetration"]["depth_m"] for row in run["trace"])
        state, ctrl = np.array(report["initial_physics_state"]), np.array(report["initial_ctrl"])
        predictor = DirectionalPredictor(model, .25, report["runs"][0]["initial_loads_n"])
        first = predictor.rollout(state, ctrl, ctrl)
        predictor.rollout(state, ctrl, ctrl + .001)
        repeated = predictor.rollout(state, ctrl, ctrl)
        np.testing.assert_array_equal(first[0], repeated[0])
        assert first[1:] == repeated[1:]
        print("Verified three paired branches, unchanged baseline, budgets, hand guard and native replay")
    print("PASS: directional hand-load bounds")


if __name__ == "__main__":
    main()
