import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from native_contact_seed import NativeProblem
from support_order_matrix import supported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    loaded = {"loads_n": [20.] * 4, "floor_contacts": [True] * 4, "gaps_m": [0.] * 4}
    assert supported(loaded, None)
    for swing in range(4):
        assert not supported(loaded, swing)
        row = {name: values.copy() for name, values in loaded.items()}
        row["loads_n"][swing], row["floor_contacts"][swing], row["gaps_m"][swing] = 0, False, .02
        assert supported(row, swing) and not supported(row, None)
        for other in range(4):
            if other != swing:
                assert not supported(row, other)
        row["gaps_m"][swing] = .0149
        assert not supported(row, swing)
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and len(report["runs"]) == 10
        assert report["mujoco"] == mujoco.__version__ == "3.8.0"
        support_path = Path(report["support_report"])
        assert hashlib.sha256(support_path.read_bytes()).hexdigest() == report["support_report_sha256"]
        entries = [entry for entry in json.loads(support_path.read_text())["runs"] if entry["branch"] == "freeze"]
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        assert [(run["anchor"], run["swing"]) for run in report["runs"]] == [(anchor, swing) for anchor in (1, 2) for swing in (None, 0, 1, 2, 3)]
        replay_count = 0
        for run in report["runs"]:
            entry = entries[run["anchor"] - 1]
            assert run["anchor_state_sha256"] == entry["anchor"]["paired_state_sha256"]
            assert run["relaxed"]["iterations"] <= 150 and run["solution"]["iterations"] <= 150
            problem = NativeProblem(entry, run["swing"], run["relaxed"])
            solution = run["solution"]
            measured = problem.evaluate(np.array(solution["values"]))
            np.testing.assert_allclose(measured["qpos"], solution["qpos"], atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(measured["qacc"][:3]), solution["base_linear_acceleration_mps2"], atol=1e-10)
            np.testing.assert_allclose(np.max(np.abs(measured["qacc"][3:])), solution["max_angular_joint_acceleration_radps2"], atol=1e-9)
            np.testing.assert_allclose(np.min(measured["ineq"]), solution["ineq_min_scaled"], atol=1e-10)
            expected = np.linalg.norm(measured["qacc"][:3]) <= .01 and np.max(np.abs(measured["qacc"][3:])) <= .1 and np.min(measured["ineq"]) >= -1e-5
            assert solution["constraint_passed"] == expected
            assert bool(run["replays"]) == expected
            if len(run["replays"]) == 2:
                assert run["replays"][0]["passed"]
            for replay in run["replays"]:
                replay_count += 1
                settled = [row for row in replay["trace"] if row["elapsed_s"] > (1 if replay["branch"] == "hold" else 4)]
                assert replay["sustained_support"] == bool(settled and all(supported(row, run["swing"]) for row in settled))
                assert replay["passed"] == (replay["safe"] and replay["sustained_support"])
                assert replay["max_penetration_m"] >= max(row["penetration"]["depth_m"] for row in replay["trace"])
                saved = np.load(replay["trajectory"], allow_pickle=False)
                if replay["branch"] == "transfer":
                    np.testing.assert_array_equal(saved["initial_physics_state"], entry["anchor"]["physical_state"])
                    np.testing.assert_array_equal(saved["initial_ctrl"], entry["anchor"]["previous_ctrl"])
                assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0))) <= .010000001
                data = mujoco.MjData(problem.model)
                mujoco.mj_setState(problem.model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                for index, control in enumerate(saved["ctrl"]):
                    data.ctrl[:] = control
                    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                    for substep in range(round(.02 / problem.model.opt.timestep)):
                        mujoco.mj_step(problem.model, data)
                    mujoco.mj_forward(problem.model, data)
                    np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
        print(f"Verified 10 cells and {replay_count} native replays")
    print("PASS: limb-independent support/air predicates and supplied matrix integrity")


if __name__ == "__main__":
    main()
