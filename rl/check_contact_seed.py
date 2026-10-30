import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from contact_seed import ContactProblem, constraints_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("support_report", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    assert constraints_passed(np.zeros(6), np.zeros(4))
    assert not constraints_passed(np.full(6, .01), np.zeros(4))
    assert not constraints_passed(np.zeros(6), np.full(4, -.01))
    assert not constraints_passed(np.full(6, np.nan), np.zeros(4))
    source = json.loads(args.support_report.read_text())
    entries = [entry for entry in source["runs"] if entry["branch"] == "freeze"]
    problem = ContactProblem(entries[0], 3)
    values = np.zeros(29 + len(problem.points) * 3)
    initial = problem.evaluate(values)
    np.testing.assert_allclose(initial["eq"][:3], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(initial["qpos"], problem.nominal, atol=1e-12)
    values[31::3] = 1 / len(problem.points)
    loaded = problem.evaluate(values)
    np.testing.assert_allclose(loaded["eq"][:3], [0, 0, 0], atol=1e-9)
    assert set(point[4] for point in problem.points) == {0, 1, 2}
    assert np.min(loaded["ineq"]) < 0
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and len(report["runs"]) == 4
        assert report["mujoco"] == mujoco.__version__
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        for run in report["runs"]:
            problem = ContactProblem(entries[run["anchor"] - 1], run["swing"])
            solution = run["solution"]
            measured = problem.evaluate(np.array(solution["values"]))
            np.testing.assert_allclose(measured["qpos"], solution["qpos"], atol=1e-10)
            np.testing.assert_allclose(measured["ctrl"], solution["ctrl"], atol=1e-9)
            np.testing.assert_allclose(np.max(np.abs(measured["eq"])), solution["eq_max_scaled"], atol=1e-10)
            np.testing.assert_allclose(np.min(measured["ineq"]), solution["ineq_min_scaled"], atol=1e-10)
            if solution["constraint_passed"]:
                assert solution["eq_max_scaled"] <= 1e-5 and solution["ineq_min_scaled"] >= -1e-5
                assert solution["servo_reconstruction_error_nm"] < 1e-8
            for replay in run["replays"]:
                assert replay["max_penetration_m"] >= max(row["penetration"]["depth_m"] for row in replay["trace"])
                saved = np.load(replay["trajectory"], allow_pickle=False)
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
    print("PASS: gravity/force sign, contact selection, constraint reconstruction and supplied native replays")


if __name__ == "__main__":
    main()
