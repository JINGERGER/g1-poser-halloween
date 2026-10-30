import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from staged_transfer import STAGES, StageProblem, stage_reached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    row = {"loads_n": [20, 20, 65, 20], "gaps_m": [0, 0, -.001, 0], "floor_contacts": [True] * 4}
    assert stage_reached(row, STAGES[0]) and not stage_reached(row, STAGES[1])
    row["loads_n"][2], row["gaps_m"][2], row["floor_contacts"][2] = 0, .02, False
    assert stage_reached(row, STAGES[-1]) and not stage_reached(row, STAGES[-2])
    row["floor_contacts"][2] = True
    assert not stage_reached(row, STAGES[-1])
    matrix = json.loads(args.matrix.read_text())
    entries = [entry for entry in json.loads(Path(matrix["support_report"]).read_text())["runs"] if entry["branch"] == "freeze"]
    origin = next(run["solution"] for run in matrix["runs"] if run["anchor"] == 2 and run["swing"] is None)
    terminal = next(run["solution"] for run in matrix["runs"] if run["anchor"] == 2 and run["swing"] == 2)
    problem = StageProblem(entries[1], origin, origin, terminal, STAGES[0])
    initial = np.array(origin["values"])
    before = problem.evaluate(initial)
    repeated = problem.evaluate(initial)
    np.testing.assert_array_equal(before["ineq"], repeated["ineq"])
    assert np.min(before["ineq"]) < 0
    assert before["metrics"]["loads_n"][2] > 70
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed"
        assert report["matrix_sha256"] == hashlib.sha256(args.matrix.read_bytes()).hexdigest()
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        assert 1 <= len(report["solves"]) <= len(STAGES)
        previous = origin
        accepted = 0
        for index, solve in enumerate(report["solves"]):
            assert solve["stage"] == STAGES[index]
            solution = solve["solution"]
            problem = StageProblem(entries[1], previous, origin, terminal, solve["stage"])
            measured = problem.evaluate(np.array(solution["values"]))
            np.testing.assert_allclose(np.linalg.norm(measured["qacc"][:3]), solution["base_linear_acceleration_mps2"], atol=1e-10)
            np.testing.assert_allclose(np.min(measured["ineq"]), solution["ineq_min_scaled"], atol=1e-10)
            if "optimization_ineq_min_scaled" in solution:
                from tolerance_transfer import acceleration_constraints

                expected_margin = min(np.min(measured["ineq"]), np.min(acceleration_constraints(measured["qacc"])))
                np.testing.assert_allclose(expected_margin, solution["optimization_ineq_min_scaled"], atol=1e-9)
                physical_passed = np.linalg.norm(measured["qacc"][:3]) <= .01 and np.max(np.abs(measured["qacc"][3:])) <= .1 and np.min(measured["ineq"]) >= -1e-5
                assert solution["constraint_passed"] == physical_passed
            assert solution["iterations"] <= 150
            if solution["constraint_passed"]:
                accepted += 1
                previous = solution
            else:
                assert index == len(report["solves"]) - 1
        replay = report["replay"]
        assert len(replay["stages"]) <= accepted + 1
        if accepted < len(STAGES):
            assert not replay["complete_lift_passed"]
        assert replay["max_penetration_m"] >= max(row["penetration"]["depth_m"] for row in replay["trace"])
        saved = np.load(replay["trajectory"], allow_pickle=False)
        np.testing.assert_array_equal(saved["initial_physics_state"], entries[1]["anchor"]["physical_state"])
        np.testing.assert_array_equal(saved["initial_ctrl"], entries[1]["anchor"]["previous_ctrl"])
        assert np.max(np.abs(np.diff(np.vstack((saved["initial_ctrl"], saved["ctrl"])), axis=0))) <= .010000001
        model = problem.model
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
    print("PASS: load/gap gates, immutable cached constraints, stop-on-failure and supplied original-state replay")


if __name__ == "__main__":
    main()
