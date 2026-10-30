import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from native_contact_seed import NativeProblem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("support_report", type=Path)
    parser.add_argument("contact_report", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    support = json.loads(args.support_report.read_text())
    previous = json.loads(args.contact_report.read_text())
    entries = [entry for entry in support["runs"] if entry["branch"] == "freeze"]
    old = previous["runs"][0]
    problem = NativeProblem(entries[0], old["swing"], old["solution"])
    initial = problem.initial.copy()
    first = problem.evaluate(initial)
    shifted = initial.copy()
    shifted[29] += .01
    problem.evaluate(shifted)
    repeated = problem.evaluate(initial)
    np.testing.assert_array_equal(first["qacc"], repeated["qacc"])
    np.testing.assert_array_equal(first["ineq"], repeated["ineq"])
    assert not problem.data.qvel.any() and not problem.data.xfrc_applied.any() and not problem.data.qfrc_applied.any()
    fresh = mujoco.MjData(problem.model)
    fresh.qpos[:] = first["qpos"]
    fresh.ctrl[:] = first["ctrl"]
    mujoco.mj_forward(problem.model, fresh)
    np.testing.assert_array_equal(first["qacc"], fresh.qacc)
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and len(report["runs"]) == 4
        for name, digest in report["source_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
        for run, old in zip(report["runs"], previous["runs"]):
            assert (run["anchor"], run["swing"]) == (old["anchor"], old["swing"])
            problem = NativeProblem(entries[run["anchor"] - 1], run["swing"], old["solution"])
            solution = run["solution"]
            measured = problem.evaluate(np.array(solution["values"]))
            np.testing.assert_allclose(measured["qpos"], solution["qpos"], atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(measured["qacc"][:3]), solution["base_linear_acceleration_mps2"], atol=1e-10)
            np.testing.assert_allclose(np.min(measured["ineq"]), solution["ineq_min_scaled"], atol=1e-10)
            assert solution["iterations"] <= 150
            if solution["constraint_passed"]:
                assert solution["base_linear_acceleration_mps2"] <= .01 and solution["max_angular_joint_acceleration_radps2"] <= .1
                assert solution["ineq_min_scaled"] >= -1e-5
            if len(run["replays"]) == 2:
                assert run["replays"][0]["passed"]
            for replay in run["replays"]:
                assert replay["max_penetration_m"] >= max(row["penetration"]["depth_m"] for row in replay["trace"])
                saved = np.load(replay["trajectory"], allow_pickle=False)
                data = mujoco.MjData(problem.model)
                mujoco.mj_setState(problem.model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                assert np.max(np.abs(np.diff(saved["ctrl"], axis=0)), initial=0) <= .010000001
                for index, control in enumerate(saved["ctrl"]):
                    data.ctrl[:] = control
                    for substep in range(round(.02 / problem.model.opt.timestep)):
                        mujoco.mj_step(problem.model, data)
                    mujoco.mj_forward(problem.model, data)
                    np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-9, rtol=0)
    print("PASS: fresh native forward dynamics, warmstart isolation, reported constraints and supplied physical replays")


if __name__ == "__main__":
    main()
