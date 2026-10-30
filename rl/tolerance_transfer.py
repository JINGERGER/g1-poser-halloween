import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from prepare import ROOT
from staged_transfer import STAGES, StageProblem, replay_path


def acceleration_constraints(qacc):
    limits = np.r_[[.005 / np.sqrt(3)] * 3, [.05] * 26]
    return np.r_[1 - qacc / limits, 1 + qacc / limits]


def rank_diagnostic(problem):
    values = problem.initial.copy()
    diagnostics = []
    for epsilon in (1e-6, 1e-5, 1e-4):
        columns, load_gradient = [], []
        for column in range(len(values)):
            offset = np.zeros_like(values)
            offset[column] = epsilon
            plus, minus = problem.evaluate(values + offset), problem.evaluate(values - offset)
            columns.append((plus["eq"] - minus["eq"]) / (2 * epsilon))
            load_gradient.append((plus["metrics"]["loads_n"][2] - minus["metrics"]["loads_n"][2]) / (2 * epsilon))
        jacobian = np.array(columns).T
        singular = np.linalg.svd(jacobian, compute_uv=False)
        gradient = np.array(load_gradient)
        tangent = gradient - jacobian.T @ np.linalg.lstsq(jacobian.T, gradient, rcond=1e-6)[0]
        diagnostics.append({"epsilon": epsilon, "singular_values": singular.tolist(), "relative_rank_1e_6": int(sum(singular > singular[0] * 1e-6)),
                            "load_gradient_norm": float(np.linalg.norm(gradient)), "tangent_load_gradient_norm": float(np.linalg.norm(tangent))})
    return diagnostics


class ToleranceStageProblem(StageProblem):
    def solve(self):
        bounds = [(-1, 1)] * 29 + list(map(tuple, self.model.actuator_ctrlrange))
        initial = np.clip(self.initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        self.deadline = time.monotonic() + 90
        last = initial.copy()
        history = []

        def inequality(values):
            measured = self.evaluate(values)
            return np.r_[measured["ineq"], acceleration_constraints(measured["qacc"])]

        def callback(values):
            nonlocal last
            last = values.copy()
            measured = self.evaluate(values)
            history.append({"acceleration_norm": float(np.linalg.norm(measured["qacc"][:3])), "ineq_min_scaled": float(np.min(inequality(values)))})

        try:
            result = minimize(lambda values: self.evaluate(values)["objective"], initial, method="SLSQP", bounds=bounds,
                              constraints=[{"type": "ineq", "fun": inequality}], callback=callback,
                              options={"maxiter": 150, "ftol": 1e-8, "eps": 1e-6})
            values, success, message = result.x, bool(result.success), str(result.message)
        except TimeoutError as error:
            values, success, message = last, False, str(error)
        self.deadline = float("inf")
        measured = self.evaluate(values)
        linear = float(np.linalg.norm(measured["qacc"][:3]))
        angular_joint = float(np.max(np.abs(measured["qacc"][3:])))
        minimum = float(np.min(measured["ineq"]))
        optimization_minimum = float(np.min(inequality(values)))
        passed = linear <= .01 and angular_joint <= .1 and minimum >= -1e-5
        return {"solver_success": success, "message": message, "constraint_passed": bool(passed),
                "base_linear_acceleration_mps2": linear, "max_angular_joint_acceleration_radps2": angular_joint,
                "ineq_min_scaled": minimum, "optimization_ineq_min_scaled": optimization_minimum,
                "optimization_margin_satisfied": bool(optimization_minimum >= -1e-5),
                "qpos": measured["qpos"].tolist(), "ctrl": measured["ctrl"].tolist(), "values": values.tolist(),
                "native_metrics": measured["metrics"], "history": history, "iterations": len(history), "evaluations": self.evaluations}


def main():
    parser = argparse.ArgumentParser(description="Bounded acceleration-tolerance formulation; same staged task and physical gates")
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    assert mujoco.__version__ == "3.8.0"
    matrix = json.loads(args.matrix.read_text())
    assert matrix["status"] == "completed"
    for name, digest in matrix["source_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    support_path = Path(matrix["support_report"])
    assert hashlib.sha256(support_path.read_bytes()).hexdigest() == matrix["support_report_sha256"]
    entries = [entry for entry in json.loads(support_path.read_text())["runs"] if entry["branch"] == "freeze"]
    origin = next(row["solution"] for row in matrix["runs"] if row["anchor"] == 2 and row["swing"] is None)
    terminal = next(row["solution"] for row in matrix["runs"] if row["anchor"] == 2 and row["swing"] == 2)
    destination = ROOT / "output/rl/tolerance_transfer" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    sources = dict(matrix["source_sha256"])
    sources.update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("staged_transfer.py", "tolerance_transfer.py")})
    report = {"status": "running", "matrix": str(args.matrix.resolve()), "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
              "source_sha256": sources, "mujoco": mujoco.__version__, "training_updates": 0,
              "protocol": {"stages": STAGES, "max_iterations_per_stage": 150, "timeout_per_stage_s": 90,
                           "linear_component_bound": .005 / np.sqrt(3), "angular_joint_bound": .05,
                           "physical_linear_norm_gate": .01, "physical_angular_joint_gate": .1,
                           "stop_on_first_failed_waypoint": True, "restarts": 0},
              "solves": [], "replay": None,
              "limitations": "One evidence-driven numerical formulation comparison, not a budget extension of the exact-equality run. Acceleration inequalities are tighter than unchanged physical validation tolerances. Local rank diagnosis is not global infeasibility proof. No landing, progression, robust control, RL or hardware claim."}
    started = time.monotonic()
    accepted, previous = [], origin
    try:
        report["rank_diagnostic"] = rank_diagnostic(ToleranceStageProblem(entries[1], origin, origin, terminal, STAGES[0]))
        for stage in STAGES:
            problem = ToleranceStageProblem(entries[1], previous, origin, terminal, stage)
            solution = problem.solve()
            row = {"stage": stage, "solution": solution}
            report["solves"].append(row)
            print(json.dumps({"stage": stage["name"], "passed": solution["constraint_passed"], "message": solution["message"],
                              "load_n": solution["native_metrics"]["loads_n"][2], "gap_m": solution["native_metrics"]["gaps_m"][2],
                              "acceleration": solution["base_linear_acceleration_mps2"], "angular_joint_acceleration": solution["max_angular_joint_acceleration_radps2"]}), flush=True)
            if not solution["constraint_passed"]:
                break
            accepted.append(row)
            previous = solution
        report["replay"] = replay_path(entries[1], origin, accepted, destination / "loaded_path.npz")
        report["status"] = "completed"
        print(json.dumps({key: value for key, value in report["replay"].items() if key != "trace"}), flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
