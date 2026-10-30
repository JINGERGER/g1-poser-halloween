import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import scipy
from scipy.optimize import minimize

from contact_seed import ContactProblem, replay
from load_transfer import physical_metrics
from prepare import ROOT, SITES


class NativeProblem(ContactProblem):
    def __init__(self, entry, swing, seed):
        super().__init__(entry, swing)
        self.initial = np.r_[seed["values"][:29], seed["ctrl"]]
        self.stance_xy = np.array([self.data.site(SITES[index]).xpos[:2].copy() for index in self.stance])
        self.cached = None

    def evaluate(self, values):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("90s native solve budget exhausted")
        if self.cached is not None and np.array_equal(self.cached[0], values):
            return self.cached[1]
        self.evaluations += 1
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.nominal
        mujoco.mj_integratePos(self.model, self.data.qpos, values[:29] * self.scale, 1)
        self.data.ctrl[:] = values[29:]
        mujoco.mj_forward(self.model, self.data)
        measured = physical_metrics(self.model, self.data)
        joint_limits = self.model.jnt_range[self.joints]
        gaps = np.array(measured["gaps_m"])
        other_gaps = [mujoco.mj_geomDistance(self.model, self.data, geom, self.floor, 2, None) for geom in self.other]
        xy = np.array([self.data.site(SITES[index]).xpos[:2] for index in self.stance])
        inequalities = np.r_[(.0045 - measured["penetration"]["depth_m"]) / .01,
                             (np.array(measured["loads_n"])[self.stance] - 10) / self.weight,
                             (.002 - gaps[self.stance]) / .01, np.array(other_gaps) / .01,
                             1 - np.abs((xy - self.stance_xy).ravel()) / .03,
                             .95 - np.array(measured["torque_ratios"]),
                             self.data.qpos[7:] - joint_limits[:, 0], joint_limits[:, 1] - self.data.qpos[7:],
                             self.data.qpos[2] - .16, .65 - self.data.qpos[2],
                             (1 - measured["other_total_n"]) / self.weight]
        if self.swing is not None:
            inequalities = np.r_[inequalities, (gaps[self.swing] - .02) / .01,
                                 (1 - measured["loads_n"][self.swing]) / self.weight]
        result = {"eq": self.data.qacc.copy() / np.r_[[9.81] * 3, [10.] * 26], "ineq": inequalities,
                  "qpos": self.data.qpos.copy(), "ctrl": self.data.ctrl.copy(), "qacc": self.data.qacc.copy(), "metrics": measured,
                  "objective": float(.01 * np.mean(values[:29] ** 2) + .001 * np.mean((values[29:] - self.initial[29:]) ** 2))}
        self.cached = values.copy(), result
        return result

    def solve(self):
        bounds = [(-1, 1)] * 29 + list(map(tuple, self.model.actuator_ctrlrange))
        initial = np.clip(self.initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        self.deadline = time.monotonic() + 90
        history = []
        last = initial.copy()

        def callback(values):
            nonlocal last
            last = values.copy()
            measured = self.evaluate(values)
            history.append({"eq_max_scaled": float(np.max(np.abs(measured["eq"]))), "ineq_min_scaled": float(np.min(measured["ineq"]))})

        try:
            result = minimize(lambda values: self.evaluate(values)["objective"], initial, method="SLSQP", bounds=bounds,
                              constraints=[{"type": "eq", "fun": lambda values: self.evaluate(values)["eq"]},
                                           {"type": "ineq", "fun": lambda values: self.evaluate(values)["ineq"]}],
                              callback=callback, options={"maxiter": 150, "ftol": 1e-8, "eps": 1e-6})
            values, success, message = result.x, bool(result.success), str(result.message)
        except TimeoutError as error:
            values, success, message = last, False, str(error)
        self.deadline = float("inf")
        measured = self.evaluate(values)
        linear = float(np.linalg.norm(measured["qacc"][:3]))
        angular_joint = float(np.max(np.abs(measured["qacc"][3:])))
        minimum = float(np.min(measured["ineq"]))
        passed = linear <= .01 and angular_joint <= .1 and minimum >= -1e-5
        return {"solver_success": success, "message": message, "constraint_passed": bool(passed),
                "base_linear_acceleration_mps2": linear, "max_angular_joint_acceleration_radps2": angular_joint,
                "ineq_min_scaled": minimum, "qpos": measured["qpos"].tolist(), "ctrl": measured["ctrl"].tolist(),
                "native_metrics": measured["metrics"], "values": values.tolist(), "history": history,
                "evaluations": self.evaluations, "iterations": len(history)}


def main():
    parser = argparse.ArgumentParser(description="Native soft-contact equilibrium refinement; frozen model and bounded budget")
    parser.add_argument("support_report", type=Path)
    parser.add_argument("contact_report", type=Path)
    args = parser.parse_args()
    assert mujoco.__version__ == "3.8.0"
    support = json.loads(args.support_report.read_text())
    previous = json.loads(args.contact_report.read_text())
    assert support["status"] == previous["status"] == "completed"
    assert previous["support_report_sha256"] == hashlib.sha256(args.support_report.read_bytes()).hexdigest()
    for name, digest in previous["source_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
    entries = [entry for entry in support["runs"] if entry["branch"] == "freeze"]
    assert len(entries) == 2 and len(previous["runs"]) == 4
    destination = ROOT / "output/rl/native_contact_seed" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_updates": 0, "mujoco": mujoco.__version__, "scipy": scipy.__version__,
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("native_contact_seed.py", "contact_seed.py", "load_transfer.py", "diagnose_short_crawl.py", "servo_force.py")},
              "support_report": str(args.support_report.resolve()), "contact_report": str(args.contact_report.resolve()),
              "support_report_sha256": hashlib.sha256(args.support_report.read_bytes()).hexdigest(),
              "contact_report_sha256": hashlib.sha256(args.contact_report.read_bytes()).hexdigest(),
              "protocol": {"maxiter": 150, "per_solve_timeout_s": 90, "minimum_stance_load_n": 10, "optimization_penetration_m": .0045,
                           "replay_penetration_m": .005, "base_acceleration_mps2": .01, "angular_joint_acceleration_radps2": .1},
              "limitations": "Local native equilibrium, not global feasibility proof. Free base and 23 positions/servo targets; zero velocity scratch states only. Native solver warmstart cleared each evaluation. 4.5mm optimization margin within unchanged 5mm physical gate. Three-second fixed transition tested only after hold passes. No full gait, RL or hardware claim."}
    started = time.monotonic()
    try:
        for old in previous["runs"]:
            entry = entries[old["anchor"] - 1]
            assert old["model_sha256"] == entry["model_sha256"]
            problem = NativeProblem(entry, old["swing"], old["solution"])
            solution = problem.solve()
            result = {"anchor": old["anchor"], "swing": old["swing"], "solution": solution,
                      "model": entry["model"], "model_sha256": entry["model_sha256"], "replays": []}
            if solution["constraint_passed"]:
                hold = replay(problem, solution, "hypothetical_hold", destination / f"anchor{old['anchor']}_swing{old['swing']}_hold.npz")
                result["replays"].append(hold)
                if hold["passed"]:
                    result["replays"].append(replay(problem, solution, "loaded_transfer", destination / f"anchor{old['anchor']}_swing{old['swing']}_transfer.npz"))
            report["runs"].append(result)
            (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"anchor": old["anchor"], "swing": old["swing"],
                              **{key: solution[key] for key in ("constraint_passed", "message", "base_linear_acceleration_mps2", "max_angular_joint_acceleration_radps2", "ineq_min_scaled", "iterations")},
                              "replays": [{key: value for key, value in row.items() if key != "trace"} for row in result["replays"]]}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
