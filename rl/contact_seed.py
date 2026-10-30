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

from load_transfer import physical_metrics, smooth
from prepare import ROOT, SUPPORTS
from servo_force import position_servo_force, validate_position_servos


def constraints_passed(equality, inequality):
    return bool(np.isfinite(equality).all() and np.isfinite(inequality).all()
                and np.max(np.abs(equality)) <= 1e-5 and np.min(inequality) >= -1e-5)


class ContactProblem:
    def __init__(self, entry, swing):
        self.entry, self.swing = entry, swing
        path = Path(entry["model"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["model_sha256"]
        self.model = mujoco.MjModel.from_xml_path(str(path))
        validate_position_servos(self.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, self.data, np.array(entry["anchor"]["physical_state"]), mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = entry["anchor"]["previous_ctrl"]
        mujoco.mj_forward(self.model, self.data)
        self.nominal = self.data.qpos.copy()
        self.floor = self.model.geom("floor").id
        self.supports = [self.model.geom(name).id for name in SUPPORTS]
        self.stance = [index for index in range(4) if index != swing]
        self.weight = float(sum(self.model.body_mass) * np.linalg.norm(self.model.opt.gravity))
        self.points = []
        for contact in self.data.contact:
            if self.floor not in (contact.geom1, contact.geom2):
                continue
            geom = int(contact.geom2 if contact.geom1 == self.floor else contact.geom1)
            if geom not in [self.supports[index] for index in self.stance]:
                continue
            body = int(self.model.geom_bodyid[geom])
            local = self.data.xmat[body].reshape(3, 3).T @ (contact.pos - self.data.xpos[body])
            self.points.append((body, local.copy(), contact.pos.copy(), float(min(contact.friction[:2])), self.supports.index(geom)))
        assert set(point[4] for point in self.points) == set(self.stance)
        self.other = [index for index in range(self.model.ngeom) if index != self.floor and index not in self.supports
                      and ((self.model.geom_contype[index] & self.model.geom_conaffinity[self.floor])
                           or (self.model.geom_contype[self.floor] & self.model.geom_conaffinity[index]))]
        self.scale = np.r_[[.15, .15, .08], [.4] * 3, [.5] * 23]
        self.joints = self.model.actuator_trnid[:, 0]
        self.limits = self.model.jnt_actfrcrange[self.joints]
        self.cached = None
        self.deadline = float("inf")
        self.evaluations = 0

    def evaluate(self, values):
        if time.monotonic() > self.deadline:
            raise TimeoutError("Fixed 90s solve budget exhausted")
        if self.cached is not None and np.array_equal(values, self.cached[0]):
            return self.cached[1]
        self.evaluations += 1
        self.data.qpos[:] = self.nominal
        mujoco.mj_integratePos(self.model, self.data.qpos, values[:29] * self.scale, 1)
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        forces = values[29:].reshape(-1, 3) * self.weight
        generalized = np.zeros(self.model.nv)
        drift = []
        friction = []
        for point, force in zip(self.points, forces):
            body, local, initial, mu, limb = point
            world = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ local
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jacobian, None, world, body)
            generalized += jacobian.T @ force
            drift.extend((world - initial) / np.array([.005, .005, .002]))
            friction.append((mu * force[2] - abs(force[0]) - abs(force[1])) / self.weight)
        required = self.data.qfrc_bias - self.data.qfrc_passive - generalized
        torque = required[6:]
        ctrl = (torque - self.model.actuator_biasprm[:, 0] - self.model.actuator_biasprm[:, 1] * self.data.qpos[7:]) / self.model.actuator_gainprm[:, 0]
        gaps = np.array([mujoco.mj_geomDistance(self.model, self.data, geom, self.floor, 2, None) for geom in self.supports])
        other_gaps = [mujoco.mj_geomDistance(self.model, self.data, geom, self.floor, 2, None) for geom in self.other]
        loads = np.array([sum(force[2] for point, force in zip(self.points, forces) if point[4] == limb) for limb in self.stance])
        collision_gap = min((contact.dist for contact in self.data.contact if self.floor not in (contact.geom1, contact.geom2)), default=0)
        joint_limits = self.model.jnt_range[self.joints]
        inequalities = np.r_[1 - np.abs(drift), friction, (loads - 5) / self.weight,
                             (gaps[self.stance] + .005) / .01, (.002 - gaps[self.stance]) / .01,
                             np.array(other_gaps) / .01, (collision_gap + .0005) / .01,
                             (torque - .95 * self.limits[:, 0]) / 100, (.95 * self.limits[:, 1] - torque) / 100,
                             ctrl - self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1] - ctrl,
                             self.data.qpos[7:] - joint_limits[:, 0], joint_limits[:, 1] - self.data.qpos[7:],
                             self.data.qpos[2] - .16, .65 - self.data.qpos[2]]
        if self.swing is not None:
            inequalities = np.r_[inequalities, (gaps[self.swing] - .02) / .01]
        result = {"eq": required[:6] / self.weight, "ineq": inequalities, "qpos": self.data.qpos.copy(),
                  "ctrl": ctrl.copy(), "forces": forces.copy(), "gaps": gaps.copy(), "torque": torque.copy(),
                  "objective": float(.01 * np.mean(values[:29] ** 2) + .001 * np.mean(values[29:] ** 2))}
        self.cached = values.copy(), result
        return result

    def solve(self):
        initial = np.zeros(29 + 3 * len(self.points))
        initial[31::3] = 1 / len(self.points)
        bounds = [(-1, 1)] * 29 + [bound for point in self.points for bound in ((-1, 1), (-1, 1), (0, 1))]
        self.deadline = time.monotonic() + 90
        history = []
        last = initial.copy()

        def callback(values):
            nonlocal last
            last = values.copy()
            measured = self.evaluate(values)
            history.append({"eq_max": float(np.max(np.abs(measured["eq"]))), "ineq_min": float(np.min(measured["ineq"]))})

        try:
            result = minimize(lambda values: self.evaluate(values)["objective"], initial, method="SLSQP", bounds=bounds,
                              constraints=[{"type": "eq", "fun": lambda values: self.evaluate(values)["eq"]},
                                           {"type": "ineq", "fun": lambda values: self.evaluate(values)["ineq"]}],
                              callback=callback, options={"maxiter": 150, "ftol": 1e-8})
            values, success, message = result.x, bool(result.success), str(result.message)
        except TimeoutError as error:
            values, success, message = last, False, str(error)
        self.deadline = float("inf")
        measured = self.evaluate(values)
        eq_max, ineq_min = float(np.max(np.abs(measured["eq"]))), float(np.min(measured["ineq"]))
        accepted = constraints_passed(measured["eq"], measured["ineq"])
        reconstructed = position_servo_force(self.model, measured["ctrl"], measured["qpos"][7:], np.zeros(23))
        self.data.qpos[:] = measured["qpos"]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = measured["ctrl"]
        mujoco.mj_forward(self.model, self.data)
        native = physical_metrics(self.model, self.data)
        return {"solver_success": success, "message": message, "constraint_passed": bool(accepted), "eq_max_scaled": eq_max,
                "ineq_min_scaled": ineq_min, "qpos": measured["qpos"].tolist(), "ctrl": measured["ctrl"].tolist(),
                "gaps_m": measured["gaps"].tolist(), "contact_forces_world_n": measured["forces"].tolist(),
                "servo_reconstruction_error_nm": float(np.max(np.abs(reconstructed - measured["torque"]))),
                "native_initial_metrics": native, "native_initial_base_qacc": self.data.qacc[:6].tolist(),
                "contact_points": [{"body": point[0], "local": point[1].tolist(), "initial_world": point[2].tolist(), "mu": point[3], "limb": point[4]} for point in self.points],
                "values": values.tolist(), "iterations": len(history), "evaluations": self.evaluations, "history": history}


def replay(problem, solution, branch, output):
    model = problem.model
    data = mujoco.MjData(model)
    if branch == "hypothetical_hold":
        data.qpos[:] = solution["qpos"]
        data.ctrl[:] = solution["ctrl"]
    else:
        mujoco.mj_setState(model, data, np.array(problem.entry["anchor"]["physical_state"]), mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = problem.entry["anchor"]["previous_ctrl"]
    mujoco.mj_forward(model, data)
    initial_ctrl = data.ctrl.copy()
    initial_state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    start_time = float(data.time)
    rows, qpos, qvel, controls = [], [data.qpos.copy()], [data.qvel.copy()], []
    peak_penetration = 0.0
    steps = 150 if branch == "hypothetical_hold" else 300
    for step in range(steps):
        blend = 1 if branch == "hypothetical_hold" else smooth(((step + 1) * .02 - .5) / 3)
        target = initial_ctrl * (1 - blend) + np.array(solution["ctrl"]) * blend
        data.ctrl[:] = np.clip(data.ctrl + np.clip(target - data.ctrl, -.01, .01), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        controls.append(data.ctrl.copy())
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        for substep in range(round(.02 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak_penetration = max(peak_penetration, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time - start_time - (step + 1) * .02) > 1e-7:
            raise AssertionError("Invalid physics state or implicit reset")
        row = physical_metrics(model, data)
        peak_penetration = max(peak_penetration, row["penetration"]["depth_m"])
        row["elapsed_s"] = (step + 1) * .02
        rows.append(row)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        if row["other_total_n"] > 1 or peak_penetration > .005 or not .16 <= data.qpos[2] <= .65:
            break
    settled = [row for row in rows if row["elapsed_s"] > (1 if branch == "hypothetical_hold" else 4)]
    tripod = bool(problem.swing == 3 and settled and all(row["loads_n"][3] <= 1 and not row["floor_contacts"][3] and row["gaps_m"][3] >= .015
                                and all(row["loads_n"][limb] > 5 for limb in (0, 1, 2)) for row in settled))
    stance = bool(settled and all(all(row["loads_n"][limb] > 5 and row["floor_contacts"][limb] for limb in problem.stance) for row in settled))
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    mean_slip = float(np.mean(slips)) if slips else None
    near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows]))
    safe = (len(rows) == steps and max(row["other_total_n"] for row in rows) <= 1 and peak_penetration <= .005
            and all(.16 <= row["base_height_m"] <= .65 for row in rows)
            and mean_slip is not None and mean_slip <= .02 and near_limit <= .1)
    np.savez_compressed(output, qpos=qpos, qvel=qvel, ctrl=controls, initial_physics_state=initial_state, dt=.02)
    return {"branch": branch, "duration_s": len(rows) * .02, "safe": bool(safe), "sustained_tripod": tripod,
            "sustained_stance": stance, "passed": bool(safe and (tripod if problem.swing == 3 else stance)), "max_penetration_m": peak_penetration, "mean_loaded_slip_mps": mean_slip,
            "near_limit_fraction": near_limit, "trajectory": str(output), "trace": rows}


def main():
    parser = argparse.ArgumentParser(description="Force-constrained static candidates, then unassisted native holds/transfers")
    parser.add_argument("support_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.support_report.read_text())
    entries = [entry for entry in source["runs"] if entry["branch"] == "freeze"]
    assert len(entries) == 2 and source["status"] == "completed"
    assert mujoco.__version__ == "3.8.0", "Keep the experimental physics engine unchanged"
    destination = ROOT / "output/rl/contact_seed" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_updates": 0, "mujoco": mujoco.__version__, "scipy": scipy.__version__,
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("contact_seed.py", "load_transfer.py", "diagnose_short_crawl.py", "servo_force.py")},
              "support_report_sha256": hashlib.sha256(args.support_report.read_bytes()).hexdigest(),
              "limitations": "Local static force relaxation, not dynamic optimization/infeasibility proof. Anchor contact points, unilateral force and conservative L1 friction cone; no contact moments. Native soft-contact force consistency is not assumed: hypothetical hold is separate from loaded-state transfer. Fixed 3s transition, no timing optimization. No RL or hardware claim."}
    started = time.monotonic()
    try:
        for index, entry in enumerate(entries):
            for swing in (None, 3):
                problem = ContactProblem(entry, swing)
                solution = problem.solve()
                result = {"anchor": index + 1, "swing": swing, "model": entry["model"], "model_sha256": entry["model_sha256"],
                          "solution": solution, "replays": []}
                if solution["constraint_passed"]:
                    for branch in ("hypothetical_hold", "loaded_transfer"):
                        result["replays"].append(replay(problem, solution, branch, destination / f"anchor{index + 1}_swing{swing}_{branch}.npz"))
                report["runs"].append(result)
                (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"anchor": index + 1, "swing": swing, "constraint_passed": solution["constraint_passed"],
                                  "message": solution["message"], "eq_max": solution["eq_max_scaled"], "ineq_min": solution["ineq_min_scaled"],
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
