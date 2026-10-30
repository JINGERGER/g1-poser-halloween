import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from load_transfer import physical_metrics
from prepare import ROOT
from reference_gait import paired_cycles
from support_handoff import leg_offset, pulse


DT = 0.02
STEPS = 200
GENERATIONS = 6
POPULATION = 16


def offsets(knots):
    values = np.vstack((np.zeros(23), knots, np.zeros(23)))
    elapsed = np.arange(1, STEPS + 1) * DT
    segment = np.minimum(elapsed.astype(int), 3)
    fraction = elapsed - segment
    blend = 6 * fraction ** 5 - 15 * fraction ** 4 + 10 * fraction ** 3
    return values[segment] * (1 - blend[:, None]) + values[segment + 1] * blend[:, None]


def controls(initial, seed, knots, ranges):
    desired = initial + seed + offsets(knots)
    result = []
    previous = initial.copy()
    for target in desired:
        previous = np.clip(previous + np.clip(target - previous, -.01, .01), ranges[:, 0], ranges[:, 1])
        result.append(previous.copy())
    return np.array(result)


class Experiment:
    def __init__(self, entry):
        self.entry = entry
        path = Path(entry["model"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["model_sha256"]
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.initial = np.array(entry["anchor"]["physical_state"])
        self.initial_ctrl = np.array(entry["anchor"]["previous_ctrl"])
        self.substeps = round(DT / self.model.opt.timestep)
        assert abs(self.substeps * self.model.opt.timestep - DT) < 1e-10
        self.reset()
        self.origin = self.data.qpos[:2].copy()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        self.heading = rotation.reshape(3, 3)[:2, 2].copy()
        self.heading /= np.linalg.norm(self.heading)
        delta, self.ik_error = leg_offset(self.model, self.data.qpos.copy())
        self.ik_seed = np.array([pulse((step + 1) * DT) * delta for step in range(STEPS)])
        self.protocol = json.loads(Path(__file__).with_name("reference_eval.json").read_text())

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, self.initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = self.initial_ctrl
        mujoco.mj_forward(self.model, self.data)

    def rollout(self, targets, detailed=False):
        self.reset()
        start = float(self.data.time)
        rows, positions, velocities = [], [self.data.qpos.copy()], [self.data.qvel.copy()]
        peak_penetration = 0.0
        for step, target in enumerate(targets):
            self.data.ctrl[:] = target
            assert not self.data.xfrc_applied.any() and not self.data.qfrc_applied.any()
            for substep in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
                peak_penetration = max(peak_penetration, max((-contact.dist for contact in self.data.contact), default=0))
            mujoco.mj_forward(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or abs(self.data.time - start - (step + 1) * DT) > 1e-7:
                raise AssertionError("Nonfinite state or implicit reset")
            row = physical_metrics(self.model, self.data)
            displacement = self.data.qpos[:2] - self.origin
            row.update(elapsed_s=(step + 1) * DT, progress_m=float(displacement @ self.heading),
                       lateral_m=float(displacement @ np.array([-self.heading[1], self.heading[0]])))
            rows.append(row)
            if detailed:
                positions.append(self.data.qpos.copy())
                velocities.append(self.data.qvel.copy())
        cycles = paired_cycles(rows, self.protocol, DT)
        peak_penetration = max(peak_penetration, max(row["penetration"]["depth_m"] for row in rows))
        max_other = max(row["other_total_n"] for row in rows)
        slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
        mean_slip = float(np.mean(slips)) if slips else None
        near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows]))
        safe = (peak_penetration <= .005 and max_other <= 1 and all(.16 <= row["base_height_m"] <= .65 for row in rows)
                and mean_slip is not None and mean_slip <= .02 and near_limit <= .1)
        swing = [row for row in rows if 1.0 <= row["elapsed_s"] <= 1.8]
        task_loss = float(np.mean([(max(0, .015 - row["gaps_m"][3]) / .015) ** 2 + min(row["loads_n"][3] / 100, 3) ** 2 for row in swing])
                          + (max(0, .01 - rows[-1]["progress_m"]) / .01) ** 2)
        violation = max(0, peak_penetration / .005 - 1) + max(0, max_other - 1) / 10
        violation += max(0, (mean_slip or 0) / .02 - 1) + max(0, near_limit / .1 - 1)
        violation += sum(not .16 <= row["base_height_m"] <= .65 for row in rows) / STEPS
        settled = float(np.sqrt(np.mean(self.data.qvel ** 2))) < .2
        summary = {"safe": bool(safe), "task_loss": task_loss, "violation": float(violation), "cycles": cycles,
                   "progress_m": rows[-1]["progress_m"], "max_lateral_m": max(abs(row["lateral_m"]) for row in rows),
                   "max_penetration_m": float(peak_penetration), "max_other_load_n": max_other,
                   "mean_loaded_slip_mps": mean_slip, "near_torque_limit_fraction": near_limit,
                   "minimum_right_foot_load_n": min(row["loads_n"][3] for row in rows),
                   "maximum_right_foot_gap_m": max(row["gaps_m"][3] for row in rows), "settled": settled,
                   "passed": bool(safe and settled and cycles[3]["count"] >= 1 and rows[-1]["progress_m"] >= .01
                                  and max(abs(row["lateral_m"]) for row in rows) <= .03)}
        if detailed:
            return summary, rows, np.array(positions), np.array(velocities)
        return summary


def rank(summary):
    return (not summary["safe"], summary["violation"] if not summary["safe"] else 0, summary["task_loss"])


def search(experiment, branch, deadline):
    seed = experiment.ik_seed if branch == "ik" else np.zeros_like(experiment.ik_seed)
    rng = np.random.default_rng(20261019)
    mean = np.zeros((3, 23))
    std = np.full_like(mean, .06)
    best_knots = mean.copy()
    baseline = experiment.rollout(controls(experiment.initial_ctrl, seed, mean, experiment.model.actuator_ctrlrange))
    best = baseline
    history = []
    for generation in range(GENERATIONS):
        candidates = np.clip(mean + rng.normal(size=(POPULATION, 3, 23)) * std, -.2, .2)
        candidates[0] = best_knots
        results = []
        for knots in candidates:
            if time.monotonic() >= deadline:
                return best_knots, baseline, best, history, False
            summary = experiment.rollout(controls(experiment.initial_ctrl, seed, knots, experiment.model.actuator_ctrlrange))
            results.append(summary)
            history.append({"generation": generation, "candidate": len(results) - 1, **summary})
            if rank(summary) < rank(best):
                best, best_knots = summary, knots.copy()
        elite = candidates[sorted(range(POPULATION), key=lambda index: rank(results[index]))[:4]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), .01)
    return best_knots, baseline, best, history, True


def main():
    parser = argparse.ArgumentParser(description="Bounded native-physics direct shooting: IK seed versus no seed, no RL updates")
    parser.add_argument("support_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.support_report.read_text())
    entries = [entry for entry in source["runs"] if entry["branch"] == "freeze"]
    assert source["status"] == "completed" and len(entries) == 2
    destination = ROOT / "output/rl/physical_seed" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_updates": 0, "generations": GENERATIONS, "population": POPULATION,
              "duration_s": STEPS * DT, "per_branch_timeout_s": 120, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "support_report_sha256": hashlib.sha256(args.support_report.read_bytes()).hexdigest(),
              "limitations": "Bounded 69-parameter open-loop shooting, not globally optimal or a feedback controller. IK is initial guess only. Fixed timing and right-foot task. Two loaded anchors, one optimizer RNG seed. Endpoint forces/slip/torques, substep penetration. A passed single-limb fragment is not full crawling or hardware proof."}
    started = time.monotonic()
    try:
        for anchor_index, entry in enumerate(entries):
            experiment = Experiment(entry)
            for branch in ("none", "ik"):
                branch_started = time.monotonic()
                knots, baseline, best, history, completed = search(experiment, branch, branch_started + 120)
                seed = experiment.ik_seed if branch == "ik" else np.zeros_like(experiment.ik_seed)
                targets = controls(experiment.initial_ctrl, seed, knots, experiment.model.actuator_ctrlrange)
                checked, trace, qpos, qvel = experiment.rollout(targets, detailed=True)
                assert checked == best
                artifact = destination / f"anchor{anchor_index + 1}_{branch}.npz"
                np.savez_compressed(artifact, ctrl=targets, qpos=qpos, qvel=qvel, knots=knots, dt=DT)
                result = {"anchor": anchor_index + 1, "branch": branch, "completed_budget": completed,
                          "evaluations": len(history) + 1, "elapsed_s": time.monotonic() - branch_started,
                          "initial_physics_state": experiment.initial.tolist(), "initial_ctrl": experiment.initial_ctrl.tolist(),
                          "model": entry["model"], "model_sha256": entry["model_sha256"], "ik_error_m": experiment.ik_error,
                          "baseline": baseline, "best": best, "history": history, "trace": trace, "trajectory": str(artifact)}
                report["runs"].append(result)
                (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"anchor": anchor_index + 1, "branch": branch, "evaluations": result["evaluations"], **best}), flush=True)
        report["status"] = "completed" if all(run["completed_budget"] for run in report["runs"]) else "budget_limited"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
