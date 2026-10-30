import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from load_transfer import physical_metrics, smooth
from native_contact_seed import NativeProblem
from prepare import ROOT


STAGES = (
    {"name": "load_60_70N", "load": [60, 70], "gap": [-.0045, .002], "guide": .1},
    {"name": "load_35_45N", "load": [35, 45], "gap": [-.0045, .002], "guide": .25},
    {"name": "load_10_20N", "load": [10, 20], "gap": [-.0045, .002], "guide": .45},
    {"name": "unload_0_5N", "load": [0, 5], "gap": [-.0045, .002], "guide": .6},
    {"name": "lift_5_10mm", "load": [0, 1], "gap": [.005, .01], "guide": .7},
    {"name": "lift_15_25mm", "load": [0, 1], "gap": [.015, .025], "guide": .8},
)


def stage_reached(row, stage):
    if not all(row["floor_contacts"][limb] and row["loads_n"][limb] > 5 for limb in (0, 1, 3)):
        return False
    load, gap = row["loads_n"][2], row["gaps_m"][2]
    return bool(stage["load"][0] - 1e-5 <= load <= stage["load"][1] + 1e-5
                and stage["gap"][0] - 1e-6 <= gap <= stage["gap"][1] + 1e-6
                and (stage["gap"][0] <= 0 or not row["floor_contacts"][2]))


class StageProblem(NativeProblem):
    def __init__(self, entry, previous, origin, terminal, stage):
        super().__init__(entry, 2, previous)
        self.stage = stage
        self.guide = (1 - stage["guide"]) * np.array(origin["values"]) + stage["guide"] * np.array(terminal["values"])
        self.previous = np.array(previous["values"])

    def evaluate(self, values):
        raw = super().evaluate(values)
        load = raw["metrics"]["loads_n"][2]
        gap = raw["metrics"]["gaps_m"][2]
        local_limit = np.r_[[.04, .04, .02], [.15] * 3, [.25] * 23]
        inequalities = np.r_[raw["ineq"][:-2],
                             (load - self.stage["load"][0]) / self.weight, (self.stage["load"][1] - load) / self.weight,
                             (gap - self.stage["gap"][0]) / .01, (self.stage["gap"][1] - gap) / .01,
                             1 - np.abs((values[:29] - self.previous[:29]) * self.scale) / local_limit]
        objective = (.01 * np.mean((values[:29] - self.guide[:29]) ** 2)
                     + .001 * np.mean((values[29:] - self.guide[29:]) ** 2)
                     + .001 * np.mean((values[:29] - self.previous[:29]) ** 2))
        return {**raw, "ineq": inequalities, "objective": float(objective)}


def replay_path(entry, origin, accepted, output):
    model = mujoco.MjModel.from_xml_path(entry["model"])
    data = mujoco.MjData(model)
    initial_state = np.array(entry["anchor"]["physical_state"])
    initial_ctrl = np.array(entry["anchor"]["previous_ctrl"])
    mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    start_time = float(data.time)
    rows, controls, qpos, qvel, stages = [], [], [data.qpos.copy()], [data.qvel.copy()], []
    waypoints = [{"name": "four_support_settle", "ctrl": origin["ctrl"], "stage": None}]
    waypoints += [{"name": row["stage"]["name"], "ctrl": row["solution"]["ctrl"], "stage": row["stage"]} for row in accepted]
    peak = physical_metrics(model, data)["penetration"]["depth_m"]
    failure = None
    for waypoint_index, waypoint in enumerate(waypoints):
        previous = data.ctrl.copy()
        local_rows = []
        for step in range(200):
            elapsed = (step + 1) * .02
            blend = smooth((elapsed - (.5 if waypoint_index == 0 else 0)) / 3)
            target = previous * (1 - blend) + np.array(waypoint["ctrl"]) * blend
            data.ctrl[:] = np.clip(data.ctrl + np.clip(target - data.ctrl, -.01, .01), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            controls.append(data.ctrl.copy())
            assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
            for substep in range(round(.02 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time - start_time - len(controls) * .02) > 1e-7:
                raise AssertionError("Invalid state or implicit reset")
            row = physical_metrics(model, data)
            row.update(elapsed_s=len(controls) * .02, stage=waypoint["name"], stage_elapsed_s=elapsed)
            rows.append(row)
            local_rows.append(row)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            peak = max(peak, row["penetration"]["depth_m"])
            if peak > .005:
                failure = "penetration_over_5mm"
            elif row["other_total_n"] > 1:
                failure = "non_support_load_over_1N"
            elif not .16 <= row["base_height_m"] <= .65:
                failure = "base_height"
            if failure:
                break
        tail = local_rows[-10:]
        if waypoint["stage"] is None:
            reached = all(all(row["loads_n"][limb] > 5 and row["floor_contacts"][limb] for limb in range(4)) for row in tail)
        else:
            reached = all(stage_reached(row, waypoint["stage"]) for row in tail)
        reached = bool(not failure and len(local_rows) == 200 and reached)
        stages.append({"name": waypoint["name"], "reached": reached, "duration_s": len(local_rows) * .02,
                       "final_loads_n": local_rows[-1]["loads_n"], "final_gaps_m": local_rows[-1]["gaps_m"]})
        if not reached:
            failure = failure or "stage_not_reached_without_extension"
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    mean_slip = float(np.mean(slips)) if slips else None
    near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows]))
    safe = failure is None and mean_slip is not None and mean_slip <= .02 and near_limit <= .1
    np.savez_compressed(output, qpos=qpos, qvel=qvel, ctrl=controls, initial_physics_state=initial_state, initial_ctrl=initial_ctrl, dt=.02)
    return {"failure": failure, "duration_s": len(rows) * .02, "safe": bool(safe), "prefix_reached": all(row["reached"] for row in stages),
            "complete_lift_passed": bool(safe and len(accepted) == len(STAGES) and len(stages) == len(STAGES) + 1 and all(row["reached"] for row in stages)),
            "max_penetration_m": peak, "mean_loaded_slip_mps": mean_slip, "near_limit_fraction": near_limit,
            "stages": stages, "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser(description="One bounded native equilibrium continuation, then original-state staged motor replay")
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
    origin_run = next(row for row in matrix["runs"] if row["anchor"] == 2 and row["swing"] is None)
    terminal_run = next(row for row in matrix["runs"] if row["anchor"] == 2 and row["swing"] == 2)
    assert origin_run["replays"][0]["passed"] and terminal_run["replays"][0]["passed"]
    origin, terminal, entry = origin_run["solution"], terminal_run["solution"], entries[1]
    destination = ROOT / "output/rl/staged_transfer" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    sources = dict(matrix["source_sha256"])
    sources["staged_transfer.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "running", "matrix": str(args.matrix.resolve()), "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
              "source_sha256": sources, "mujoco": mujoco.__version__, "anchor": 2, "swing": 2, "training_updates": 0,
              "protocol": {"stages": STAGES, "max_iterations_per_stage": 150, "timeout_per_stage_s": 90,
                           "stop_on_first_failed_waypoint": True, "restarts": 0, "ramp_s": 3, "stage_s": 4, "tail_gate_s": .2},
              "solves": [], "replay": None,
              "limitations": "Single previously successful static anchor/limb, not generalization. Native continuation guides use a high-clearance terminal pose but stage gaps are bounded. No live qpos writes, no external forces, no retries or threshold relaxation. Complete lift excludes landing/progression/full crawl."}
    started = time.monotonic()
    previous = origin
    accepted = []
    try:
        for stage in STAGES:
            problem = StageProblem(entry, previous, origin, terminal, stage)
            solution = problem.solve()
            row = {"stage": stage, "solution": solution}
            report["solves"].append(row)
            print(json.dumps({"stage": stage["name"], "passed": solution["constraint_passed"], "message": solution["message"],
                              "acceleration": solution["base_linear_acceleration_mps2"], "angular_joint_acceleration": solution["max_angular_joint_acceleration_radps2"],
                              "load_n": solution["native_metrics"]["loads_n"][2], "gap_m": solution["native_metrics"]["gaps_m"][2]}), flush=True)
            if not solution["constraint_passed"]:
                break
            accepted.append(row)
            previous = solution
        report["replay"] = replay_path(entry, origin, accepted, destination / "loaded_path.npz")
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
