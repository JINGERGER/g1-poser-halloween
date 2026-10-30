import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import scipy

from contact_seed import ContactProblem
from load_transfer import physical_metrics, smooth
from native_contact_seed import NativeProblem
from prepare import ROOT, SITES


def supported(row, swing):
    stance = [limb for limb in range(4) if limb != swing]
    loaded = all(row["loads_n"][limb] > 5 and row["floor_contacts"][limb] for limb in stance)
    if swing is None:
        return bool(loaded)
    return bool(loaded and row["loads_n"][swing] <= 1 and not row["floor_contacts"][swing] and row["gaps_m"][swing] >= .015)


def replay(problem, solution, branch, output):
    model = problem.model
    data = mujoco.MjData(model)
    if branch == "hold":
        data.qpos[:] = solution["qpos"]
        data.ctrl[:] = solution["ctrl"]
    else:
        mujoco.mj_setState(model, data, np.array(problem.entry["anchor"]["physical_state"]), mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = problem.entry["anchor"]["previous_ctrl"]
    mujoco.mj_forward(model, data)
    initial_ctrl = data.ctrl.copy()
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    started = float(data.time)
    rows, positions, velocities, controls = [], [data.qpos.copy()], [data.qvel.copy()], []
    peak = physical_metrics(model, data)["penetration"]["depth_m"]
    steps = 150 if branch == "hold" else 300
    substeps = round(.02 / model.opt.timestep)
    assert abs(substeps * model.opt.timestep - .02) < 1e-10
    for step in range(steps):
        blend = 1 if branch == "hold" else smooth(((step + 1) * .02 - .5) / 3)
        target = initial_ctrl * (1 - blend) + np.array(solution["ctrl"]) * blend
        data.ctrl[:] = np.clip(data.ctrl + np.clip(target - data.ctrl, -.01, .01), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        controls.append(data.ctrl.copy())
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        for substep in range(substeps):
            mujoco.mj_step(model, data)
            peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time - started - (step + 1) * .02) > 1e-7:
            raise AssertionError("Invalid state or implicit reset")
        row = physical_metrics(model, data)
        row["elapsed_s"] = (step + 1) * .02
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        peak = max(peak, row["penetration"]["depth_m"])
        if peak > .005 or row["other_total_n"] > 1 or not .16 <= row["base_height_m"] <= .65:
            break
    settled = [row for row in rows if row["elapsed_s"] > (1 if branch == "hold" else 4)]
    stance_passed = bool(settled and all(supported(row, problem.swing) for row in settled))
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    mean_slip = float(np.mean(slips)) if slips else None
    near_limit = float(np.mean([max(row["torque_ratios"]) >= .95 for row in rows]))
    safe = (len(rows) == steps and peak <= .005 and max(row["other_total_n"] for row in rows) <= 1
            and all(.16 <= row["base_height_m"] <= .65 for row in rows)
            and mean_slip is not None and mean_slip <= .02 and near_limit <= .1)
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, initial_physics_state=initial, initial_ctrl=initial_ctrl, dt=.02)
    return {"branch": branch, "swing": problem.swing, "duration_s": len(rows) * .02, "safe": bool(safe),
            "sustained_support": stance_passed, "passed": bool(safe and stance_passed), "max_penetration_m": peak,
            "mean_loaded_slip_mps": mean_slip, "near_limit_fraction": near_limit, "trace": rows, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Paired first-limb matrix; equal bounded force/native solves, no retraining")
    parser.add_argument("support_report", type=Path)
    args = parser.parse_args()
    assert mujoco.__version__ == "3.8.0"
    support = json.loads(args.support_report.read_text())
    entries = [entry for entry in support["runs"] if entry["branch"] == "freeze"]
    assert support["status"] == "completed" and len(entries) == 2
    destination = ROOT / "output/rl/support_order_matrix" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    sources = ("support_order_matrix.py", "contact_seed.py", "native_contact_seed.py", "load_transfer.py", "diagnose_short_crawl.py", "servo_force.py", "prepare.py")
    report = {"status": "running", "runs": [], "training_updates": 0, "mujoco": mujoco.__version__, "scipy": scipy.__version__,
              "support_report": str(args.support_report.resolve()), "support_report_sha256": hashlib.sha256(args.support_report.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
              "protocol": {"anchors": 2, "swing_order": [None, 0, 1, 2, 3], "stages_per_cell": 2,
                           "max_iterations_per_stage": 150, "timeout_per_stage_s": 90, "restarts": 0,
                           "native_stance_min_n": 10, "native_clearance_m": .02, "native_penetration_m": .0045,
                           "replay_stance_min_n": 5, "replay_clearance_m": .015, "replay_penetration_m": .005},
              "limitations": "Two fixed loaded anchors; equal stage iteration/time limits, not global optimality or identical solver effort. Fresh force relaxation for every limb, including right foot. Native refinement always uses its own last force candidate, not a best-of-history checkpoint. Static candidates and hypothetical holds are not reachable motion; transfer is checked separately. No landing/forward progression/full gait or hardware claim."}
    started = time.monotonic()
    try:
        for index, entry in enumerate(entries):
            for swing in (None, 0, 1, 2, 3):
                cell_started = time.monotonic()
                relaxed = ContactProblem(entry, swing).solve()
                problem = NativeProblem(entry, swing, relaxed)
                solution = problem.solve()
                result = {"anchor": index + 1, "swing": swing, "limb": "all_support" if swing is None else SITES[swing],
                          "model": entry["model"], "model_sha256": entry["model_sha256"],
                          "anchor_state_sha256": entry["anchor"]["paired_state_sha256"], "relaxed": relaxed,
                          "solution": solution, "replays": []}
                if solution["constraint_passed"]:
                    hold = replay(problem, solution, "hold", destination / f"anchor{index + 1}_limb{swing}_hold.npz")
                    result["replays"].append(hold)
                    if hold["passed"]:
                        result["replays"].append(replay(problem, solution, "transfer", destination / f"anchor{index + 1}_limb{swing}_transfer.npz"))
                result["elapsed_s"] = time.monotonic() - cell_started
                report["runs"].append(result)
                (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"anchor": index + 1, "limb": result["limb"], "relaxed_passed": relaxed["constraint_passed"],
                                  "native_passed": solution["constraint_passed"], "native_message": solution["message"],
                                  "native_linear_acceleration": solution["base_linear_acceleration_mps2"],
                                  "native_angular_joint_acceleration": solution["max_angular_joint_acceleration_radps2"],
                                  "elapsed_s": result["elapsed_s"],
                                  "replays": [{key: value for key, value in replay_result.items() if key not in ("trace", "trajectory")} for replay_result in result["replays"]]}), flush=True)
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
