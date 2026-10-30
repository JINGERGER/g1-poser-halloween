import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from component_unload_ramps import guard
from load_transfer import physical_metrics, smooth
from prepare import ROOT
from target_load_ablation import digest


BRANCHES = (("baseline", 0., 3., 6.), ("early_return", .25, 1., 4.), ("extended_return", .25, 3., 6.))


def envelope(elapsed, return_start):
    return smooth(elapsed / .6) * (1 - smooth((elapsed - return_start) / 1.))


def run(model, saved, target, branch, output):
    label, amplitude, return_start, duration = branch
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_ctrl = np.array(saved["initial_ctrl"])
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    initial = physical_metrics(model, data)
    peak = initial["penetration"]["depth_m"]
    assert guard(model, data, initial, peak) is None
    positions, velocities, controls, rows = [data.qpos.copy()], [data.qvel.copy()], [], []
    start_time, failure = float(data.time), None
    period, steps = round(.02 / model.opt.timestep), round(duration / model.opt.timestep)
    assert abs(period * model.opt.timestep - .02) < 1e-12
    for step in range(steps):
        if step % period == 0:
            desired = initial_ctrl + amplitude * envelope((step // period + 1) * .02, return_start) * (target - initial_ctrl)
            data.ctrl[:] += np.clip(desired - data.ctrl, -.01, .01)
            assert np.all(data.ctrl >= model.actuator_ctrlrange[:, 0]) and np.all(data.ctrl <= model.actuator_ctrlrange[:, 1])
        controls.append(data.ctrl.copy())
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        mujoco.mj_step(model, data)
        peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        elapsed = (step + 1) * model.opt.timestep
        assert abs(data.time - start_time - elapsed) < 1e-8
        row = physical_metrics(model, data)
        row.update(elapsed_s=float(elapsed), com_displacement_m=float(np.linalg.norm(np.array(row["com_xy"]) - initial["com_xy"])),
                   velocity_max=float(max(abs(data.qvel))))
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        peak = max(peak, row["penetration"]["depth_m"])
        failure = guard(model, data, row, peak)
        if failure:
            break
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, initial_physics_state=saved["initial_physics_state"],
                        initial_ctrl=initial_ctrl, dt=model.opt.timestep)
    return {"branch": label, "amplitude": amplitude, "return_start_s": return_start, "planned_duration_s": duration,
            "duration_s": len(rows) * model.opt.timestep, "completed": failure is None and len(rows) == steps,
            "failure": failure, "initial_metrics": initial, "peak_penetration_m": peak, "trace": rows,
            "trajectory": str(output), "trajectory_sha256": digest(output)}


def compare(result, baseline):
    rows, original = result["trace"], baseline["trace"]
    start, duration = result["return_start_s"], result["planned_duration_s"]
    hold_indices = [index for index, row in enumerate(rows) if start - .2 < row["elapsed_s"] <= start]
    hold_complete = result["duration_s"] >= start and baseline["duration_s"] >= start
    reduction = None
    if hold_complete:
        reduction = float(1 - np.mean([rows[index]["loads_n"][0] for index in hold_indices]) /
                          np.mean([original[index]["loads_n"][0] for index in hold_indices]))
    output = {"pre_return_reduction_fraction": reduction, "pre_return_window_completed": hold_complete,
              "recovery_passed": False, "final_load_error_max_n": None, "final_com_error_to_baseline_m": None}
    if not result["completed"] or baseline["duration_s"] < duration:
        return output
    tail = [index for index, row in enumerate(rows) if row["elapsed_s"] > duration - .5]
    errors = [float(np.max(np.abs(np.array(rows[index]["loads_n"]) - original[index]["loads_n"]))) for index in tail]
    com_errors = [float(np.linalg.norm(np.array(rows[index]["com_xy"]) - original[index]["com_xy"])) for index in tail]
    output.update(final_load_error_max_n=errors[-1], final_com_error_to_baseline_m=com_errors[-1],
                  tail_load_error_max_n=max(errors), tail_com_error_max_m=max(com_errors),
                  tail_velocity_max=max(rows[index]["velocity_max"] for index in tail),
                  tail_loaded_slip_max_mps=max(rows[index]["loaded_slip_mps"] or 0 for index in tail))
    output["recovery_passed"] = bool(max(errors) <= 5 and max(com_errors) <= .005
                                      and all(rows[index]["velocity_max"] <= .05 and all(rows[index]["floor_contacts"])
                                              and min(rows[index]["loads_n"]) > 5
                                              and rows[index]["loaded_slip_mps"] is not None
                                              and rows[index]["loaded_slip_mps"] <= .02 for index in tail))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ramp_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.ramp_report.read_text())
    assert source["status"] == "completed"
    selected = next(run for run in source["runs"] if run["amplitude"] == .25)
    assert selected["completed"] and selected["sustained_reduction_passed"]
    for key in ("model", "source_trajectory"):
        assert digest(source[key]) == source[key + "_sha256"]
    path = Path(source["model"])
    assert digest(path.with_name("manifest.json")) == source["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    destination = ROOT / "output/rl/component_unload_return" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.ramp_report.resolve()), "source_report_sha256": digest(args.ramp_report),
              "model": str(path), "model_sha256": digest(path), "source_trajectory": source["source_trajectory"],
              "source_trajectory_sha256": source["source_trajectory_sha256"], "candidate_ctrl": source["candidate_ctrl"],
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "component_unload_ramps.py", "load_transfer.py", "diagnose_short_crawl.py")},
              "protocol": {"branches": BRANCHES, "rise_s": .6, "return_s": 1, "control_dt_s": .02, "max_ctrl_delta_rad": .01,
                           "recovery_tail_s": .5, "load_error_limit_n": 5, "com_error_limit_m": .005,
                           "velocity_max": .05, "loaded_slip_max_mps": .02, "restarts": 0},
              "limitations": "Fixed 25% direction and two predeclared return timings; no amplitude search. Same initial state, native motor-only rollouts. Failure stops simulation, not physical recovery. Recovery compares matched-time baseline, not absolute initial pose. Prior contact-sensitivity failure retained.", "runs": []}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with np.load(source["source_trajectory"], allow_pickle=False) as saved:
            for branch in BRANCHES:
                result = run(model, saved, np.array(source["candidate_ctrl"]), branch, destination / f"{branch[0]}.npz")
                result.update(compare(result, report["runs"][0] if report["runs"] else result))
                report["runs"].append(result)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: value for key, value in result.items() if key not in ("trace", "initial_metrics")}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
