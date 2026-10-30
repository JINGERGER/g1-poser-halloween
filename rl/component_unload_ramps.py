import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from load_transfer import physical_metrics, smooth
from prepare import ROOT
from target_load_ablation import digest


AMPLITUDES = (0., .1, .25, .5)


def guard(model, data, row, peak):
    if peak > .005:
        return "penetration_over_5mm"
    if row["other_total_n"] > 1:
        return "non_support_load_over_1N"
    if not .16 <= row["base_height_m"] <= .65:
        return "base_height"
    if min(row["loads_n"][1:]) <= 5 or not all(row["floor_contacts"][1:]):
        return "remaining_support_lost"
    if max(row["torque_ratios"]) >= .95:
        return "instantaneous_torque_over_95percent"
    joints = model.actuator_trnid[:, 0]
    angles = data.qpos[model.jnt_qposadr[joints]]
    if np.any(angles < model.jnt_range[joints, 0] - .001) or np.any(angles > model.jnt_range[joints, 1] + .001):
        return "joint_limit"
    return None


def run(model, saved, candidate_ctrl, amplitude, output):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial_ctrl = np.array(saved["initial_ctrl"])
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    initial = physical_metrics(model, data)
    peak = initial["penetration"]["depth_m"]
    assert guard(model, data, initial, peak) is None
    positions, velocities, controls, rows = [data.qpos.copy()], [data.qvel.copy()], [], []
    initial_time = float(data.time)
    failure = None
    period_steps = round(.02 / model.opt.timestep)
    total_steps = round(1 / model.opt.timestep)
    assert abs(period_steps * model.opt.timestep - .02) < 1e-12
    assert abs(total_steps * model.opt.timestep - 1) < 1e-12
    for step in range(total_steps):
        if step % period_steps == 0:
            control_time = (step // period_steps + 1) * .02
            desired = initial_ctrl + amplitude * smooth(control_time / .6) * (candidate_ctrl - initial_ctrl)
            data.ctrl[:] += np.clip(desired - data.ctrl, -.01, .01)
            assert np.all(data.ctrl >= model.actuator_ctrlrange[:, 0]) and np.all(data.ctrl <= model.actuator_ctrlrange[:, 1])
        controls.append(data.ctrl.copy())
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        mujoco.mj_step(model, data)
        peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        elapsed = (step + 1) * model.opt.timestep
        assert abs(data.time - initial_time - elapsed) < 1e-8
        row = physical_metrics(model, data)
        row.update(elapsed_s=float(elapsed), com_displacement_m=float(np.linalg.norm(np.array(row["com_xy"]) - initial["com_xy"])),
                   velocity_max=float(max(abs(data.qvel))))
        peak = max(peak, row["penetration"]["depth_m"])
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        failure = guard(model, data, row, peak)
        if failure:
            break
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls,
                        initial_physics_state=saved["initial_physics_state"], initial_ctrl=initial_ctrl, dt=model.opt.timestep)
    return {"amplitude": amplitude, "completed": failure is None and len(rows) == total_steps, "failure": failure,
            "duration_s": len(rows) * model.opt.timestep, "initial_metrics": initial, "peak_penetration_m": peak,
            "min_left_load_n": min(row["loads_n"][0] for row in rows), "final_left_load_n": rows[-1]["loads_n"][0],
            "trace": rows, "trajectory": str(output), "trajectory_sha256": digest(output)}


def compare(result, baseline):
    tail = [index for index, row in enumerate(result["trace"]) if row["elapsed_s"] > .8]
    if not result["completed"] or not baseline["completed"] or not tail:
        return {"sustained_reduction_passed": False, "tail_reduction_fraction": None}
    original = np.mean([baseline["trace"][index]["loads_n"][0] for index in tail])
    current = np.mean([result["trace"][index]["loads_n"][0] for index in tail])
    reduction = float(1 - current / original)
    return {"sustained_reduction_passed": reduction >= .2, "tail_reduction_fraction": reduction}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_report", type=Path)
    args = parser.parse_args()
    chain = json.loads(args.staged_report.read_text())
    assert chain["status"] == "completed"
    for unused in range(2):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    margin = chain
    assert margin["optimization"]["success"]
    for unused in range(3):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    initial = next(entry for entry in chain["seeds"] if entry["seed"] == "old_equilibrium")
    assert initial["hold"]["reached"]
    trajectory = initial["hold"]["trajectory"]
    assert digest(trajectory) == initial["hold"]["trajectory_sha256"]
    path = Path(margin["model"])
    assert digest(path) == margin["model_sha256"]
    assert digest(path.with_name("manifest.json")) == margin["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(path))
    destination = ROOT / "output/rl/component_unload_ramps" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.staged_report.resolve()), "source_report_sha256": digest(args.staged_report),
              "source_trajectory": trajectory, "source_trajectory_sha256": digest(trajectory), "model": str(path), "model_sha256": digest(path),
              "manifest_sha256": digest(path.with_name("manifest.json")), "candidate_ctrl": margin["candidate_ctrl"],
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "load_transfer.py", "diagnose_short_crawl.py")},
              "protocol": {"amplitudes": AMPLITUDES, "ramp_s": .6, "duration_s": 1, "control_period_s": .02,
                           "max_ctrl_delta_rad": .01, "sustained_reduction_fraction": .2, "comparison_tail_s": .2,
                           "instantaneous_torque_ratio_limit": .95, "joint_range_tolerance_rad": .001,
                           "safety_check_period_s": model.opt.timestep, "optimization": False, "restarts": 0},
              "limitations": "Four fixed motor-driven probes, not trajectory optimization. Same full physics initial state; no resets inside rollouts. LP control is a direction only, never applied as an instantaneous full jump. Reduction is not clearance or a step; prior contact sensitivity failure retained.", "runs": []}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with np.load(trajectory, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["initial_ctrl"], margin["baseline_ctrl"])
            np.testing.assert_array_equal(saved["qpos"][0], margin["qpos"])
            for amplitude in AMPLITUDES:
                result = run(model, saved, np.array(margin["candidate_ctrl"]), amplitude, destination / f"amplitude_{amplitude}.npz")
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
