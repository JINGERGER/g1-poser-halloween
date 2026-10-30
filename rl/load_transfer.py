import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnose_short_crawl import observe
from prepare import ROOT, SITES, SUPPORTS
from reference_gait import paired_cycles
from support_handoff import pulse


def smooth(fraction):
    fraction = np.clip(fraction, 0, 1)
    return float(6 * fraction ** 5 - 15 * fraction ** 4 + 10 * fraction ** 3)


def bounded_vector(vector, maximum):
    vector = np.asarray(vector, dtype=float)
    return vector * min(1, maximum / max(float(np.linalg.norm(vector)), 1e-12))


def unloaded(loads):
    loads = np.asarray(loads)
    return bool(loads[3] <= 5 and np.all(loads[:3] > 5))


def posture(model, scratch, anchor, sites, shift, lift):
    scratch.qpos[:7] = anchor[:7]
    scratch.qpos[:2] += shift
    targets = sites.copy()
    targets[3, 2] += lift
    jacobian = np.zeros((3, model.nv))
    lower = np.maximum(model.jnt_range[1:, 0], anchor[7:] - 0.25)
    upper = np.minimum(model.jnt_range[1:, 1], anchor[7:] + 0.25)
    for iteration in range(15):
        mujoco.mj_forward(model, scratch)
        matrices, errors = [], []
        for index, name in enumerate(SITES):
            mujoco.mj_jacSite(model, scratch, jacobian, None, model.site(name).id)
            matrices.append(jacobian[:, 6:].copy())
            errors.append(targets[index] - scratch.site(name).xpos)
        matrix = np.vstack(matrices)
        error = np.concatenate(errors)
        delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + np.eye(12) * 1e-5, error)
        scratch.qpos[7:] = np.clip(scratch.qpos[7:] + np.clip(delta, -0.02, 0.02), lower, upper)
    mujoco.mj_forward(model, scratch)
    residual = float(np.max(np.linalg.norm(np.array([scratch.site(name).xpos for name in SITES]) - targets, axis=1)))
    return scratch.qpos[7:] - anchor[7:], residual


def physical_metrics(model, data):
    row = observe(model, data, data.ctrl)
    floor = model.geom("floor").id
    support_ids = [model.geom(name).id for name in SUPPORTS]
    contacts = [False] * 4
    weighted_slip, load = 0.0, 0.0
    for index, contact in enumerate(data.contact):
        if floor not in (contact.geom1, contact.geom2):
            continue
        geom = int(contact.geom2 if contact.geom1 == floor else contact.geom1)
        if geom not in support_ids:
            continue
        contacts[support_ids.index(geom)] = True
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, contact.pos, int(model.geom_bodyid[geom]))
        normal = max(0, force[0])
        load += normal
        weighted_slip += normal * float(np.linalg.norm((jacobian @ data.qvel)[:2]))
    row.update(floor_contacts=contacts, loaded_slip_mps=weighted_slip / load if load > 1 else None,
               com_xy=data.subtree_com[1, :2].tolist(), base_xy=data.qpos[:2].tolist())
    return row


def trial(entry, branch, destination):
    model_path = Path(entry["model"])
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != entry["model_sha256"]:
        raise ValueError("Support model changed")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    anchor = entry["anchor"]
    mujoco.mj_setState(model, data, np.array(anchor["physical_state"], dtype=float), mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = anchor["previous_ctrl"]
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    scratch.qpos[:] = initial
    initial_ctrl = data.ctrl.copy()
    sites = np.array([data.site(name).xpos.copy() for name in SITES])
    initial_com = data.subtree_com[1, :2].copy()
    remaining_center = sites[:3, :2].mean(axis=0)
    desired_shift = bounded_vector(remaining_center - initial_com, 0.08)
    initial_row = physical_metrics(model, data)
    if not anchor["ready"] or not all(load > 1 for load in initial_row["loads_n"]) or initial_row["other_total_n"] > 1:
        raise ValueError("Anchor is not load bearing")
    rows, qpos, qvel, controls = [], [data.qpos.copy()], [data.qvel.copy()], []
    start_time = float(data.time)
    armed_at = None
    unloaded_samples = 0
    failure = None
    substep_peak_ratio = 0.0
    substep_penetration = 0.0
    clipping = 0
    previous = initial_row
    joint_ids = model.actuator_trnid[:, 0]
    limits = model.jnt_actfrcrange[joint_ids]
    for step in range(375):
        elapsed = step * 0.02
        shift = np.zeros(2)
        lift = 0.0
        residual = 0.0
        if branch == "transfer":
            unloaded_samples = unloaded_samples + 1 if unloaded(previous["loads_n"]) else 0
            if armed_at is None and 0.5 <= elapsed <= 4.0 and unloaded_samples >= 5:
                armed_at = elapsed
            envelope = smooth((elapsed - 0.5) / 3.0)
            if armed_at is None and elapsed > 4.5:
                envelope *= 1 - smooth((elapsed - 4.5) / 2.0)
            elif armed_at is not None:
                lift = 0.02 * pulse(elapsed - armed_at)
                if elapsed - armed_at < 1.75 and not unloaded(previous["loads_n"]):
                    failure = "unload_lost_before_or_during_lift"
                    break
                if lift > 0 and not np.all(np.asarray(previous["loads_n"][:3]) > 5):
                    failure = "remaining_support_lost_during_swing"
                    break
                if elapsed > armed_at + 3:
                    envelope *= 1 - smooth((elapsed - armed_at - 3) / 1.0)
            planned = desired_shift * envelope
            feedback = bounded_vector(0.3 * (initial_com + planned - data.subtree_com[1, :2]), 0.02) * envelope
            shift = bounded_vector(planned + feedback, 0.08)
            offset, residual = posture(model, scratch, initial, sites, shift, lift)
            if residual > 0.005:
                failure = "kinematic_residual_over_5mm"
                break
            desired = initial_ctrl + offset
        else:
            desired = initial_ctrl.copy()
        target = np.clip(desired, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        clipping += int(np.count_nonzero(desired != target))
        data.ctrl[:] = data.ctrl + np.clip(target - data.ctrl, -0.01, 0.01)
        controls.append(data.ctrl.copy())
        for substep in range(round(0.02 / model.opt.timestep)):
            if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise AssertionError("External assistance is forbidden")
            mujoco.mj_step(model, data)
            forces = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
            substep_peak_ratio = max(substep_peak_ratio, float(np.max(np.abs(forces) / np.where(forces >= 0, limits[:, 1], -limits[:, 0]))))
            substep_penetration = max(substep_penetration, max((-float(contact.dist) for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time - start_time - (step + 1) * 0.02) > 1e-7:
            raise AssertionError("Invalid physics state or implicit reset")
        row = physical_metrics(model, data)
        substep_penetration = max(substep_penetration, row["penetration"]["depth_m"])
        row.update(elapsed_s=(step + 1) * 0.02, commanded_root_shift=shift.tolist(), requested_site_lift_m=lift, ik_residual_m=residual)
        rows.append(row)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        previous = row
        if row["other_total_n"] > 1:
            failure = "non_support_load_over_1N"
        elif substep_penetration > 0.005:
            failure = "penetration_over_5mm"
        elif data.qpos[2] < 0.16:
            failure = "base_below_16cm"
        if failure:
            break
    protocol = json.loads(Path(__file__).with_name("reference_eval.json").read_text())
    cycles = paired_cycles(rows, protocol, 0.02)
    np.savez_compressed(destination, qpos=np.array(qpos), qvel=np.array(qvel), ctrl=np.array(controls), dt=0.02)
    return {"branch": branch, "model": str(model_path), "model_sha256": entry["model_sha256"],
            "anchor_hash": anchor["paired_state_sha256"], "initial_physics_state": anchor["physical_state"],
            "initial_ctrl": initial_ctrl.tolist(), "initial_metrics": initial_row,
            "unclipped_center_shift_m": (remaining_center - initial_com).tolist(), "bounded_shift_m": desired_shift.tolist(),
            "duration_s": len(rows) * 0.02, "failure": failure, "unload_trigger_s": armed_at,
            "min_right_foot_load_n": min((row["loads_n"][3] for row in rows), default=None),
            "max_right_foot_gap_m": max((row["gaps_m"][3] for row in rows), default=None),
            "actual_com_displacement_m": (np.array(rows[-1]["com_xy"]) - initial_com).tolist() if rows else None,
            "mean_loaded_slip_mps": float(np.mean([row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None])) if rows else None,
            "cycles": cycles, "substep_peak_torque_ratio": substep_peak_ratio,
            "substep_max_penetration_m": substep_penetration, "clipped_ctrl_components": clipping,
            "completed_safe_lift": bool(branch == "transfer" and not failure and len(rows) == 375 and cycles[3]["count"] >= 1),
            "trace": rows, "trajectory": str(destination)}


def main():
    parser = argparse.ArgumentParser(description="Bounded support-load transfer probe, not a general crawl controller")
    parser.add_argument("support_report", type=Path)
    args = parser.parse_args()
    support = json.loads(args.support_report.read_text())
    if support["status"] != "completed":
        raise ValueError("Expected a completed support experiment")
    entries = [row for row in support["runs"] if row["branch"] == "freeze"]
    if len(entries) != 2:
        raise ValueError("Require both training-seed anchors")
    destination = ROOT / "output/rl/load_transfer" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_updates": 0,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "support_report_sha256": hashlib.sha256(args.support_report.read_bytes()).hexdigest(),
              "limitations": "Two fixed anchors, one bounded transfer direction each, no parameter search. Three-support site centroid is a geometric heuristic, not friction-cone/dynamic feasibility proof. Virtual root translation exists only in the IK scratch model; live physics receives bounded ctrl only. COM feedback plus actual load-triggered 2cm site pulse is not force-optimized whole-body control. Safety uses substep penetration and endpoint non-support load; safe-lift flag is nominal only, not crawling/robustness/hardware acceptance."}
    try:
        for index, entry in enumerate(entries):
            for branch in ("freeze", "transfer"):
                result = trial(entry, branch, destination / f"seed{index + 1}_{branch}.npz")
                result["training_seed"] = index + 1
                report["runs"].append(result)
                (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: value for key, value in result.items() if key not in ("trace", "initial_metrics", "initial_physics_state", "initial_ctrl")}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
