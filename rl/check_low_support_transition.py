import json
from datetime import datetime

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState


def solve_reference(task, rise, unpitch, hand_lift):
    model = task.model
    probe = mujoco.MjData(model)
    probe.qpos[:] = task.nominal
    mujoco.mj_forward(model, probe)
    support_goal = probe.geom_xpos[task.supports].copy()
    foot_rotations = probe.geom_xmat[task.supports[2:]].copy()
    nodes = []
    angles = task.crawl_target.copy()
    for phase in np.linspace(0., 1., 11):
        base = task.nominal[:7].copy()
        base[2] += rise * phase
        rotation = np.array([np.cos(unpitch * phase / 2.), 0., -np.sin(unpitch * phase / 2.), 0.])
        mujoco.mju_mulQuat(base[3:7], rotation, task.nominal[3:7])
        goals = support_goal.copy()
        goals[:2, 2] += hand_lift * phase

        def residual(values):
            probe.qpos[:7] = base
            probe.qpos[task.addresses] = values
            mujoco.mj_forward(model, probe)
            errors = probe.geom_xpos[task.supports] - goals
            errors[:2] *= 100.
            errors[2:, :2] *= 10.
            errors[2:, 2] *= 100.
            orientation = (probe.geom_xmat[task.supports[2:]] - foot_rotations).ravel()
            return np.r_[errors.ravel(), orientation * 2., .03 * (values - task.crawl_target)]

        fit = least_squares(residual, angles, bounds=(model.jnt_range[task.joints, 0], model.jnt_range[task.joints, 1]),
                            max_nfev=150, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        angles = fit.x.copy()
        residual(angles)
        gaps = [float(mujoco.mj_geomDistance(model, probe, geom, task.floor, 2., None)) for geom in task.supports]
        penetration = max([0.] + [-float(contact.dist) for contact in probe.contact])
        self_penetration = max([0.] + [-float(contact.dist) for contact in probe.contact
                                     if task.floor not in (contact.geom1, contact.geom2)])
        nodes.append({"phase": float(phase), "qpos": probe.qpos.tolist(), "fit_success": bool(fit.success),
                      "hand_error_m": float(np.max(np.linalg.norm(probe.geom_xpos[task.supports[:2]] - goals[:2], axis=1))),
                      "foot_xy_shift_m": float(np.max(np.linalg.norm((probe.geom_xpos[task.supports[2:]] - goals[2:])[:, :2], axis=1))),
                      "support_gaps_m": gaps, "penetration_m": penetration, "self_penetration_m": self_penetration})
    return nodes


def target_at(nodes, addresses, elapsed, duration=.8):
    if not np.isfinite(duration) or duration <= 0.:
        raise ValueError("Duration must be finite and positive")
    phase = np.clip(elapsed / duration, 0., 1.)
    coordinate = (1. - phase * phase * (3. - 2. * phase)) * (len(nodes) - 1)
    lower = min(int(coordinate), len(nodes) - 2)
    fraction = coordinate - lower
    poses = np.array([node["qpos"] for node in nodes])
    return ((1. - fraction) * poses[lower] + fraction * poses[lower + 1])[addresses]


def execute(task, nodes, output, name, mode="reference"):
    if mode not in ("reference", "hold", "direct"):
        raise ValueError("Unknown execution mode")
    state = {"qpos": nodes[-1]["qpos"], "qvel": np.zeros(task.model.nv).tolist()}
    task.target = np.array(state["qpos"])[task.addresses].copy()
    try:
        task.reset(state)
    except ValueError as error:
        return {"success": False, "executed": False, "reason": str(error)}
    if mode == "direct":
        result = task.finish()
        path = output / f"{name}.npz"
        task.save_and_replay(path)
        return {"executed": True, "mode": mode, **result, "trajectory": str(path.relative_to(ROOT)),
                "trajectory_sha256": digest(path), "replay_passed": True}
    failures = []
    result = None
    for step in range(100):
        try:
            target = task.target if mode == "hold" else target_at(nodes, task.addresses, (step + 1) * .02)
            task.execution.transition_step(target)
        except UnsafeState:
            failures = task.failure.copy()
            break
        if task.ready():
            result = task.finish()
            break
    transition = np.array(task.phases) == 0
    samples = np.array(task.samples)[transition]
    hands = samples[:, 4:6] > 5.
    first = [float(samples[np.flatnonzero(hands[:, side])[0], 0]) if hands[:, side].any() else None for side in range(2)]
    consecutive, longest = 0, 0
    for loaded in hands.all(axis=1):
        consecutive = consecutive + 1 if loaded else 0
        longest = max(longest, consecutive)
    path = output / f"{name}.npz"
    task.save_and_replay(path)
    return {"executed": True, "mode": mode, "success": bool(result and result["success"]),
            "failure": failures if failures else ([] if result else ["no_readiness_after_2s"]),
            "handoff": result, "first_hand_load_s": first,
            "hand_load_time_difference_s": abs(first[0] - first[1]) if None not in first else None,
            "both_hands_loaded_s": max(0, longest - 1) * .002,
            "min_mean_hand_gap_m": float(samples[:, 8:10].mean(axis=1).min()),
            "peak_torque_ratio": float(samples[:, 13].max()),
            "transition_end_time_s": float(samples[-1, 0]),
            "trajectory": str(path.relative_to(ROOT)), "trajectory_sha256": digest(path), "replay_passed": True}


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    output = ROOT / "output/rl/low_support_transition" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "protocol": "Three predeclared low-support starts: pelvis rise 5/10/15cm, "
              "world-y unpitch .2/.4/.6rad, both hands lifted 4/8/12cm. Eleven IK nodes from crawl outward. "
              "Soft foot xy/orientation goals, no COM support-polygon constraint. "
              "Reverse joint reference for .8s, then hold up to 2s total; free base evolves only through physics. "
              "Zero-velocity curriculum initialization, native PD and unmodified per-substep gates. "
              "Existing readiness then continuous 2s frozen crawl. Not a standing-start proof or reconstructed video.",
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("check_low_support_transition.py", "stand_transition_task.py", "near_handoff_task.py", "transition_execution.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path), "cases": []}
    report["control_protocol"] = "Post-hoc diagnostic controls added after the first reference result: " \
        "same initial state, hold initial joint target until readiness, or directly invoke frozen crawl. " \
        "No claim of incremental reference benefit without outperforming these controls."
    report_path = output / "report.json"
    try:
        for level in (1, 2, 3):
            nodes = solve_reference(task, .05 * level, .2 * level, .04 * level)
            case = {"level": level, "nodes": nodes}
            screened = all(node["fit_success"] and node["hand_error_m"] < .005
                           and node["penetration_m"] <= .005 and node["self_penetration_m"] <= .001
                           and max(abs(gap) for gap in node["support_gaps_m"][2:]) < .005 for node in nodes)
            case["geometry_screen_passed"] = screened
            case["result"] = execute(task, nodes, output, f"level_{level}") if screened else {
                "executed": False, "success": False, "reason": "IK nodes failed geometry screen"}
            if screened:
                case["controls"] = {mode: execute(task, nodes, output, f"level_{level}_{mode}", mode)
                                    for mode in ("hold", "direct")}
            report["cases"].append(case)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"level": level, "geometry": screened, "result": case["result"]}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
