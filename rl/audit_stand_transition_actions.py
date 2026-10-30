import json
from datetime import datetime

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState


def reference_target(start, goal, names, elapsed, duration, route):
    if not np.isfinite(duration) or duration <= 0.:
        raise ValueError("Duration must be finite and positive")
    if route not in ("together", "legs_lead", "arms_lead"):
        raise ValueError("Unknown reference route")
    phase = np.clip(elapsed / duration, 0., 1.)
    if route == "together":
        return start + (goal - start) * phase * phase * (3. - 2. * phase)
    arms = np.array([any(part in name for part in ("shoulder", "elbow", "wrist")) for name in names])
    leading = arms if route == "arms_lead" else ~arms
    fraction = np.where(leading, .75, .25)
    middle = start + fraction * (goal - start)
    local = np.clip(phase * 2. if phase <= .5 else (phase - .5) * 2., 0., 1.)
    blend = local * local * (3. - 2. * local)
    return start + blend * (middle - start) if phase <= .5 else middle + blend * (goal - middle)


def action_audit(task):
    names = [task.model.joint(int(joint)).name for joint in task.joints]
    differences = np.abs(task.crawl_target - task.target)
    rng = np.random.default_rng(20261023)
    offsets = np.zeros((2048, 23))
    for unused in range(300):
        offsets += .02 * np.tanh(rng.normal(0., np.exp(-2.3), offsets.shape))
    return {
        "joint_endpoints": [{"joint": name, "standing_rad": float(start), "crawl_rad": float(goal),
                             "difference_rad": float(delta), "kp": float(task.model.actuator_gainprm[index, 0]),
                             "kd": float(-task.model.actuator_biasprm[index, 2]),
                             "joint_force_range_nm": task.model.jnt_actfrcrange[task.joints[index]].tolist()}
                            for index, (name, start, goal, delta) in enumerate(zip(names, task.target, task.crawl_target, differences))],
        "integrated_target_rate_limit_rad_s": 1.,
        "exact_endpoint_integrator_min_time_s": float(differences.max()),
        "exact_endpoint_smoothstep_min_time_s_at_1rad_s": float(1.5 * differences.max()),
        "initial_raw_gaussian_std": float(np.exp(-2.3)),
        "unclipped_zero_mean_random_walk_6s_coordinate_rms_rad": float(np.sqrt(np.mean(offsets ** 2))),
        "unclipped_zero_mean_random_walk_6s_coordinate_abs_p99_rad": float(np.quantile(np.abs(offsets), .99)),
        "limitations": "Random-walk diagnostic excludes joint clipping and lowering seed, not a learned policy rollout. "
                        "Endpoint time is a bound on commanded targets, not actual joint motion or necessary handoff time. "
                        "Lowering seed adds target velocity; model torque limits are not a hardware speed/thermal envelope."}


def run_candidate(task, state, duration, route, output):
    task.reset(state)
    names = [task.model.joint(int(joint)).name for joint in task.joints]
    previous = task.target.copy()
    peak_rate, peak_tracking = 0., 0.
    result = None
    for step in range(round((duration + .5) / .02)):
        target = reference_target(task.target, task.crawl_target, names, (step + 1) * .02, duration, route)
        peak_rate = max(peak_rate, float(np.max(np.abs(target - previous)) / .02))
        previous = target
        try:
            task.execution.transition_step(target)
        except UnsafeState:
            result = {"success": False, "failure": task.failure.copy(), "phase": "transition"}
            break
        peak_tracking = max(peak_tracking, float(np.max(np.abs(task.data.qpos[task.addresses] - target))))
        if task.ready():
            result = {**task.finish(), "phase": "handoff"}
            break
    if result is None:
        result = {"success": False, "failure": ["reference_completed_without_readiness"], "phase": "transition"}
    samples = np.array(task.samples)
    transition = np.array(task.phases) == 0
    rows = samples[transition]
    hand_loads = rows[:, 4:6] > 5.
    contact_times = [float(rows[np.flatnonzero(hand_loads[:, side])[0], 0]) if hand_loads[:, side].any() else None
                     for side in range(2)]
    duration_loaded, longest = 0, 0
    for loaded in np.all(hand_loads, axis=1):
        duration_loaded = duration_loaded + 1 if loaded else 0
        longest = max(longest, duration_loaded)
    path = output / f"{route}_{duration:.2f}s.npz"
    floor_contacts = []
    for index, contact in enumerate(task.data.contact):
        if task.floor not in (contact.geom1, contact.geom2):
            continue
        geom = contact.geom2 if contact.geom1 == task.floor else contact.geom1
        wrench = np.zeros(6)
        mujoco.mj_contactForce(task.model, task.data, index, wrench)
        floor_contacts.append({"geom": task.model.geom(geom).name,
                               "body": task.model.body(int(task.model.geom_bodyid[geom])).name,
                               "support": geom in task.supports, "normal_force_n": float(wrench[0]),
                               "distance_m": float(contact.dist)})
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, task.data.qpos[3:7])
    task.save_and_replay(path)
    return {"route": route, "reference_duration_s": duration, **result,
            "end_time_s": float(task.data.time), "min_transition_height_m": float(rows[:, 3].min()),
            "min_mean_hand_gap_m": float(rows[:, 8:10].mean(axis=1).min()),
            "first_hand_load_times_s": contact_times,
            "hand_load_time_difference_s": abs(contact_times[0] - contact_times[1]) if None not in contact_times else None,
            "both_hands_loaded_s": max(0, longest - 1) * .002,
            "peak_transition_hand_load_n": rows[:, 4:6].max(axis=0).tolist(),
            "peak_transition_torque_ratio": float(rows[:, 13].max()),
            "peak_penetration_m": np.max(task.penetrations, axis=0).tolist(),
            "final_floor_contacts": floor_contacts,
            "final_body_up_world": rotation.reshape(3, 3)[:, 2].tolist(),
            "max_executed_target_rate_rad_s": peak_rate,
            "max_completed_control_step_tracking_error_rad": peak_tracking,
            "trajectory": str(path.relative_to(ROOT)), "sha256": digest(path), "replay_passed": True}


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/stand_action_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "audit": action_audit(task), "initial_state": state,
              "protocol": "Predeclared 3 routes x durations 0.5, 1, 2.5, 5 seconds plus 0.5s settling. "
                          "Joint-space references between existing endpoints, NOT reconstructed video motion. "
                          "Native PD, original force limits and every-substep physical gates; stop on first failure. "
                          "One nominal standing start, no initial velocity, no base overrides or fixed feet. "
                          "Diagnostic references bypass the 1rad/s training integrator, not actuator limits. "
                          "Handoff only if existing readiness passes, followed by actual 2s frozen crawl. "
                          "Not a training run, safety certification or broad feasibility test.",
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("audit_stand_transition_actions.py", "stand_transition_task.py", "near_handoff_task.py",
                           "transition_execution.py", "train_near_handoff.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "bundle_sha256": digest(task.bundle_path), "policy_sha256": digest(task.policy_path),
              "cases": []}
    report_path = output / "report.json"
    try:
        for route in ("together", "legs_lead", "arms_lead"):
            for duration in (.5, 1., 2.5, 5.):
                result = run_candidate(task, state, duration, route, output)
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path)


if __name__ == "__main__":
    main()
