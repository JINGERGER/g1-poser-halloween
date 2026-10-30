import json
from datetime import datetime

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from artifact_utils import digest
from check_standing_endpoint import standing_pose
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def schedule(phase, route):
    if route == "coordinated":
        return smooth(phase), smooth(phase), smooth(phase)
    if route == "lean_first":
        return smooth(phase / .7), smooth((phase - .15) / .85), smooth(phase)
    if route == "lower_first":
        return smooth((phase - .15) / .85), smooth(phase / .7), smooth(phase)
    raise ValueError("Unknown full-body route")


def solve_route(task, route):
    model = task.model
    initial = standing_pose(model, .6).qpos.copy()
    probe = mujoco.MjData(model)
    probe.qpos[:] = initial
    mujoco.mj_forward(model, probe)
    support_start = probe.geom_xpos[task.supports].copy()
    probe.qpos[:] = task.nominal
    mujoco.mj_forward(model, probe)
    support_end = probe.geom_xpos[task.supports].copy()
    tangent = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1., initial, task.nominal)
    lower = np.r_[[-.5, -.5, -.65, -.3, -.1, -.3], model.jnt_range[task.joints, 0] - initial[task.addresses]]
    upper = np.r_[[.5, .5, .1, .3, 1.8, .3], model.jnt_range[task.joints, 1] - initial[task.addresses]]
    assert np.array_equal(task.addresses, np.arange(7, model.nq))
    nodes = [{"phase": 0., "qpos": initial.tolist(), "endpoint": True}]
    values = np.zeros(model.nv)
    for phase in np.linspace(0., 1., 21)[1:-1]:
        pitch, height, hands = schedule(phase, route)
        desired = tangent * smooth(phase)
        desired[2] = tangent[2] * height
        desired[4] = tangent[4] * pitch
        support_goal = support_start + smooth(phase) * (support_end - support_start)
        support_goal[:2] = support_start[:2] + hands * (support_end[:2] - support_start[:2])

        def residual(candidate):
            probe.qpos[:] = initial
            mujoco.mj_integratePos(model, probe.qpos, candidate, 1.)
            mujoco.mj_forward(model, probe)
            support_error = probe.geom_xpos[task.supports] - support_goal
            support_error[:2] *= 30.
            support_error[2:, :2] *= 5.
            support_error[2:, 2] *= 100.
            base_error = (candidate[:6] - desired[:6]) * [10., 10., 20., 10., 10., 10.]
            self_penetration = max([0.] + [-float(contact.dist) - .0005 for contact in probe.contact
                                          if task.floor not in (contact.geom1, contact.geom2)])
            other_penetration = max([0.] + [-float(contact.dist) for contact in probe.contact
                                           if task.floor in (contact.geom1, contact.geom2)
                                           and (contact.geom2 if contact.geom1 == task.floor else contact.geom1) not in task.supports])
            return np.r_[support_error.ravel(), base_error, .1 * (candidate[6:] - desired[6:]),
                         200. * self_penetration, 200. * other_penetration]

        result = least_squares(residual, values, bounds=(lower, upper), max_nfev=80)
        values = result.x.copy()
        residual(values)
        nodes.append({"phase": float(phase), "qpos": probe.qpos.tolist(), "fit_success": bool(result.success),
                      "cost": float(result.cost), "evaluations": int(result.nfev),
                      "hand_error_m": float(np.max(np.linalg.norm(probe.geom_xpos[task.supports[:2]] - support_goal[:2], axis=1))),
                      "self_penetration_m": max([0.] + [-float(contact.dist) for contact in probe.contact
                                                        if task.floor not in (contact.geom1, contact.geom2)])})
    nodes.append({"phase": 1., "qpos": task.nominal.tolist(), "endpoint": True})
    return nodes


def joint_target(nodes, addresses, phase):
    coordinate = float(np.clip(phase, 0., 1.)) * (len(nodes) - 1)
    index = min(int(coordinate), len(nodes) - 2)
    fraction = coordinate - index
    return ((1. - fraction) * np.array(nodes[index]["qpos"]) + fraction * np.array(nodes[index + 1]["qpos"]))[addresses]


def execute(task, nodes, output, route):
    state = {"qpos": nodes[0]["qpos"], "qvel": np.zeros(task.model.nv).tolist()}
    task.reset(state)
    result = None
    for step in range(150):
        try:
            task.execution.transition_step(joint_target(nodes, task.addresses, (step + 1) * .02 / 2.))
        except UnsafeState:
            break
        if task.ready():
            result = task.finish()
            break
    mask = np.array(task.phases) == 0
    rows = np.array(task.samples)[mask]
    quaternions = np.array(task.positions)[1:][mask, 3:7]
    body_forward = []
    rotation = np.zeros(9)
    for quaternion in quaternions:
        mujoco.mju_quat2Mat(rotation, quaternion)
        body_forward.append(float(rotation.reshape(3, 3)[0, 2]))
    contacts = []
    for index, contact in enumerate(task.data.contact):
        if task.floor in (contact.geom1, contact.geom2):
            geom = contact.geom2 if contact.geom1 == task.floor else contact.geom1
            wrench = np.zeros(6)
            mujoco.mj_contactForce(task.model, task.data, index, wrench)
            contacts.append({"body": task.model.body(int(task.model.geom_bodyid[geom])).name,
                             "normal_force_n": float(wrench[0]), "support": geom in task.supports})
    path = output / f"{route}.npz"
    task.save_and_replay(path)
    return {"success": bool(result and result["success"]), "handoff": result,
            "failure": task.failure or ([] if result else ["no_readiness_after_3s"]),
            "transition_time_s": float(rows[-1, 0]), "any_hand_loaded": bool(np.any(rows[:, 4:6] > 5.)),
            "min_mean_hand_gap_m": float(rows[:, 8:10].mean(axis=1).min()),
            "body_up_world_x_range": [min(body_forward), max(body_forward)],
            "final_floor_contacts": contacts, "replay_passed": True,
            "trajectory": str(path.relative_to(ROOT)), "trajectory_sha256": digest(path)}


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    output = ROOT / "output/rl/full_transition_routes" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "protocol": "Three full-start routes: coordinated, lean first, lower first. "
              "Joint and floating-base offline IK with moving hand/foot xy goals and soft base targets; no COM polygon constraint. "
              "19 fitted nodes plus exact standing/crawl endpoints. Fixed 2s reference plus 1s settling, "
              "native PD and unchanged per-2ms physical gates. No base actuation or state overrides during rollout. "
              "All routes executed regardless of IK convergence; dynamic failures stop immediately. "
              "Readiness triggers same-instance frozen crawl for 2s. No optimization from rollout outcomes or new RL training.",
              "limitations": "Three prescribed route families, not exhaustive search or hardware validation. "
              "IK contact penalties are nonsmooth local penalties, not continuous collision or dynamics guarantees.",
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("compare_full_transition_routes.py", "stand_transition_task.py", "near_handoff_task.py", "transition_execution.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "bundle_sha256": digest(task.bundle_path), "policy_sha256": digest(task.policy_path), "cases": []}
    path = output / "report.json"
    try:
        for route in ("coordinated", "lean_first", "lower_first"):
            nodes = solve_route(task, route)
            result = execute(task, nodes, output, route)
            report["cases"].append({"route": route, "nodes": nodes, "result": result})
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"route": route, **result}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
