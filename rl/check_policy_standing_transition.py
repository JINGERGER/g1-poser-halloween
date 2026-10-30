import json
from datetime import datetime

import numpy as np
import mujoco
import torch

from artifact_utils import digest
from crawl_control import CrawlController
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from position_torque_guard import PositionTorqueGuard, InfeasiblePositionTarget


def run_case(name, command, shaped, blend, output, guarded=False):
    task = StandTransitionTask()
    mapping = {task.reference.joint(int(joint)).name: index for index, joint in enumerate(task.reference.actuator_trnid[:, 0])}
    for actuator, joint in enumerate(task.joints):
        joint_name = task.model.joint(int(joint)).name
        if joint_name in mapping:
            source = mapping[joint_name]
            task.native_gain[actuator, 0] = task.reference.actuator_gainprm[source, 0] * 1.4
            task.native_bias[actuator, 1:3] = task.reference.actuator_biasprm[source, 1:3] * 1.4
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    guard = PositionTorqueGuard(task.model) if guarded else None
    controller = CrawlController(task.model, task.constants, task.policy, command=command,
                                 ankle_min_target=-.2 if shaped else None, foot_clearance_boost=.08 if shaped else 0.)
    result = None
    for step in range(300):
        controller.update(task.data)
        fraction = min(1., (step + 1) * .02 / blend) if blend else 1.
        target = task.target + fraction * (task.data.ctrl.copy() - task.target)
        try:
            if guard is None:
                task.execution.transition_step(target)
            else:
                for unused in range(10):
                    if task.data.qfrc_applied.any() or task.data.xfrc_applied.any():
                        raise ValueError("Assistance forces prohibited")
                    task.data.ctrl[:] = guard.apply(task.data, target)
                    mujoco.mj_step(task.model, task.data)
                    mujoco.mj_forward(task.model, task.data)
                    if not np.isfinite(task.data.qpos).all() or not np.isfinite(task.data.qvel).all():
                        raise ValueError("Nonfinite state")
                    task._observe_step(task.model, task.data)
        except InfeasiblePositionTarget:
            task.failure = ["infeasible_guard_target"]
            break
        except UnsafeState:
            break
        if task.ready():
            result = task.finish()
            break
    progress = task.progress()
    path = output / f"{name}.npz"
    task.save_and_replay(path)
    return {"name": name, "command": command, "crawl_shaping": shaped, "blend_s": blend, "torque_guard": guarded,
            "success": bool(result and result["success"]), "handoff": result,
            "failure": task.failure or ([] if result and result["success"] else ["no_successful_handoff"]),
            **progress, "trajectory": str(path.relative_to(ROOT)), "sha256": digest(path), "replay_passed": True}


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    output = ROOT / "output/rl/policy_standing_transition" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "protocol": "Four fixed closed-loop pretrained-policy transfer controls, true standing "
              "zero velocity, reference PD x1.4 (unmatched wrists native), preserved actuator force limits. "
              "Transition height ceiling .95m as existing standing task; all other per-substep gates unchanged. "
              "Crawl shaping on/off, command .5/0, optional .25s target blend from standing. "
              "Policy history retained during transition. Existing readiness then frozen shaped crawl for 2s. "
              "No training or state overrides. Same modified robot assets, not upstream original-model validation.",
              "source_sha256": digest(__file__), "guard_sha256": digest(ROOT / "rl/position_torque_guard.py"), "policy_sha256": digest(task.policy_path),
              "bundle_sha256": digest(task.bundle_path),
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}, "cases": []}
    path = output / "report.json"
    report["additional_controls"] = "Two raw pretrained closed-loop controls (command 0/.5) use a per-2ms " \
        "position-target projection into 80% predicted force limits, with no force-limit or physical gate changes. " \
        "This constrains instantaneous PD demand, not a continuous-time guarantee. No guard after frozen-crawl handoff."
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for config in (("frozen", .5, True, 0.), ("raw_forward", .5, False, 0.),
                       ("raw_zero", 0., False, 0.), ("raw_zero_blend", 0., False, .25)):
            result = run_case(*config, output)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for command in (0., .5):
            result = run_case(f"guarded_raw_{command}", command, False, 0., output, guarded=True)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
