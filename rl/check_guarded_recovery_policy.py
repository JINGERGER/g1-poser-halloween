import json
from datetime import datetime

import numpy as np
import torch

from arm_clearance_guard import ArmClearanceGuard
from artifact_utils import digest
from recovery_transition_task import RecoveryTransitionTask
from reference_policy_baseline import ROOT
from train_near_handoff import ActorCritic


def main():
    torch.set_num_threads(1)
    source = ROOT / "output/rl/closed_loop_transition/2026-10-23_16-27-00-393302/report.json"
    original = json.loads(source.read_text())
    checkpoint = ROOT / original["checkpoint"]
    assert digest(checkpoint) == original["checkpoint_sha256"]
    task = RecoveryTransitionTask()
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    if original["assets"] != assets or original["policy_sha256"] != digest(task.policy_path):
        raise ValueError("Evaluation assets differ from training assets")
    task.guard = ArmClearanceGuard(task.model)
    network = ActorCritic(139)
    network.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["network"])
    output = ROOT / "output/rl/guarded_recovery_policy" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"source_report_sha256": digest(source), "checkpoint_sha256": digest(checkpoint),
              "guard_sha256": digest(ROOT / "rl/arm_clearance_guard.py"),
              "protocol": "Strict nominal standing test of final event-policy checkpoint with arm/leg target-clearance projection. "
              "24 forearm/hand/elbow versus thigh/shin pairs, local signed-distance Jacobians, 2cm margin, "
              "six weighted halfspace projection passes per2ms, then unchanged 80% instantaneous PD torque guard. "
              "Only position commands change, no physical state/model overrides. "
              "Local kinematic projection is NOT a collision-free or dynamic safety guarantee; original strict gates remain. "
              "If nominal succeeds, evaluate fresh seeds2000..2007; otherwise stop this guard experiment.",
              "cases": [], "status": "running"}
    path = output / "report.json"
    try:
        for seed in [0, *range(2000, 2008)]:
            state = task.make_states([seed])[0]
            if seed == 0:
                state["qvel"] = np.zeros(task.model.nv).tolist()
            observation = task.reset(state)
            for unused in range(task.horizon):
                with torch.no_grad():
                    action = network.actor(torch.from_numpy(observation)).numpy()
                observation, reward, done, result = task.step(action)
                if done:
                    break
            assert done
            trajectory = output / f"seed_{seed}.npz"
            task.save_and_replay(trajectory)
            report["cases"].append({"seed": seed, **result, "trajectory": str(trajectory.relative_to(ROOT)),
                                    "trajectory_sha256": digest(trajectory), "replay_passed": True})
            print(json.dumps(report["cases"][-1]), flush=True)
            if seed == 0 and not result["success"]:
                break
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["projected_substeps"] = task.guard.projections
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
