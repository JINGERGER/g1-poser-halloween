import json
from datetime import datetime

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import PositionTorqueGuard, InfeasiblePositionTarget
from recovery_transition_task import RecoveryTransitionTask
from reference_policy_baseline import ROOT
from stand_transition_task import UnsafeState
from train_near_handoff import ActorCritic


class PreparedArmGuard(PositionTorqueGuard):
    def __init__(self, model):
        super().__init__(model, fraction=.8)
        self.names = [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]]

    def apply(self, data, requested):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        target = requested.copy()
        if rotation.reshape(3, 3)[0, 2] < .8:
            left = self.names.index("left_shoulder_roll_joint")
            right = self.names.index("right_shoulder_roll_joint")
            target[left], target[right] = max(target[left], .9), min(target[right], -.9)
            for side in ("left", "right"):
                index = self.names.index(f"{side}_shoulder_pitch_joint")
                target[index] = min(target[index], .3)
        return super().apply(data, target)


def main():
    torch.set_num_threads(1)
    source = ROOT / "output/rl/closed_loop_transition/2026-10-23_16-27-00-393302/report.json"
    original = json.loads(source.read_text())
    checkpoint = ROOT / original["checkpoint"]
    assert digest(checkpoint) == original["checkpoint_sha256"]
    network = ActorCritic(139)
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["network"])
    output = ROOT / "output/rl/prepared_arm_transition" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(__file__), "training_report_sha256": digest(source),
              "protocol": "Two fixed strict nominal standing tests: native PD opens arms over .6s, then final event-policy. "
              "Preparation target: shoulder rolls +/-1rad, shoulder pitches -.3rad, all other standing targets unchanged. "
              "Second variant retains shoulder rolls >=.9rad outward and pitch <=.3rad until body-up world-x >=.8. "
              "Torque guard remains .8 predicted fraction. No state/time reset at policy start; integrated target "
              "exposes actual final preparation control. Full original physical gates and final crawl checks remain. "
              "This is an untrained event-controller hypothesis, not validated human demonstration reconstruction.", "cases": []}
    path = output / "report.json"
    try:
        for retain in (False, True):
            task = RecoveryTransitionTask()
            state = task.make_states([0])[0]
            state["qvel"] = np.zeros(task.model.nv).tolist()
            task.reset(state)
            prepared = task.target.copy()
            for index, name in enumerate(task.names):
                if name.endswith("shoulder_roll_joint"):
                    prepared[index] = 1. if name.startswith("left") else -1.
                elif name.endswith("shoulder_pitch_joint"):
                    prepared[index] = -.3
            result = None
            preparation_passed = False
            try:
                for step in range(30):
                    phase = (step + 1) / 30.
                    blend = phase * phase * (3. - 2. * phase)
                    task.guard.advance(task.execution, task.target + blend * (prepared - task.target))
                preparation_passed = True
                task.integrated = task.data.ctrl.copy()
                if retain:
                    task.guard = PreparedArmGuard(task.model)
                observation = task.observation()
                for unused in range(task.horizon):
                    with torch.no_grad():
                        action = network.actor(torch.from_numpy(observation)).numpy()
                    observation, reward, done, result = task.step(action)
                    if done:
                        break
                assert done
            except (UnsafeState, InfeasiblePositionTarget) as error:
                result = {"success": False, "failure": task.failure or [type(error).__name__], **task.progress()}
            trajectory = output / f"retain_{retain}.npz"
            task.save_and_replay(trajectory)
            case = {"retain_until_forward": retain, "preparation_passed": preparation_passed, **result,
                    "trajectory": str(trajectory.relative_to(ROOT)), "trajectory_sha256": digest(trajectory), "replay_passed": True}
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
