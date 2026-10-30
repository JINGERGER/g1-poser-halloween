import hashlib
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_low_support_transition import target_at
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState


def readiness_details(task):
    row, unused, unused_penetrations = task.measure()
    window = np.asarray(task.samples[-51:])
    enough = len(window) == 51
    linear = float(np.linalg.norm(task.data.qvel[:3]))
    angular = float(np.linalg.norm(task.data.qvel[3:6]))
    mixed = float(np.linalg.norm(task.data.qvel[:6]))
    rms = float(np.sqrt(np.mean((task.data.qpos[task.addresses] - task.crawl_target) ** 2)))
    checks = {"history_100ms": enough, "four_loads_5n_for_window": bool(enough and np.all(window[:, 4:8] > 5.)),
              "height_for_window": bool(enough and np.all(window[:, 3] <= .4)),
              "mixed_velocity_below_0_5": mixed < .5, "joint_rms_below_0_35": rms < .35}
    assert all(checks.values()) == task.ready()
    return {"ready": task.ready(), "checks": checks, "linear_speed_m_s": linear,
            "angular_speed_rad_s": angular, "mixed_velocity_norm": mixed, "joint_rms_rad": rms,
            "loads_n": row[4:8], "gaps_m": row[8:12], "height_m": row[3]}


def main():
    torch.set_num_threads(1)
    source = ROOT / "output/rl/low_support_transition/2026-10-23_14-46-00-951505/report.json"
    original = json.loads(source.read_text())
    nodes = original["cases"][0]["nodes"]
    assert original["cases"][0]["geometry_screen_passed"]
    task = StandTransitionTask()
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    assert assets == original["assets"]
    assert digest(task.policy_path) == original["policy_sha256"]
    assert digest(task.bundle_path) == original["bundle_sha256"]
    state = {"qpos": nodes[-1]["qpos"], "qvel": np.zeros(task.model.nv).tolist()}
    task.target = np.array(state["qpos"])[task.addresses].copy()
    output = ROOT / "output/rl/handoff_window" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(source), "assets": assets,
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "check_low_support_transition.py", "stand_transition_task.py",
                           "near_handoff_task.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "Fixed low-support initial pose, zero velocity. Native-PD reference prefix of 0/40/80/120/200ms. "
              "Paired direct handoff versus continuing same reference until original readiness (maximum2s). "
              "Identical integration-state hashes at each branch, full per2ms strict gates and exact trajectory replay. "
              "Each trial continuous from its low-support reset; not a standing transition or full basin estimate. "
              "Readiness bypass only in diagnostic direct branches; no production gate/model changes.", "cases": []}
    path = output / "report.json"
    try:
        for prefix_steps in (0, 2, 4, 6, 10):
            prefix_hash = None
            for mode in ("direct", "wait"):
                task.reset(state)
                for step in range(prefix_steps):
                    task.execution.transition_step(target_at(nodes, task.addresses, (step + 1) * .02))
                specification = mujoco.mjtState.mjSTATE_INTEGRATION
                snapshot = np.empty(mujoco.mj_stateSize(task.model, specification))
                mujoco.mj_getState(task.model, task.data, snapshot, specification)
                current_hash = hashlib.sha256(snapshot.tobytes()).hexdigest()
                if prefix_hash is not None:
                    assert current_hash == prefix_hash
                prefix_hash = current_hash
                before = readiness_details(task)
                result = None
                try:
                    if mode == "wait":
                        for step in range(prefix_steps, 100):
                            if task.ready():
                                break
                            task.execution.transition_step(target_at(nodes, task.addresses, (step + 1) * .02))
                    if mode == "direct" or task.ready():
                        at_handoff = readiness_details(task)
                        handoff_time = float(task.data.time)
                        result = task.finish()
                except UnsafeState:
                    pass
                if result is None:
                    at_handoff, handoff_time = None, None
                    result = {"success": False, "failure": task.failure or ["no_readiness_after_2s"]}
                trajectory = output / f"prefix_{prefix_steps}_{mode}.npz"
                task.save_and_replay(trajectory)
                entry = {"prefix_s": prefix_steps * .02, "mode": mode, "prefix_state_sha256": prefix_hash,
                         "before": before, "at_handoff": at_handoff, "handoff_time_s": handoff_time,
                         "result": result, "trajectory": str(trajectory.relative_to(ROOT)),
                         "trajectory_sha256": digest(trajectory), "replay_passed": True}
                report["cases"].append(entry)
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"prefix_s": entry["prefix_s"], "mode": mode, "ready": before["ready"],
                                  "handoff_s": handoff_time, "result": result}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
