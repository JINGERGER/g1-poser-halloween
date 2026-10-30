import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


def specifications():
    cases = [{"name": "control", "velocity_delta": [0.] * 6, "rotation": [0.] * 3, "rise_m": 0.}]
    for axis, name in enumerate(("vx", "vy", "vz", "wx", "wy", "wz")):
        values = (-.8, -.4, .4, .8) if axis < 3 else (-2., -1., 1., 2.)
        for value in values:
            velocity = np.zeros(6)
            velocity[axis] = value
            cases.append({"name": f"{name}_{value:+g}", "velocity_delta": velocity.tolist(),
                          "rotation": [0.] * 3, "rise_m": 0.})
    for axis, name in enumerate(("roll", "pitch")):
        for value in (-.1, -.05, .05, .1):
            rotation = np.zeros(3)
            rotation[axis] = value
            cases.append({"name": f"{name}_{value:+g}", "velocity_delta": [0.] * 6,
                          "rotation": rotation.tolist(), "rise_m": 0.})
    for rise in (.02, .05, .1):
        cases.append({"name": f"rise_{rise}", "velocity_delta": [0.] * 6,
                      "rotation": [0.] * 3, "rise_m": rise})
    rng = np.random.default_rng(26026)
    for index in range(8):
        cases.append({"name": f"combined_{index}",
                      "velocity_delta": rng.uniform([-.2, -.1, -.4, -.5, -.5, -.5], [.2, .1, 0., .5, .5, .5]).tolist(),
                      "rotation": [float(rng.uniform(-.03, .03)), float(rng.uniform(-.05, .05)), 0.], "rise_m": 0.})
    assert len(cases) == 44 and len({case["name"] for case in cases}) == 44
    return cases


def initial_state(task, nominal, specification):
    pose = np.array(nominal, dtype=float).copy()
    velocity = np.zeros(task.model.nv)
    velocity[:6] = specification["velocity_delta"]
    probe = mujoco.MjData(task.model)
    probe.qpos[:] = pose
    mujoco.mj_forward(task.model, probe)
    original_gap = min(float(mujoco.mj_geomDistance(task.model, probe, geom, task.floor, 2., None))
                       for geom in task.supports)
    rotation = np.array(specification["rotation"])
    angle = np.linalg.norm(rotation)
    adjustment = 0.
    if angle:
        quaternion = np.r_[np.cos(angle / 2.), np.sin(angle / 2.) * rotation / angle]
        mujoco.mju_mulQuat(pose[3:7], quaternion, np.array(nominal[3:7]))
        probe.qpos[:] = pose
        mujoco.mj_forward(task.model, probe)
        rotated_gap = min(float(mujoco.mj_geomDistance(task.model, probe, geom, task.floor, 2., None))
                          for geom in task.supports)
        adjustment = original_gap - rotated_gap
    pose[2] += adjustment + specification["rise_m"]
    return {"qpos": pose.tolist(), "qvel": velocity.tolist()}, float(adjustment)


def main():
    torch.set_num_threads(1)
    source = ROOT / "output/rl/low_support_transition/2026-10-23_14-46-00-951505/report.json"
    original = json.loads(source.read_text())
    nominal = original["cases"][0]["nodes"][-1]["qpos"]
    task = StandTransitionTask()
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    assert assets == original["assets"]
    assert digest(task.policy_path) == original["policy_sha256"]
    assert digest(task.bundle_path) == original["bundle_sha256"]
    task.target = np.array(nominal)[task.addresses].copy()
    output = ROOT / "output/rl/handoff_envelope" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(source), "assets": assets,
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "stand_transition_task.py", "near_handoff_task.py",
                           "transition_execution.py", "crawl_control.py")},
              "protocol": "44 predeclared independent low-pose resets; immediate frozen-controller handoff, no readiness gate. "
              "Translation velocity is world-frame m/s; free-joint angular velocity is body-frame rad/s. "
              "Pose rotations are world-axis rotation vectors (left quaternion multiplication). "
              "Rotated poses translate vertically to preserve minimum support-floor gap, explicitly recorded. "
              "No pose or velocity changes after reset. Native PD initial screening and original strict per2ms gates. "
              "2s crawl/cycles/forward6cm acceptance; failed resets counted separately. "
              "Load peaks are sampled normal loads, impulses rectangular sums over observed duration, including failure samples; "
              "neither is a calibrated material stress or impact-only impulse estimate. "
              "Single-axis sweeps plus eight fixed-seed coupled probes, not a certified continuous basin, "
              "not standing-transition success, no hardware impact threshold or independent statistical validation.",
              "specifications": specifications(), "cases": []}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for specification in report["specifications"]:
            state, adjustment = initial_state(task, nominal, specification)
            entry = {"name": specification["name"], "state": state, "floor_alignment_m": adjustment}
            try:
                task.reset(state)
            except ValueError as error:
                row, gates, penetration = task.measure()
                entry.update(executed=False, success=False, initial_gates={name: bool(value) for name, value in gates.items()}, rejection=str(error))
            else:
                row, gates, penetration = task.measure()
                entry.update(executed=True, initial_gates={name: bool(value) for name, value in gates.items()}, initial_loads_n=row[4:8],
                             initial_gaps_m=row[8:12], initial_penetrations_m=penetration)
                result = task.finish()
                trajectory = output / f"{specification['name']}.npz"
                task.save_and_replay(trajectory)
                samples = np.asarray(task.samples)
                entry.update(success=result["success"], result=result, replay_passed=True,
                             trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=digest(trajectory),
                             sampled_peak_loads_n=np.max(samples[:, 4:8], axis=0).tolist(),
                             sampled_normal_impulses_ns=(np.sum(samples[:, 4:8], axis=0) * .002).tolist(),
                             observed_duration_s=float(samples[-1, 0]),
                             first_200ms_peak_hand_load_n=float(np.max(samples[:100, 4:6])))
            report["cases"].append(entry)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(entry["name"], entry["executed"], entry["success"], entry.get("result", {}).get("failure", entry.get("rejection")), flush=True)
        report["summary"] = {"planned": 44, "executed": sum(case["executed"] for case in report["cases"]),
                             "successful": sum(case["success"] for case in report["cases"]),
                             "failure_counts": dict(Counter(reason for case in report["cases"]
                                                            for reason in case.get("result", {}).get("failure", []))),
                             "failed_acceptance_gates": dict(Counter(name for case in report["cases"]
                                                                      for name, passed in case.get("result", {}).get("gates", {}).items()
                                                                      if not passed))}
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
