import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "output/rl/short_step_evaluation/2026-10-19_21-08-38-251264/report.json"


def verify(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def load_replay(report_path, label):
    report = json.loads(report_path.read_text())
    if report["status"] != "completed":
        raise ValueError("Evaluation is not complete")
    bundle_path = verify(report["bundle"], report["bundle_sha256"])
    bundle = json.loads(bundle_path.read_text())
    model_path = verify(bundle["model"], bundle["model_sha256"])
    for path, expected in bundle["mesh_sha256"].items():
        verify(path, expected)
    run = next((entry for entry in report["runs"] if entry["label"] == label), None)
    if run is None:
        raise ValueError(f"Case {label} not found in {report_path}")
    trajectory = verify(run["trajectory"], run["trajectory_sha256"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with np.load(trajectory, allow_pickle=False) as saved:
        positions = saved["qpos"].copy()
        velocities = saved["qvel"].copy()
    times = np.array([0.] + [row["elapsed_s"] for row in run["trace"]])
    if (len(times) < 2 or positions.shape != (len(times), model.nq)
            or velocities.shape != (len(times), model.nv)
            or not all(np.isfinite(values).all() for values in (times, positions, velocities))
            or not np.all(np.diff(times) > 0)
            or not np.isclose(times[-1], run["elapsed_s"])):
        raise ValueError("Invalid trajectory shape, values or timestamps")
    return model, positions, velocities, times, run


def main():
    parser = argparse.ArgumentParser(description="Saved physical evaluation playback, NOT a live controller")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--case", choices=("ppo_deterministic", "ik_zero_residual", "known_unload_return", "hold"), default="ppo_deterministic")
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    model, positions, velocities, times, run = load_replay(args.report, args.case)
    print(f"{args.case}: {len(times)} frames, {times[-1]:.3f}s, success={run['success']}, "
          f"stage={run['max_stage']}, failure={run['failure']}, deadline={run['deadline']}", flush=True)
    print("Saved-state playback only; no new physics rollout or training. Contact metrics belong to the saved report.", flush=True)
    if args.check_only:
        return
    import mujoco.viewer

    data = mujoco.MjData(model)
    data.qpos[:] = positions[0]
    data.qvel[:] = velocities[0]
    mujoco.mj_forward(model, data)
    controls = {"paused": False, "restart": False}

    def key_callback(key):
        if key == 32:
            controls["paused"] = not controls["paused"]
        elif key in (82, 114):
            controls["restart"] = True

    print("Space: pause/resume; R: restart; mouse: orbit/zoom. Holds final pose; close window to exit.", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = positions[0, :3]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -22
            viewer.opt.geomgroup[2] = 1
            viewer.opt.geomgroup[3] = 0
        elapsed = 0.
        previous = time.monotonic()
        while viewer.is_running():
            now = time.monotonic()
            if controls["restart"]:
                elapsed = 0.
                controls["restart"] = False
            elif not controls["paused"]:
                elapsed = min(times[-1], elapsed + (now - previous) * args.speed)
            previous = now
            frame = min(len(times) - 1, np.searchsorted(times, elapsed, side="right") - 1)
            with viewer.lock():
                data.qpos[:] = positions[frame]
                data.qvel[:] = velocities[frame]
                data.time = times[frame]
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
