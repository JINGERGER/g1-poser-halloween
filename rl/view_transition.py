import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import mujoco
import numpy as np

from artifact_utils import digest
from crawl_scene import make_halloween_scene
from crawl_witches import animate_witches


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "output/rl/coordinated_transition_v1/2026-10-30_16-08-24-240354/report.json"
MODEL = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"


def main():
    parser = argparse.ArgumentParser(description="Saved transition playback, not live physics or a trained transition policy")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scene", choices=("halloween", "plain"), default="halloween")
    parser.add_argument("--start", type=float, help="Simulation time; defaults to recorded stage start, otherwise zero")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--case", default="nominal")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("speed must be finite and positive")
    report = json.loads(args.report.read_text())
    run = report
    if "cases" in report:
        run = next((case for case in report["cases"] if case.get("label") == args.case), None)
        if run is None:
            parser.error("Requested case not found")
    if not run.get("passed", run.get("success", False)) or not run.get("replay_passed"):
        raise ValueError("Expected a passed, replay-verified report")
    trajectory = ROOT / run["trajectory"]
    if digest(trajectory) != run["trajectory_sha256"]:
        raise ValueError("Trajectory checksum mismatch")
    with np.load(trajectory, allow_pickle=False) as saved:
        positions, velocities = saved["qpos"].copy(), saved["qvel"].copy()
        times = np.r_[0., saved["samples"][:, 0]]
        if not np.all(np.isin(saved["phase"], (0, 1))):
            raise ValueError("Unsupported execution phases")
    start = run.get("stage", {}).get("start_time_s", 0.) if args.start is None else args.start
    if not np.isfinite(start) or not 0. <= start <= times[-1]:
        parser.error("start must be within the trajectory")
    with TemporaryDirectory(prefix="g1-transition-view-") as directory:
        path = make_halloween_scene(MODEL, Path(directory) / "scene.xml") if args.scene == "halloween" else MODEL
        model = mujoco.MjModel.from_xml_path(str(path))
        model.geom_solref[:, 0] = .005
        if (positions.shape != (len(times), model.nq) or velocities.shape != (len(times), model.nv)
                or not all(np.isfinite(values).all() for values in (positions, velocities, times))
                or not np.allclose(np.diff(times), model.opt.timestep, atol=1e-9, rtol=0.)
                or not np.isclose(times[-1], run.get("duration_s", run.get("elapsed_s")))):
            raise ValueError("Invalid archive dimensions or timebase")
        print(f"Verified archive: {times[-1]:.2f}s; displaying from {start:.2f}s at {args.speed:g}x.", flush=True)
        print("Saved-state playback only; no new physics. Full nominal transition:", report.get("full_transition_validated", False), flush=True)
        if report.get("full_transition_validated"):
            print("Nominal research demo only; perturbation robustness is NOT validated.", flush=True)
        data = mujoco.MjData(model)

        def frame_at(elapsed):
            frame = int(np.clip(np.searchsorted(times, elapsed, side="right") - 1, 0, len(times) - 1))
            data.qpos[:], data.qvel[:], data.time = positions[frame], velocities[frame], times[frame]
            mujoco.mj_forward(model, data)
            if args.scene == "halloween":
                animate_witches(model, data)

        frame_at(start)
        if args.check_only:
            frame_at(times[-1])
            print("Scene and first/final playback frames loaded successfully.", flush=True)
            return
        from mujoco import viewer as mj_viewer

        controls = {"paused": False, "seek": None}

        def key_callback(key):
            if key == 32:
                controls["paused"] = not controls["paused"]
            elif key in (82, 114):
                controls["seek"] = start
            elif key == 268:
                controls["seek"] = 0.
            elif key == 269:
                controls["seek"] = float(times[-1])

        print("Space: pause; R: replay selected segment; Home: full start; End: final pose. Close window to exit.", flush=True)
        with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.cam.lookat[2] = .5
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135., -20.
                viewer.opt.geomgroup[2], viewer.opt.geomgroup[3] = 1, 0
            elapsed, previous = start, time.monotonic()
            while viewer.is_running():
                now = time.monotonic()
                if controls["seek"] is not None:
                    elapsed, controls["seek"] = controls["seek"], None
                elif not controls["paused"]:
                    elapsed += (now - previous) * args.speed
                    elapsed = start if args.loop and elapsed > times[-1] else min(times[-1], elapsed)
                previous = now
                with viewer.lock():
                    frame_at(elapsed)
                viewer.sync()
                time.sleep(1 / 60)


if __name__ == "__main__":
    main()
