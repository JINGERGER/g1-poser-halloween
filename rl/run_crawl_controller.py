import argparse
from datetime import datetime
import json
import time

import numpy as np
import torch

from reference_policy_baseline import ROOT, materialize, run
from artifact_utils import digest
from crawl_scene import HALLOWEEN, make_halloween_scene
from crawl_witches import animate_witches


def main():
    parser = argparse.ArgumentParser(description="Run the verified flat-ground modified-G1 crawl candidate for 10 seconds.")
    parser.add_argument("--viewer", action="store_true", help="Use mjpython on macOS; displays live native physics, not qpos replay.")
    parser.add_argument("--scene", choices=("halloween", "plain"), default="halloween",
                        help="Visual background only; grassy Halloween cemetery with non-colliding decorations and no ramp.")
    args = parser.parse_args()
    folder, constants = materialize()
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    bundle_path = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
    initial = np.array(json.loads(bundle_path.read_text())["initial_qpos"])
    torch.set_num_threads(1)
    policy = torch.jit.load(str(folder / "policy_crawl.pt"), map_location="cpu").eval()
    output = ROOT / "output/rl/crawl_controller" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    base_model_path = model_path
    if args.scene == "halloween":
        model_path = make_halloween_scene(base_model_path, output / "scene_halloween.xml")
    viewer = None
    deadline = None

    def display(model, data, period):
        nonlocal viewer, deadline
        if args.scene == "halloween":
            animate_witches(model, data)
        if not args.viewer:
            return True
        if viewer is None:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.distance = 3.7 if args.scene == "halloween" else 2.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            deadline = time.monotonic()
        if not viewer.is_running():
            return False
        with viewer.lock():
            viewer.cam.lookat[:] = data.qpos[:3]
            if args.scene == "halloween":
                viewer.cam.lookat[2] += .45
        viewer.sync()
        deadline += period
        time.sleep(max(0., deadline - time.monotonic()))
        return True

    print("Flat-ground simulation only: reference PD, solref time constant 0.005, ankle target >=-0.2rad.")
    print("The 0.08m lift request is a bounded controller correction, NOT an achieved 8cm clearance or hardware validation.")
    try:
        trial = run(folder, constants, policy, 10, .5, output / "trajectory.npz",
                    model_path=model_path, gain_profile="reference", initial_qpos=initial,
                    contact_time_constant=.005, physics_timestep=.002,
                    ankle_min_target=-.2, foot_clearance_boost=.08,
                    step_callback=display)
        report = {"source_sha256": digest(__file__), "runner_sha256": digest(ROOT / "rl/reference_policy_baseline.py"),
                  "policy_sha256": digest(folder / "policy_crawl.pt"), "bundle_sha256": digest(bundle_path),
                  "scene": args.scene, "base_model_sha256": digest(base_model_path),
                  "theme_sha256": digest(HALLOWEEN) if args.scene == "halloween" else None,
                  "live_physics": True, "run": trial}
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: value for key, value in trial.items() if key != "trace"}, indent=2))
        print(path)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
