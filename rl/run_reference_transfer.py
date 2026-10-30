from datetime import datetime
import json

import mujoco
import numpy as np
import torch

from reference_policy_baseline import ROOT, materialize, run
from target_load_ablation import digest


def description(model):
    return {
        "mass_kg": float(model.body_mass.sum()),
        "actuators": {
            model.joint(int(joint)).name: {
                "range": model.jnt_range[joint].tolist(),
                "kp": float(model.actuator_gainprm[actuator, 0]),
                "kv": float(-model.actuator_biasprm[actuator, 2]),
                "actuator_force_limited": bool(model.actuator_forcelimited[actuator]),
                "actuator_force_range": model.actuator_forcerange[actuator].tolist(),
                "joint_force_limited": bool(model.jnt_actfrclimited[joint]),
                "joint_force_range": model.jnt_actfrcrange[joint].tolist(),
            }
            for actuator, joint in enumerate(model.actuator_trnid[:, 0])
        },
    }


def main():
    folder, constants = materialize()
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    bundle_path = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
    bundle = json.loads(bundle_path.read_text())
    policy = torch.jit.load(str(folder / "policy_crawl.pt"), map_location="cpu").eval()
    torch.set_num_threads(1)
    output = ROOT / "output/rl/reference_transfer" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {
        "status": "running", "source_sha256": digest(__file__),
        "runner_sha256": digest(ROOT / "rl/reference_policy_baseline.py"),
        "policy_sha256": digest(folder / "policy_crawl.pt"),
        "bundle_sha256": digest(bundle_path),
        "target_assets": {str(path.relative_to(model_path.parent)): digest(path)
                          for path in sorted(model_path.parent.rglob("*")) if path.is_file()},
        "reference": description(mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))),
        "target": description(mujoco.MjModel.from_xml_path(str(model_path))),
        "protocol": "Eight fixed cases: reference/native gains, reference/audited qpos with zero initial velocity, 8/20ms controls, +0.5m/s command. No model geometry, inertia, force or joint limit changes. Real wrist DOFs observed and actuated; missing reference wrist gains retain native gains. Diagnostic only, no strict crawl acceptance.",
        "runs": [],
    }
    try:
        for gains in ("native", "reference"):
            for initial in ("reference_pose", "audited_pose"):
                for substeps in (4, 10):
                    trial = run(folder, constants, policy, substeps, .5,
                                output / f"{gains}_{initial}_{substeps}.npz",
                                model_path=model_path, gain_profile=gains,
                                initial_qpos=np.array(bundle["initial_qpos"]) if initial == "audited_pose" else None)
                    trial["initial_pose"] = initial
                    report["runs"].append(trial)
                    print(json.dumps({key: value for key, value in trial.items() if key != "trace"}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
