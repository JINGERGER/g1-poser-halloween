import json
from datetime import datetime

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from audit_reference_transfer import audit
from reference_policy_baseline import ROOT, materialize, run
from verify_crawl_candidate import reconstruct


def main():
    folder, constants = materialize()
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    bundle_path = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
    initial = np.array(json.loads(bundle_path.read_text())["initial_qpos"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    torch.set_num_threads(1)
    policy = torch.jit.load(str(folder / "policy_crawl.pt"), map_location="cpu").eval()
    output = ROOT / "output/rl/crawl_handoff" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    cases = [("nominal", 0., None, 0.), ("height_plus_5mm", .005, None, 0.),
             ("vx_plus", 0., 0, .05), ("vx_minus", 0., 0, -.05),
             ("vz_down", 0., 2, -.05), ("pitch_rate_plus", 0., 4, .1),
             ("pitch_rate_minus", 0., 4, -.1)]
    report = {
        "status": "running", "source_sha256": digest(__file__),
        "runner_sha256": digest(ROOT / "rl/reference_policy_baseline.py"),
        "audit_sha256": digest(ROOT / "rl/audit_reference_transfer.py"),
        "event_verifier_sha256": digest(ROOT / "rl/verify_crawl_candidate.py"),
        "policy_sha256": digest(folder / "policy_crawl.pt"), "bundle_sha256": digest(bundle_path),
        "protocol": "Seven predeclared one-factor entry probes, not training or a measured attraction basin. "
                    "Frozen crawl controller, 2ms physics, 20ms policy, +0.5m/s command, previous action zero. "
                    "Free-joint linear velocities are world-frame; angular velocities are local-frame. "
                    "No state reset after initialization. All failures retained; no parameter search.",
        "limitations": ["No standing-to-crawl transition", "No combined perturbations or joint-state perturbations",
                        "No action-history or control blending test", "No hardware validation"],
        "runs": [],
    }
    try:
        for name, height, velocity_index, velocity_value in cases:
            pose, velocity = initial.copy(), np.zeros(model.nv)
            pose[2] += height
            if velocity_index is not None:
                velocity[velocity_index] = velocity_value
            trial = run(folder, constants, policy, 10, .5, output / f"{name}.npz",
                        model_path=model_path, gain_profile="reference", initial_qpos=pose,
                        initial_qvel=velocity, contact_time_constant=.005, physics_timestep=.002,
                        ankle_min_target=-.2, foot_clearance_boost=.08)
            checked = audit(trial, folder)
            with np.load(checked["samples"], allow_pickle=False) as saved:
                samples, positions = saved["samples"], saved["support_positions"]
                if not np.isfinite(samples).all() or not np.isfinite(positions).all():
                    raise ValueError("Nonfinite audit samples")
                events, cycles, consecutive = reconstruct(samples, positions)
                gates = {
                    "replay": checked["replay_passed"],
                    "completed_10s": trial["failure"] is None and trial["elapsed_s"] >= 10. - 1e-8,
                    "forward_6cm": trial["base_displacement_m"][0] >= .06,
                    "penetration_5mm": max(float(saved["penetrations"].max()), checked["initial_penetration_m"]) <= .005,
                    "self_penetration_1mm": max(float(saved["self_penetrations"].max()), checked["initial_self_penetration_m"]) <= .001,
                    "other_floor_load_1n": float(samples[:, 12].max()) <= 1.,
                    "torque_below_95percent": float(samples[:, 13].max()) < .95,
                    "joint_limits": float(samples[:, 14].max()) <= .001,
                    "base_height": bool(np.all((samples[:, 3] >= .16) & (samples[:, 3] <= .65))),
                    "two_consecutive_alternating_cycles": consecutive,
                }
            result = {"name": name, "initial_height_offset_m": height,
                      "initial_qvel": velocity.tolist(), "passed": all(gates.values()), "gates": gates,
                      "qualified_events": [sum(event["qualified"] for event in limb) for limb in events],
                      "cycle_count": len(cycles), "trial": trial, "audit": checked}
            report["runs"].append(result)
            print(json.dumps({key: value for key, value in result.items() if key not in ("trial", "audit")}), flush=True)
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
