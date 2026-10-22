from datetime import datetime
import argparse
import json

import numpy as np
import mujoco
import torch

from reference_policy_baseline import ROOT, materialize, run
from artifact_utils import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ankle-comparison", action="store_true")
    parser.add_argument("--clearance-comparison", action="store_true")
    parser.add_argument("--clearance-extension", action="store_true")
    parser.add_argument("--verify-candidate", action="store_true")
    args = parser.parse_args()
    assert sum((args.ankle_comparison, args.clearance_comparison, args.clearance_extension, args.verify_candidate)) <= 1
    folder, constants = materialize()
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    bundle_path = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
    initial = np.array(json.loads(bundle_path.read_text())["initial_qpos"])
    torch.set_num_threads(1)
    policy = torch.jit.load(str(folder / "policy_crawl.pt"), map_location="cpu").eval()
    experiment = "transfer_ankle_comparison" if args.ankle_comparison else "transfer_contact_comparison"
    if args.clearance_comparison:
        experiment = "transfer_clearance_comparison"
    if args.clearance_extension:
        experiment = "transfer_clearance_extension"
    if args.verify_candidate:
        experiment = "crawl_candidate_verification"
    output = ROOT / "output/rl" / experiment / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {
        "status": "running", "source_sha256": digest(__file__),
        "runner_sha256": digest(ROOT / "rl/reference_policy_baseline.py"),
        "policy_sha256": digest(folder / "policy_crawl.pt"), "bundle_sha256": digest(bundle_path),
        "target_assets": {str(path.relative_to(model_path.parent)): digest(path)
                          for path in sorted(model_path.parent.rglob("*")) if path.is_file()},
        "protocol": "Six fixed contact/time-step sensitivity diagnostics. Reference PD, audited qpos, zero velocity, 20ms policy interval and +0.5m/s command fixed. Contact solref time constant .02/.01/.005 crossed with physics dt .002/.001. Runtime contact solver changes explicitly NOT original-model acceptance or calibrated hardware compliance. Geometry, inertia, friction, actuator/joint limits unchanged. No gate relaxed.",
        "runs": [],
    }
    cases = [(contact_time, timestep, None, 0.) for contact_time in (.02, .01, .005) for timestep in (.002, .001)]
    if args.ankle_comparison:
        cases = [(contact_time, .002, bound, 0.) for contact_time in (.02, .005) for bound in (-.4, -.2, 0.)]
        report["protocol"] = (
            "Six fixed ankle target safety diagnostics, motivated by foot-shin collision near actual ankle pitch -0.586rad. "
            "Lower-bound both ankle pitch actuator targets at -.4/-.2/0 crossed with contact time constant .02/.005. "
            "Reference PD, audited initial qpos, zero velocity, dt .002, 20ms policy interval, +.5m/s fixed. "
            "Only controller target bound and explicit contact solver profile change. "
            "Geometry, collision flags, inertia, friction, actuator force limits and model joint limits unchanged; no gate relaxed.")
    if args.clearance_comparison:
        cases = [(contact_time, .002, -.2, boost) for contact_time in (.02, .005) for boost in (.01, .02, .03)]
        report["protocol"] = (
            "Six fixed physics-executed controller residual diagnostics. Ankle pitch targets bounded below at -.2rad "
            "to avoid foot-shin collision. Opposite hand actual clearance gates .01/.02/.03m foot lift corrections, "
            "mapped through current-state leg Jacobian (XYZ-preserving damped least squares, each joint residual <=.2rad). "
            "Contact time constant .02/.005, reference PD, audited initial qpos, dt .002, 20ms control and +.5m/s fixed. "
            "No state projection, external force, mass/geometry/friction/force limit changes or relaxed acceptance.")
    if args.clearance_extension:
        cases = [(.005, timestep, -.2, boost) for boost in (.04, .06, .08) for timestep in (.002, .001)]
        report["protocol"] = (
            "Final fixed six-case clearance residual extension: .04/.06/.08m requested lift crossed with dt .002/.001. "
            "Motivation: first .01/.02/.03m grid eliminated self-contact and increased foot clearance but did not "
            "produce consecutive all-limb qualified cycles. Keep per-joint residual cap .2rad, ankle target >=-.2, "
            "contact time constant .005 (explicit non-original numerical profile), reference PD, audited qpos, "
            "zero velocity, 20ms control, command +.5. No other model or acceptance changes. "
            "Stop extending amplitudes after this grid; verify a candidate or move to learned adaptation.")
    if args.verify_candidate:
        cases = [(.005, .002, -.2, .08)] * 3 + [(.005, .001, -.2, .08)] + [(.005, .002, -.2, .08)] * 2
        report["protocol"] = (
            "Frozen candidate verification, NOT further search: reference PD, contact time constant .005, "
            "ankle target >=-.2, clearance request .08m (not achieved clearance), residual cap .2rad, "
            "20ms control and command +.5. Three fresh nominal 2ms rollouts, one 1ms numerical check, "
            "two initial world-yaw perturbations -.03/+.03rad. Same model geometry/inertia/friction/force limits. "
            "Success must be independently audited at every physics substep, including true alternating cycles.")
    try:
        for case_index, (contact_time, timestep, bound, boost) in enumerate(cases):
            pose = initial.copy()
            yaw = 0.
            if args.verify_candidate and case_index >= 4:
                yaw = -.03 if case_index == 4 else .03
                quaternion = np.zeros(4)
                mujoco.mju_mulQuat(quaternion, np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]), pose[3:7])
                pose[3:7] = quaternion
            trial = run(folder, constants, policy, round(.02 / timestep), .5,
                        output / f"case_{case_index}_contact_{contact_time}_dt_{timestep}_ankle_{bound}_boost_{boost}.npz",
                        model_path=model_path, gain_profile="reference", initial_qpos=pose,
                        contact_time_constant=contact_time, physics_timestep=timestep, ankle_min_target=bound,
                        foot_clearance_boost=boost)
            trial["initial_yaw_perturbation_rad"] = yaw
            trial["case_index"] = case_index
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
