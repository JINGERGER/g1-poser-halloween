import hashlib
import json

import mujoco
import numpy as np

from check_support_baseline import trial
from prepare import ASSETS, ROOT
from servo_force import position_servo_force, validate_position_servos


def apply_reference_pd(model, comparison):
    parameters = {row["joint"]: row for row in comparison}
    names = [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]]
    if len(parameters) != model.nu or set(parameters) != set(names):
        raise ValueError("Reference actuator names do not match this model")
    for index, name in enumerate(names):
        row = parameters[name]
        stiffness = float(row["reference_training_kp"])
        damping = float(row["reference_training_kd"])
        if not np.isfinite([stiffness, damping]).all() or stiffness <= 0 or damping < 0:
            raise ValueError(f"Invalid reference gains for {name}")
        model.actuator_gainprm[index, 0] = stiffness
        model.actuator_biasprm[index, 1] = -stiffness
        model.actuator_biasprm[index, 2] = -damping


def check_force_response(model):
    validate_position_servos(model)
    data = mujoco.MjData(model)
    maximum_error = 0.0
    for offset in (-0.05, 0.0, 0.05):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.ctrl[:] = np.clip(data.qpos[7:] + offset, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        data.qvel[6:] = np.linspace(-0.2, 0.2, model.nu)
        mujoco.mj_forward(model, data)
        expected = position_servo_force(model, data.ctrl, data.qpos[7:], data.qvel[6:])
        maximum_error = max(maximum_error, float(np.max(np.abs(expected - data.qfrc_actuator[6:]))))
    assert maximum_error < 1e-6
    return maximum_error


def main():
    path = ASSETS / "flat.xml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = json.loads((ROOT / "output/rl/reference_audit.json").read_text())
    if reference["model_sha256"] != digest:
        raise ValueError("Model changed: rerun audit_reference.py")
    original = mujoco.MjModel.from_xml_path(str(path))
    candidate = mujoco.MjModel.from_xml_path(str(path))
    apply_reference_pd(candidate, reference["actuator_comparison"])
    changed = []
    for name in dir(original):
        left = getattr(original, name)
        if isinstance(left, np.ndarray) and not np.array_equal(left, getattr(candidate, name)):
            changed.append(name)
    assert set(changed) == {"actuator_gainprm", "actuator_biasprm"}
    assert np.array_equal(original.actuator_gainprm[:, 1:], candidate.actuator_gainprm[:, 1:])
    assert np.array_equal(original.actuator_biasprm[:, 0], candidate.actuator_biasprm[:, 0])
    assert np.array_equal(original.actuator_biasprm[:, 3:], candidate.actuator_biasprm[:, 3:])
    variants = {}
    for label, model in (("current_pd", original), ("reference_training_pd", candidate)):
        error = check_force_response(model)
        trials = [trial(model, seed) for seed in (None, 101, 102, 103, 104)]
        for result in trials:
            trace = result["trace"]
            result["max_torque_ratio_full_run"] = max(row["max_torque_ratio"] for row in trace)
            result["near_limit_fraction_full_run"] = float(np.mean([row["max_torque_ratio"] >= 0.95 for row in trace]))
            result["first_floor_non_support_contact_s"] = next((row["time_s"] for row in trace if any("floor" in contact[:2] for contact in row["unwanted_contacts"])), None)
        variants[label] = {"force_response_max_error_nm": error, "trials": trials}
        print(label, json.dumps([{key: row[key] for key in ("reset_seed", "first_unwanted_contact_s", "first_floor_non_support_contact_s", "settled_support_screen_passed", "max_torque_ratio_full_run", "tail_summary")} for row in trials], indent=2), flush=True)
    for current, alternate in zip(variants["current_pd"]["trials"], variants["reference_training_pd"]["trials"]):
        assert current["initial_state_control_sha256"] == alternate["initial_state_control_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    report = {"model_sha256": digest, "reference_commit": reference["reference_commit"],
              "mujoco_version": mujoco.__version__, "changed_model_arrays": changed,
              "paired_initial_states_identical": True, "model_file_unchanged": True,
              "variants": variants,
              "limitations": "Reference PD numbers on OUR geometry, home pose, damping, armature, solver and self-collision settings. Not the original Isaac Lab implicit actuator or reference training USD. Five paired fixed-target rollouts, no policy/training/transfer. Endpoint sampling at 20ms. No hardware or optimal-gain claim; failure does not prove reference RL infeasible."}
    output = ROOT / "output/rl/reference_pd_check.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
