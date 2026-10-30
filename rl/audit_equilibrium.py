import hashlib
import json

import mujoco
import numpy as np

from audit_geometry import gaps_and_points, hold, unwanted_contacts
from prepare import ASSETS, ROOT, SITES


def solve(model, seed, limb):
    data = mujoco.MjData(model)
    nominal = np.array(seed["qpos"])
    joint_ids = model.actuator_trnid[:, 0]
    addresses = np.r_[2, model.jnt_qposadr[joint_ids]]
    lower = np.r_[0.16, model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[:, 0]]
    upper = np.r_[0.40, model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[:, 1]]
    initial = np.r_[nominal[addresses], seed["statics"]["ctrl"]]
    data.qpos[:] = nominal
    mujoco.mj_forward(model, data)
    targets = np.array([data.site(name).xpos[:2].copy() for name in SITES])
    stance = [index for index in range(4) if index != limb]
    limits = model.jnt_actfrcrange[joint_ids]

    def residual(values):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = nominal
        data.qpos[addresses] = values[:len(addresses)]
        data.ctrl[:] = values[len(addresses):]
        mujoco.mj_forward(model, data)
        gaps, _ = gaps_and_points(model, data)
        xy = np.array([data.site(name).xpos[:2] for name in SITES])
        torque = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
        ratios = np.abs(torque) / np.where(torque >= 0, limits[:, 1], -limits[:, 0])
        penetration = min((float(contact.dist) for contact in unwanted_contacts(model, data)), default=0.0)
        return np.r_[data.qacc[:3], data.qacc[3:] * 0.1,
                     (xy - targets).ravel() * 10,
                     (gaps[limb] - 0.02) * 100,
                     np.maximum(gaps[stance], 0) * 100,
                     np.minimum(gaps[stance] + 0.003, 0) * 100,
                     min(penetration, 0) * 100,
                     np.maximum(ratios - 0.95, 0) * 10,
                     (values - initial) * 0.001]

    values = np.clip(initial, lower, upper)
    damping = 0.01
    history = []
    for iteration in range(100):
        error = residual(values)
        history.append(float(np.linalg.norm(error)))
        jacobian = np.empty((len(error), len(values)))
        for column in range(len(values)):
            shifted = values.copy()
            shifted[column] += 1e-6
            jacobian[:, column] = (residual(shifted) - error) / 1e-6
        delta = np.linalg.lstsq(np.vstack((jacobian, np.sqrt(damping) * np.eye(len(values)))),
                                np.r_[-error, np.zeros(len(values))], rcond=None)[0]
        delta *= min(1.0, 0.02 / max(np.max(np.abs(delta)), 1e-12))
        for scale in (1, 0.5, 0.1, 0.01):
            candidate = np.clip(values + scale * delta, lower, upper)
            if np.linalg.norm(residual(candidate)) < np.linalg.norm(error):
                values = candidate
                damping = max(damping * 0.5, 1e-6)
                break
        else:
            damping *= 10
            if damping > 1e6:
                break
    residual(values)
    gaps, _ = gaps_and_points(model, data)
    penetration = min((float(contact.dist) for contact in unwanted_contacts(model, data)), default=0.0)
    torque = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
    ratio = float(np.max(np.abs(torque) / np.where(torque >= 0, limits[:, 1], -limits[:, 0])))
    acceleration = float(np.max(np.abs(data.qacc)))
    accepted = bool(np.linalg.norm(data.qacc[:3]) < 0.01 and np.max(np.abs(data.qacc[3:])) < 0.1
                    and gaps[limb] >= 0.018 and np.all(gaps[stance] <= 0)
                    and np.all(gaps[stance] >= -0.003) and penetration >= -0.0005 and ratio <= 0.95)
    result = {"limb": SITES[limb], "seed_shift_m": seed["root_shift_seed_m"],
              "qpos": data.qpos.tolist(), "statics": {"ctrl": data.ctrl.tolist()},
              "residual_history": history, "final_residual_norm": float(np.linalg.norm(residual(values))),
              "max_abs_qacc": acceleration,
              "base_linear_acceleration_mps2": float(np.linalg.norm(data.qacc[:3])),
              "max_angular_joint_acceleration_radps2": float(np.max(np.abs(data.qacc[3:]))),
              "gaps_m": gaps.tolist(), "unwanted_min_distance_m": penetration,
              "max_torque_limit_ratio": ratio, "equilibrium_screen_passed": accepted}
    result["physical_hold"] = hold(model, result, limb)
    return result


def main():
    model_path = ASSETS / "flat.xml"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    previous = json.loads((ROOT / "output/rl/geometry_audit.json").read_text())
    if previous["model_sha256"] != digest:
        raise ValueError("Model changed: rerun audit_geometry.py before using its seeds")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq == 30 and model.nv == 29 and model.nu == 23
    results = []
    for seed in previous["trials"]:
        if "physical_hold" not in seed:
            continue
        result = solve(model, seed, SITES.index(seed["limb"]))
        results.append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("qpos", "statics", "residual_history", "physical_hold")}), flush=True)
    report = {"model_sha256": digest, "mujoco_version": mujoco.__version__, "trials": results,
              "screen_thresholds": {"base_acceleration_mps2": 0.01, "angular_joint_acceleration_radps2": 0.1,
                                    "min_swing_gap_m": 0.018, "max_stance_penetration_m": 0.003,
                                    "max_unwanted_penetration_m": 0.0005, "max_torque_ratio": 0.95},
              "limitations": "Four local seeds; fixed base XY/orientation; joint positions, root height and fixed controls optimized with native zero-velocity forward dynamics. No model changes or training. Hypothetical initialization is not a reachable transfer trajectory. A small acceleration residual does not prove dynamic stability; holds are only three seconds. No infeasibility or hardware claim."}
    output = ROOT / "output/rl/equilibrium_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
