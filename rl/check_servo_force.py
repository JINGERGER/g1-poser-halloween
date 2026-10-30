import json

import mujoco
import numpy as np

from prepare import ASSETS, ROOT
from servo_force import position_servo_force, validate_position_servos


def main():
    random = np.random.default_rng(20261015)
    results = []
    for actuator_limits in (False, True):
        model = mujoco.MjModel.from_xml_path(str(ASSETS / "flat.xml"))
        if actuator_limits:
            model.actuator_forcelimited[:] = True
            model.actuator_forcerange[:] = [-5, 7]
        validate_position_servos(model)
        data = mujoco.MjData(model)
        joint_ids = model.actuator_trnid[:, 0]
        max_error = 0.0
        for sample in range(128):
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            data.qpos[7:] = random.uniform(model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
            data.qvel[6:] = random.uniform(-20, 20, model.nu)
            data.ctrl[:] = random.uniform(model.actuator_ctrlrange[:, 0] - 2, model.actuator_ctrlrange[:, 1] + 2)
            mujoco.mj_forward(model, data)
            actual = position_servo_force(model, data.ctrl, data.qpos[7:], data.qvel[6:])
            expected = data.qfrc_actuator[6:]
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            assert np.allclose(actual, expected, atol=1e-6, rtol=0)
        model.actuator_gear[0, 0] = 2
        try:
            validate_position_servos(model)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported gear accepted")
        results.append({"synthetic_actuator_limits": actuator_limits, "samples": 128,
                        "max_force_error_nm": max_error, "unsupported_gear_rejected": True})
    output = ROOT / "output/rl/servo_force_check.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
