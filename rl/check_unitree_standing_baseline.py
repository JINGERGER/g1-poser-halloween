import ast
import json
import subprocess
from datetime import datetime

import mujoco
import numpy as np
import torch
import yaml

from artifact_utils import digest
from reference_policy_baseline import ROOT


COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
REFERENCE = ROOT / "output/rl/reference_assets/unitree_rl_gym_276801e"


def load_reference():
    if not REFERENCE.exists():
        raise FileNotFoundError("Clone the pinned unitree_rl_gym checkout documented in RL_UNITREE_STANDING_BASELINE.md")
    commit = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT or subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain"], text=True).strip():
        raise ValueError("Official reference must be the clean pinned checkout")
    config_path = REFERENCE / "deploy/deploy_mujoco/configs/g1.yaml"
    config = yaml.safe_load(config_path.read_text())
    source = REFERENCE / "deploy/deploy_mujoco/deploy_mujoco.py"
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("pd_control", "get_gravity_orientation")]
    assert len(functions) == 2
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return config, namespace


def make_observation(data, action, counter, command, config, gravity):
    count = config["num_actions"]
    defaults = np.array(config["default_angles"], dtype=np.float32)
    phase = (counter * config["simulation_dt"]) % .8 / .8
    observation = np.zeros(config["num_obs"], dtype=np.float32)
    observation[:3] = data.qvel[3:6] * config["ang_vel_scale"]
    observation[3:6] = gravity(data.qpos[3:7])
    observation[6:9] = command * np.array(config["cmd_scale"], dtype=np.float32)
    observation[9:9 + count] = (data.qpos[7:] - defaults) * config["dof_pos_scale"]
    observation[9 + count:9 + 2 * count] = data.qvel[6:] * config["dof_vel_scale"]
    observation[9 + 2 * count:9 + 3 * count] = action
    observation[-2:] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
    return observation


def rollout(config, functions, mode, command, steps=5000):
    xml = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(REFERENCE))
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = config["simulation_dt"]
    assert model.nu == 12 and model.nv == 18 and model.na == 0
    assert np.all(model.actuator_gainprm[:, 0] == 1.) and not model.actuator_biasprm.any()
    data, probe = mujoco.MjData(model), mujoco.MjData(model)
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(REFERENCE))
    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    defaults = np.array(config["default_angles"], dtype=np.float32)
    kps, kds = [np.array(config[key], dtype=np.float32) for key in ("kps", "kds")]
    target, action = defaults.copy(), np.zeros(12, dtype=np.float32)
    positions, velocities, controls, observations, samples = [data.qpos.copy()], [data.qvel.copy()], [], [], []
    floor = model.geom("floor").id
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    joints = model.actuator_trnid[:, 0]
    limits = np.max(np.abs(model.jnt_actfrcrange[joints]), axis=1)
    assert np.all(model.jnt_actfrclimited[joints]) and np.all(limits > 0)
    stopped = None
    for counter in range(1, steps + 1):
        assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        data.ctrl[:] = functions["pd_control"](target, data.qpos[7:], kps, np.zeros_like(kds), data.qvel[6:], kds)
        controls.append(data.ctrl.copy())
        mujoco.mj_step(model, data)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite official-model state")
        if counter % config["control_decimation"] == 0:
            observation = make_observation(data, action, counter, command, config, functions["get_gravity_orientation"])
            observations.append(observation.copy())
            if mode == "policy":
                with torch.inference_mode():
                    action = policy(torch.from_numpy(observation).unsqueeze(0)).numpy().squeeze()
                assert action.shape == (12,) and np.isfinite(action).all()
                target = action * config["action_scale"] + defaults
        mujoco.mj_copyData(probe, model, data)
        mujoco.mj_forward(model, probe)
        loads, other, penetration, self_penetration = np.zeros(2), 0., 0., 0.
        for index, contact in enumerate(probe.contact):
            penetration = max(penetration, -contact.dist)
            if floor not in (contact.geom1, contact.geom2):
                self_penetration = max(self_penetration, -contact.dist)
                continue
            geom = contact.geom2 if contact.geom1 == floor else contact.geom1
            body = model.geom_bodyid[geom]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, probe, index, wrench)
            force = max(0., float(wrench[0]))
            if body in feet:
                loads[feet.index(body)] += force
            else:
                other += force
        tilt = float(np.arccos(np.clip(probe.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2], -1., 1.)))
        ratio = float(np.max(np.abs(data.qfrc_actuator[model.jnt_dofadr[joints]]) / limits))
        angles = data.qpos[7:]
        excess = max(0., float(np.max(model.jnt_range[joints, 0] - angles)), float(np.max(angles - model.jnt_range[joints, 1])))
        samples.append([data.time, *data.qpos[:3], tilt, np.linalg.norm(data.qvel[:3]),
                        np.linalg.norm(data.qvel[3:6]), *loads, other, penetration, self_penetration, ratio, excess])
        if data.qpos[2] < .35 or tilt > 1.:
            stopped = "height_or_tilt_emergency"
            break
    return model, {"qpos": np.array(positions), "qvel": np.array(velocities), "ctrl": np.array(controls),
                   "observations": np.array(observations), "samples": np.array(samples)}, stopped


def main():
    config, functions = load_reference()
    torch.set_num_threads(1)
    output = ROOT / "output/rl/unitree_standing_baseline" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    files = [REFERENCE / name for name in ("LICENSE", "deploy/deploy_mujoco/configs/g1.yaml",
             "deploy/deploy_mujoco/deploy_mujoco.py", "deploy/pre_train/g1/motion.pt")]
    files += sorted((REFERENCE / "resources/robots/g1_description").rglob("*"))
    report = {"status": "running", "commit": COMMIT, "source_sha256": digest(__file__),
              "assets": {str(path.relative_to(REFERENCE)): digest(path) for path in files if path.is_file()},
              "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "torch": torch.__version__, "yaml": yaml.__version__},
              "protocol": "Official unmodified 12DoF model, zero-state MjData initialization exactly as deploy_mujoco.py; "
                          "PD to default angles for first ten physics steps, then policy every 20ms after stepping. "
                          "10 simulation seconds rather than GUI wall time. No extra control clipping or reset. "
                          "Diagnostics use separate MjData so mj_forward cannot alter controller trajectory. "
                          "Three predeclared cases: default PD, policy zero command, policy official +0.5 command.",
              "sample_columns": ["time", "x", "y", "height", "tilt", "linear_speed", "angular_speed", "left_load",
                                 "right_load", "nonfoot_load", "penetration", "self_penetration", "torque_ratio", "joint_excess"],
              "limitations": "Original model only; no modified-G1 transfer or perturbation recovery. "
                             "Strict standing gates are ours, not an official certification. Contact diagnostics recomputed at saved states.",
              "runs": []}
    try:
        for name, mode, command in (("default_pd", "pd", [0., 0., 0.]),
                                    ("policy_zero", "policy", [0., 0., 0.]),
                                    ("policy_forward", "policy", config["cmd_init"])):
            model, saved, stopped = rollout(config, functions, mode, np.array(command, dtype=np.float32))
            replay = mujoco.MjData(model)
            for index, control in enumerate(saved["ctrl"]):
                replay.ctrl[:] = control
                mujoco.mj_step(model, replay)
                np.testing.assert_allclose(replay.qpos, saved["qpos"][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(replay.qvel, saved["qvel"][index + 1], atol=1e-10, rtol=0)
            samples = saved["samples"]
            steady = samples[samples[:, 0] >= 1.]
            tail = samples[samples[:, 0] >= 9.]
            gates = {"completed_10s": bool(stopped is None and samples[-1, 0] >= 10. - 1e-8),
                     "height_055_to_095": bool(np.all((samples[:, 3] >= .55) & (samples[:, 3] <= .95))),
                     "tilt_015rad": bool(np.all(samples[:, 4] <= .15)),
                     "penetration_5mm": bool(np.all(samples[:, 10] <= .005)),
                     "self_penetration_1mm": bool(np.all(samples[:, 11] <= .001)),
                     "nonfoot_load_1n": bool(np.all(samples[:, 9] <= 1.)),
                     "torque_below_95percent": bool(np.all(samples[:, 12] < .95)),
                     "joint_limits_1mrad": bool(np.all(samples[:, 13] <= .001)),
                     "both_feet_loaded_after_1s": bool(len(steady) and np.all(steady[:, 7:9] > 5.)),
                     "horizontal_drift_5cm_after_1s": bool(len(steady) and np.all(np.linalg.norm(steady[:, 1:3] - steady[0, 1:3], axis=1) <= .05)),
                     "tail_linear_speed_005": bool(len(tail) and np.all(tail[:, 5] <= .05)),
                     "tail_angular_speed_01": bool(len(tail) and np.all(tail[:, 6] <= .1))}
            path = output / f"{name}.npz"
            np.savez_compressed(path, **saved)
            result = {"name": name, "mode": mode, "command": command, "stopped": stopped,
                      "strict_standing_passed": all(gates.values()), "gates": gates, "replay_passed": True,
                      "elapsed_s": float(samples[-1, 0]), "displacement_m": (saved["qpos"][-1, :3] - saved["qpos"][0, :3]).tolist(),
                      "max_tilt_rad": float(samples[:, 4].max()), "max_torque_ratio": float(samples[:, 12].max()),
                      "trajectory": str(path), "trajectory_sha256": digest(path)}
            report["runs"].append(result)
            print(json.dumps(result), flush=True)
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
