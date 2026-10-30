import argparse
import json
from datetime import datetime

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_standing_endpoint import evaluate, sample, standing_pose
from check_unitree_standing_baseline import COMMIT, REFERENCE, load_reference
from reference_policy_baseline import ROOT
from unitree_modified_adapter import LegAdapter
from unitree_balance_probes import CASES, assess, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", action="store_true", help="Six fixed baseline/push/lowering comparisons")
    args = parser.parse_args()
    config, functions = load_reference()
    reference = mujoco.MjModel.from_xml_path(str(REFERENCE / "resources/robots/g1_description/scene.xml"))
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    policy_path = REFERENCE / "deploy/pre_train/g1/motion.pt"
    torch.set_num_threads(1)
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    experiment = "unitree_balance_probes" if args.probes else "unitree_modified_transfer"
    output = ROOT / "output/rl" / experiment / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "official_commit": COMMIT, "policy_sha256": digest(policy_path),
              "config_sha256": digest(REFERENCE / "deploy/deploy_mujoco/configs/g1.yaml"),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("check_unitree_modified_transfer.py", "unitree_modified_adapter.py", "unitree_balance_probes.py", "check_standing_endpoint.py", "check_unitree_standing_baseline.py")},
              "model_assets": {str(path.relative_to(model_path.parent)): digest(path)
                               for path in sorted(model_path.parent.rglob("*")) if path.is_file()},
              "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "torch": torch.__version__},
              "protocol": "Three fixed 5s trials from the same knee-0.6 standing pose and zero velocity. "
                          "Native position PD holds that pose; official external PD holds official default angles; "
                          "official RL uses zero command, initial default target, zero previous action and original phase. "
                          "Upper body retains native position servos and original initial targets in all cases. "
                          "Modified-model solref .005 and dt .002 fixed; policy period .02. No gain blending or state reset.",
              "interface_changes": "Official cases convert only 12 leg position servos into unit-gain torque motors in memory. "
                                   "Disable angle-valued ctrl clamps; retain joint actuator force limits, actuator force limits, "
                                   "gear, passive damping, inertia, friction, geometry and collision. XML files unchanged. "
                                   "External explicit PD is not numerically identical to implicit position-servo PD.",
              "limitations": "Not an isolated gains ablation: native and official cases differ in target angles and servo implementation. "
                             "Abrupt startup is tested without blending; failure does not rule out adaptation. "
                             "No standing-to-crawl transition, disturbance recovery or training.",
              "sample_columns": ["time", "height", "tilt", "linear_speed", "angular_speed", "left_load", "right_load",
                                 "nonfoot_load", "penetration", "self_penetration", "torque_ratio", "joint_excess", "foot_drift"],
              "runs": []}
    cases = CASES if args.probes else [(mode, mode, "none") for mode in ("native_hold", "official_pd", "official_zero")]
    if args.probes:
        report["protocol"] = (
            "Six predefined 5s cases: native and RL baselines, RL world-x/y 15N pelvis pushes during [2,2.2)s, "
            "native/RL lowering with smooth joint offsets from 2 to 3s, then held: knee +.2, hip/ankle pitch -.1rad. "
            "Same modified model, solref .005, dt .002, policy .02; no state reset, no gain switching. "
            "Push impulse 3Ns, no other force. Offset added to physical target, NOT policy observation defaults or raw previous action. "
            "Final second compared to corresponding baseline; lowering requires mean pelvis reduction >=1cm and all physical gates. "
            "Push recovery requires tail RMS linear/angular speed <=baseline+.05m/s/.1rad/s, tilt RMS <=baseline+.03rad, "
            "tail xy separation <=.15m plus all physical gates. Stepping is allowed; static gates reported separately.")
        report["limitations"] = "Finite deterministic probes, not a robust recovery guarantee. No complete transition, training or gain handoff. Lowering is a target-offset diagnostic, not an official height command."
    baselines = {}
    try:
        for name, mode, kind in cases:
            model = mujoco.MjModel.from_xml_path(str(model_path))
            model.geom_solref[:, 0] = .005
            assert model.opt.timestep == config["simulation_dt"] == .002 and model.na == 0
            adapter = LegAdapter(model, reference)
            data = standing_pose(model, .6)
            joints = model.actuator_trnid[:, 0]
            data.ctrl[:] = data.qpos[model.jnt_qposadr[joints]]
            initial_ctrl = data.ctrl.copy()
            pelvis = model.body("pelvis").id
            if mode != "native_hold":
                adapter.torque_interface(model)
            defaults = np.array(config["default_angles"], dtype=np.float32)
            target, action = defaults.copy(), np.zeros(12, dtype=np.float32)
            kps, kds = [np.array(config[key], dtype=np.float32) for key in ("kps", "kds")]

            def set_control(step):
                force, offset = inputs(kind, step, adapter.names)
                data.xfrc_applied[:] = 0.
                data.xfrc_applied[pelvis, :3] = force
                if mode != "native_hold":
                    data.ctrl[adapter.actuators] = functions["pd_control"](
                        target + offset, data.qpos[adapter.qpos], kps, np.zeros_like(kds), data.qvel[adapter.qvel], kds)
                else:
                    data.ctrl[:] = initial_ctrl
                    data.ctrl[adapter.actuators] += offset

            set_control(0)
            mujoco.mj_forward(model, data)
            feet = [model.geom(name).id for name in ("left_foot", "right_foot")]
            floor = model.geom("floor").id
            start_feet = data.geom_xpos[feet].copy()
            positions, velocities, controls, observations = [data.qpos.copy()], [data.qvel.copy()], [], []
            forces = []
            samples = [sample(model, data, feet, floor, joints, start_feet)]
            stopped = None
            for counter in range(1, 2501):
                set_control(counter - 1)
                controls.append(data.ctrl.copy())
                forces.append(data.xfrc_applied[pelvis].copy())
                assert not data.qfrc_applied.any()
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Nonfinite transfer state")
                positions.append(data.qpos.copy())
                velocities.append(data.qvel.copy())
                samples.append(sample(model, data, feet, floor, joints, start_feet))
                if counter % config["control_decimation"] == 0 and mode == "official_zero":
                    observation = adapter.observation(data, action, counter, np.zeros(3, dtype=np.float32), config,
                                                      functions["get_gravity_orientation"])
                    observations.append(observation.copy())
                    with torch.inference_mode():
                        action = policy(torch.from_numpy(observation).unsqueeze(0)).numpy().squeeze()
                    assert action.shape == (12,) and np.isfinite(action).all()
                    target = defaults + config["action_scale"] * action
                if data.qpos[2] < .4 or samples[-1][2] > .8:
                    stopped = "height_or_tilt_emergency"
                    break
            replay = mujoco.MjData(model)
            replay.qpos[:], replay.qvel[:], replay.ctrl[:] = positions[0], velocities[0], controls[0]
            mujoco.mj_forward(model, replay)
            for index, control in enumerate(controls):
                replay.ctrl[:] = control
                replay.xfrc_applied[pelvis] = forces[index]
                mujoco.mj_step(model, replay)
                mujoco.mj_forward(model, replay)
                np.testing.assert_allclose(replay.qpos, positions[index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(replay.qvel, velocities[index + 1], atol=1e-10, rtol=0)
            samples = np.array(samples)
            gates = evaluate(samples, bool(data.time >= 5. - 1e-8 and stopped is None), True)
            path = output / f"{name}.npz"
            np.savez_compressed(path, qpos=positions, qvel=velocities, ctrl=controls, observations=observations,
                                pelvis_wrench=forces,
                                samples=samples, gainprm=model.actuator_gainprm, biasprm=model.actuator_biasprm,
                                gaintype=model.actuator_gaintype, biastype=model.actuator_biastype,
                                ctrllimited=model.actuator_ctrllimited)
            result = {"mode": mode, "mapping": adapter.manifest(), "standing_passed": all(gates.values()),
                      "gates": gates, "stopped": stopped, "elapsed_s": float(data.time),
                      "displacement_m": (data.qpos[:3] - positions[0][:3]).tolist(),
                      "max_tilt_rad": float(samples[:, 2].max()), "max_torque_ratio": float(samples[:, 10].max()),
                      "max_self_penetration_m": float(samples[:, 9].max()), "max_penetration_m": float(samples[:, 8].max()),
                      "max_foot_drift_m": float(samples[:, 12].max()), "trajectory": str(path), "sha256": digest(path)}
            if args.probes:
                if kind == "none":
                    baselines[mode] = (samples.copy(), np.array(positions))
                baseline_samples, baseline_positions = baselines[mode]
                result.update(name=name, probe_kind=kind,
                              applied_impulse_ns=(np.array(forces)[:, :3].sum(axis=0) * model.opt.timestep).tolist(),
                              assessment=assess(kind, samples, baseline_samples, gates, np.array(positions), baseline_positions))
            report["runs"].append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "mapping"}), flush=True)
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
