import ast
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from fetch_reference_crawl_policy import COMMIT, EXPECTED, SIZE
from artifact_utils import digest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "g1_crawl"


def blob(name):
    folder = ROOT / "output/rl/reference_assets" / f"g1_crawl_{COMMIT}"
    relative = name.removeprefix("sim2sim_mj/")
    if relative == "poses/crawl-pose.json":
        relative = "crawl-pose.json"
    cached = folder / relative
    if (folder / "run_sim2sim.py").is_file() and cached.is_file():
        return cached.read_bytes()
    return subprocess.check_output(["git", "-C", str(REFERENCE), "show", f"{COMMIT}:{name}"])


def write_once(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == contents
    else:
        path.write_bytes(contents)


def materialize():
    folder = ROOT / "output/rl/reference_assets" / f"g1_crawl_{COMMIT}"
    policy = folder / "policy_crawl.pt"
    if not policy.is_file():
        raise FileNotFoundError("Crawl assets are missing. Run: python rl/install_crawl_assets.py")
    assert policy.stat().st_size == SIZE and digest(policy) == EXPECTED
    source = blob("sim2sim_mj/run_sim2sim.py")
    tree = ast.parse(source)
    names = {"CONFIG", "PYTORCH_JOINT_ORDER", "TRAINING_DEFAULT_ANGLES", "JOINT_LIMITS"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(constants) == names
    scene = blob("sim2sim_mj/scene.xml")
    root = ET.fromstring(scene)
    assert not root.findall(".//include")
    write_once(folder / "scene.xml", scene)
    manifest = {"commit": COMMIT, "source_script_sha256": hashlib.sha256(source).hexdigest(), "files": {"scene.xml": digest(folder / "scene.xml")}}
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file")
        assert Path(filename).name == filename
        contents = blob(f"sim2sim_mj/meshes/{filename}")
        assert not contents.startswith(b"version https://git-lfs")
        write_once(folder / "meshes" / filename, contents)
        manifest["files"]["meshes/" + filename] = hashlib.sha256(contents).hexdigest()
    pose = blob("sim2sim_mj/poses/crawl-pose.json")
    write_once(folder / "crawl-pose.json", pose)
    manifest["files"]["crawl-pose.json"] = digest(folder / "crawl-pose.json")
    write_once(folder / "reference_manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return folder, constants


def run(folder, constants, policy, control_substeps, command, output,
        model_path=None, gain_profile="reference", initial_qpos=None,
        contact_time_constant=None, physics_timestep=None, ankle_min_target=None,
        foot_clearance_boost=0., step_callback=None):
    model_path = Path(model_path or folder / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.opt.timestep == .002
    if ankle_min_target is not None:
        assert ankle_min_target in (-.4, -.2, 0.)
    assert foot_clearance_boost in (0., .01, .02, .03, .04, .06, .08)
    if physics_timestep is not None:
        assert physics_timestep in (.001, .002)
        model.opt.timestep = physics_timestep
    if contact_time_constant is not None:
        assert contact_time_constant in (.02, .01, .005)
        assert np.all(model.geom_solref[:, 0] > 0) and model.npair == 0
        model.geom_solref[:, 0] = contact_time_constant
    assert gain_profile in ("reference", "native")
    if gain_profile == "reference":
        reference = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        gains = {reference.joint(int(joint)).name: actuator
                 for actuator, joint in enumerate(reference.actuator_trnid[:, 0])}
        for actuator, joint in enumerate(model.actuator_trnid[:, 0]):
            name = model.joint(int(joint)).name
            if name in gains:
                source = gains[name]
                model.actuator_gainprm[actuator, 0] = reference.actuator_gainprm[source, 0] * 1.4
                model.actuator_biasprm[actuator, 1:3] = reference.actuator_biasprm[source, 1:3] * 1.4
    data = mujoco.MjData(model)
    pose = json.loads((folder / "crawl-pose.json").read_text())["poses"][0]
    data.qpos[:3] = pose["base_pos"]
    roll, pitch, yaw = pose["base_rpy"]
    assert roll == yaw == 0
    data.qpos[3:7] = [np.cos(pitch / 2), 0., np.sin(pitch / 2), 0.]
    order = constants["PYTORCH_JOINT_ORDER"]
    defaults = constants["TRAINING_DEFAULT_ANGLES"]
    joint_ids = {model.joint(index).name: index for index in range(1, model.njnt)}
    for name, joint in joint_ids.items():
        data.qpos[model.jnt_qposadr[joint]] = pose["joints"][name]
    if initial_qpos is not None:
        data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)
    initial_violations = [
        name for name, joint in joint_ids.items()
        if model.jnt_limited[joint] and not
        model.jnt_range[joint, 0] <= data.qpos[model.jnt_qposadr[joint]] <= model.jnt_range[joint, 1]]
    assert not initial_violations, initial_violations
    floor = next(index for index in range(model.ngeom) if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_PLANE)
    support_groups = [[], [], [], []]
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        name = model.geom(geom).name
        if name in ("left_hand_collision", "right_hand_collision"):
            support_groups[0 if name.startswith("left") else 1].append(geom)
        body = model.body(int(model.geom_bodyid[geom])).name
        if body in ("left_ankle_roll_link", "right_ankle_roll_link"):
            support_groups[2 if body.startswith("left") else 3].append(geom)
    assert all(support_groups)
    group_of = {geom: index for index, group in enumerate(support_groups) for geom in group}
    previous = np.zeros(23, dtype=np.float32)
    rows, actions, controls = [], [], []
    positions, velocities = [data.qpos.copy()], [data.qvel.copy()]
    initial = data.qpos[:3].copy()
    failure = None
    period = control_substeps * model.opt.timestep
    missing = [name for name in order if name not in joint_ids]
    for unused in range(round(10 / period)):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ [0., 0., -1.]
        relative, speeds = np.zeros(23, dtype=np.float32), np.zeros(23, dtype=np.float32)
        for index, name in enumerate(order):
            if name in joint_ids:
                joint = joint_ids[name]
                relative[index] = np.float32(data.qpos[model.jnt_qposadr[joint]]) - np.float32(defaults[name])
                speeds[index] = data.qvel[model.jnt_dofadr[joint]]
        observation = np.r_[gravity, [command, 0., 0.], relative, speeds, previous].astype(np.float32)
        assert observation.shape == (75,) and np.isfinite(observation).all()
        with torch.inference_mode():
            action = policy(torch.from_numpy(observation)).numpy().copy()
        assert action.shape == (23,) and np.isfinite(action).all()
        previous = action
        correction = {}
        if foot_clearance_boost:
            for side, opposite_hand in (("left", support_groups[1][0]), ("right", support_groups[0][0])):
                gap = float(mujoco.mj_geomDistance(model, data, opposite_hand, floor, 2., None))
                phase = np.clip((gap - .002) / .015, 0., 1.)
                phase = phase * phase * (3 - 2 * phase)
                leg_names = [f"{side}_{suffix}_joint" for suffix in ("hip_pitch", "hip_roll", "hip_yaw", "knee")]
                columns = [model.jnt_dofadr[joint_ids[name]] for name in leg_names]
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jacGeom(model, data, jacobian, None, support_groups[2 if side == "left" else 3][0])
                leg_jacobian = jacobian[:, columns]
                delta = leg_jacobian.T @ np.linalg.solve(
                    leg_jacobian @ leg_jacobian.T + .001 ** 2 * np.eye(3),
                    np.array([0., 0., foot_clearance_boost * phase]))
                correction.update(zip(leg_names, np.clip(delta, -.2, .2)))
        for actuator, joint in enumerate(model.actuator_trnid[:, 0]):
            name = model.joint(int(joint)).name
            target = np.float32(defaults[name]) + np.float32(.5) * action[order.index(name)]
            target += correction.get(name, 0.)
            limits = constants["JOINT_LIMITS"].get(name, model.jnt_range[joint])
            data.ctrl[actuator] = np.clip(target, *limits)
            if ankle_min_target is not None and name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
                data.ctrl[actuator] = max(data.ctrl[actuator], ankle_min_target)
            if model.actuator_ctrllimited[actuator]:
                data.ctrl[actuator] = np.clip(data.ctrl[actuator], *model.actuator_ctrlrange[actuator])
        actions.append(action.copy())
        controls.append(data.ctrl.copy())
        for unused_substep in range(control_substeps):
            assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = "nonfinite_physics"
            elif not .08 <= data.qpos[2] <= .8:
                failure = "diagnostic_base_height"
            elif max((-contact.dist for contact in data.contact), default=0.) > .04:
                failure = "diagnostic_penetration_over_40mm"
            if failure:
                break
        loads = np.zeros(4)
        other_load = 0.
        for index, contact in enumerate(data.contact):
            if floor not in (contact.geom1, contact.geom2):
                continue
            geom = contact.geom2 if contact.geom1 == floor else contact.geom1
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            force = max(0., float(wrench[0]))
            if geom in group_of:
                loads[group_of[geom]] += force
            else:
                other_load += force
        gaps = [min(float(mujoco.mj_geomDistance(model, data, geom, floor, 2., None)) for geom in group) for group in support_groups]
        rows.append({"time_s": float(data.time), "base_position_m": data.qpos[:3].tolist(), "loads_n": loads.tolist(),
                     "gaps_m": gaps, "other_floor_load_n": other_load,
                     "penetration_m": max((-contact.dist for contact in data.contact), default=0.)})
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        if step_callback is not None and step_callback(model, data, period) is False:
            failure = "viewer_closed"
        if failure:
            break
    np.savez_compressed(output, qpos=positions, qvel=velocities, actions=actions, ctrl=controls)
    return {"control_substeps": control_substeps, "control_dt_s": period, "command_local_z_mps": command,
            "model_path": str(model_path.resolve()), "model_sha256": digest(model_path), "gain_profile": gain_profile,
            "contact_time_constant": contact_time_constant, "physics_timestep": float(model.opt.timestep),
            "ankle_min_target": ankle_min_target,
            "foot_clearance_boost": foot_clearance_boost,
            "model_nu": model.nu, "missing_policy_joints_zero_padded": missing, "elapsed_s": float(data.time), "failure": failure,
            "base_displacement_m": (data.qpos[:3] - initial).tolist(), "max_gaps_m": np.max([row["gaps_m"] for row in rows], axis=0).tolist(),
            "max_other_floor_load_n": max(row["other_floor_load_n"] for row in rows),
            "max_penetration_m": max(row["penetration_m"] for row in rows),
            "trajectory": str(output), "trajectory_sha256": digest(output), "trace": rows,
            "limitations": "Policy-transfer diagnostic; inspect model_path and gain_profile. Emergency checks do not constitute crawl acceptance. No training, no external support, no intermediate state reset."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", nargs="+", type=float, default=[0., .15])
    args = parser.parse_args()
    assert all(np.isfinite(value) and -1. <= value <= 2.5 for value in args.commands)
    before = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain=v1"]) if (REFERENCE / ".git").exists() else None
    folder, constants = materialize()
    torch.set_num_threads(1)
    policy = torch.jit.load(str(folder / "policy_crawl.pt"), map_location="cpu").eval()
    with torch.inference_mode():
        result = policy(torch.zeros(75))
        assert result.shape == (23,) and torch.isfinite(result).all()
    output = ROOT / "output/rl/reference_policy_baseline" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "commit": COMMIT, "policy": str(folder / "policy_crawl.pt"), "policy_sha256": EXPECTED,
              "asset_manifest": str(folder / "reference_manifest.json"), "asset_manifest_sha256": digest(folder / "reference_manifest.json"),
              "source_sha256": digest(__file__), "commands": args.commands, "protocol": "Fixed 10s cases: published 8ms versus training-contract 20ms policy interval. Original scene, crawl pose, 1.4x gains. Diagnostic command comparison, not target-model success.", "runs": []}
    path = output / "report.json"
    try:
        for substeps in (4, 10):
            for command in args.commands:
                trial = run(folder, constants, policy, substeps, command, output / f"substeps_{substeps}_command_{command}.npz")
                report["runs"].append(trial)
                print(json.dumps({key: value for key, value in trial.items() if key != "trace"}), flush=True)
        if before is not None:
            assert before == subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain=v1"])
        report.update(status="completed", reference_worktree_unchanged=True if before is not None else None)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
