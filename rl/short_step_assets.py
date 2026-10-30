import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from ball_contact_assets import write_once
from load_transfer import smooth
from prepare import ROOT, SUPPORTS
from servo_force import validate_position_servos
from target_load_ablation import digest


PROTOCOL_PATH = Path(__file__).with_name("short_step_protocol.json")
INITIAL_REPORT = ROOT / "output/rl/component_initial_state/2026-10-19_20-22-52-568702/report.json"


def hand_target(elapsed, initial, radius, protocol):
    target = initial.copy()
    if elapsed < .8:
        return target
    if elapsed < 1.4:
        target[2] += smooth((elapsed - .8) / .6) * (radius + protocol["reference_clearance_m"] - initial[2])
    elif elapsed < 2.:
        target[2] = radius + protocol["reference_clearance_m"]
        target[0] += protocol["step_length_m"] * smooth((elapsed - 1.4) / .6)
    else:
        target[0] += protocol["step_length_m"]
        target[2] = (1 - smooth((elapsed - 2.) / .6)) * (radius + protocol["reference_clearance_m"]) + smooth((elapsed - 2.) / .6) * initial[2]
    return target


def prepare():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    report = json.loads(INITIAL_REPORT.read_text())
    assert report["status"] == "completed"
    assert digest(report["model"]) == report["model_sha256"]
    assert digest(report["source_report"]) == report["source_report_sha256"]
    candidate = json.loads(Path(report["source_report"]).read_text())
    model_path = Path(report["model"])
    assert digest(model_path.with_name("manifest.json")) == candidate["manifest_sha256"]
    model_manifest = json.loads(model_path.with_name("manifest.json").read_text())
    mesh_hashes = {str(model_path.parent / "meshes" / name): expected for name, expected in model_manifest["mesh_sha256"].items()}
    for path, expected in mesh_hashes.items():
        assert digest(path) == expected
    entry = next(entry for entry in report["seeds"] if entry["seed"] == "old_equilibrium")
    assert entry["equilibrium"]["passed"] and entry["hold"]["reached"]
    assert digest(entry["hold"]["trajectory"]) == entry["hold"]["trajectory_sha256"]
    with np.load(entry["hold"]["trajectory"], allow_pickle=False) as saved:
        state, ctrl = saved["initial_physics_state"].copy(), saved["initial_ctrl"].copy()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    validate_position_servos(model)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    ids = [model.geom(name).id for name in SUPPORTS]
    targets = data.geom_xpos[ids].copy()
    radius = float(model.geom_size[ids[0], 0])
    assert radius == .02
    offsets, goals, errors = [], [], []
    limits = model.jnt_range[model.actuator_trnid[:, 0]]
    for elapsed in np.arange(round(protocol["duration_s"] / protocol["control_dt_s"]) + 1) * protocol["control_dt_s"]:
        desired = targets.copy()
        desired[0] = hand_target(elapsed, targets[0], radius, protocol)
        for unused in range(25):
            mujoco.mj_forward(model, data)
            matrices = []
            for geom in ids:
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jacGeom(model, data, jacobian, None, geom)
                matrices.append(jacobian[:, 6:])
            matrix = np.vstack(matrices)
            residual = (desired - data.geom_xpos[ids]).ravel()
            if np.max(abs(residual)) < 1e-5:
                break
            delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + np.eye(12) * 1e-5, residual)
            data.qpos[7:] = np.clip(data.qpos[7:] + np.clip(delta, -.03, .03), limits[:, 0], limits[:, 1])
        mujoco.mj_forward(model, data)
        offsets.append((data.qpos[7:] - initial_qpos[7:]).tolist())
        goals.append(desired[0].tolist())
        errors.append(float(np.max(np.linalg.norm(desired - data.geom_xpos[ids], axis=1))))
    payload = {"protocol": protocol, "protocol_sha256": digest(PROTOCOL_PATH), "builder_sha256": digest(__file__),
               "initial_report": str(INITIAL_REPORT), "initial_report_sha256": digest(INITIAL_REPORT),
               "model": str(model_path), "model_sha256": digest(model_path), "mesh_sha256": mesh_hashes,
               "initial_physics_state": state.tolist(), "initial_ctrl": ctrl.tolist(), "initial_qpos": initial_qpos.tolist(),
               "initial_support_positions": targets.tolist(), "joint_offsets": offsets, "hand_targets": goals,
               "ik_error_m": errors, "mujoco": mujoco.__version__,
               "limitations": "Kinematic soft reference only, not balanced or dynamically valid. No reset to reference frames. Model and reset frozen; contact-sensitivity and hull limitations retained. One hand placement is not a full crawling cycle."}
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode()
    folder = ROOT / "output/rl/short_step_assets" / hashlib.sha256(encoded).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    write_once(folder / "bundle.json", encoded)
    return folder / "bundle.json"


if __name__ == "__main__":
    argparse.ArgumentParser(description="Freeze native reset and kinematic short-step reference").parse_args()
    print(prepare())
