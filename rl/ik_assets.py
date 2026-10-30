import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare import ROOT, SITES, SUPPORTS


PERIOD = 4.0
SPEED = 0.03
LIFT = 0.02
FRAMES = 128


def motion_time(elapsed):
    moving = np.maximum(np.asarray(elapsed, dtype=float) - 1, 0)
    ramp = np.minimum(moving, 1)
    return ramp ** 3 - 0.5 * ramp ** 4 + np.maximum(moving - 1, 0)


def foot_path(phase):
    local = (np.asarray(phase)[..., None] + [0, 0.5, 0.75, 0.25]) % 1
    swing = local >= 0.75
    fraction = np.where(swing, (local - 0.75) / 0.25, local / 0.75)
    smooth = fraction ** 3 * (10 - 15 * fraction + 6 * fraction ** 2)
    stride = SPEED * PERIOD * 0.75
    horizontal = stride * np.where(swing, smooth - 0.5, 0.5 - smooth)
    height = np.where(swing, LIFT * np.sin(np.pi * fraction) ** 2, 0)
    return horizontal, height, swing


def reference_at(table, anchor_ids, elapsed):
    phase = motion_time(elapsed) / PERIOD % 1
    frame = phase * FRAMES
    lower = np.floor(frame).astype(int) % FRAMES
    blend = (frame - np.floor(frame))[..., None]
    return (1 - blend) * table[anchor_ids, lower] + blend * table[anchor_ids, (lower + 1) % FRAMES]


def build(report_path):
    report = json.loads(report_path.read_text())
    entries = [row for row in report["runs"] if row["branch"] == "freeze"]
    if report["status"] != "completed" or len(entries) != 2 or not all(row["anchor"]["ready"] for row in entries):
        raise ValueError("Need the two verified support anchors")
    source = Path(entries[0]["model"])
    model_bytes = source.read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    if any(row["model_sha256"] != model_hash for row in entries):
        raise ValueError("Both anchors must use the exact same immutable model")
    model = mujoco.MjModel.from_xml_path(str(source))
    joints = model.actuator_trnid[:, 0]
    if not np.array_equal(model.jnt_qposadr[joints], np.arange(7, 30)):
        raise ValueError("Unexpected joint ordering")
    qpos = np.array([row["anchor"]["qpos"] for row in entries])
    qvel = np.array([row["anchor"]["qvel"] for row in entries])
    ctrl = np.array([row["anchor"]["previous_ctrl"] for row in entries])
    table = np.zeros((2, FRAMES, 23), dtype=float)
    floor = model.geom("floor").id
    support_ids = {model.geom(name).id for name in SUPPORTS}
    diagnostics = []
    for anchor_id, anchor in enumerate(qpos):
        data = mujoco.MjData(model)
        data.qpos[:] = anchor
        mujoco.mj_forward(model, data)
        sites = np.array([data.site(name).xpos.copy() for name in SITES])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, anchor[3:7])
        heading = rotation.reshape(3, 3)[:2, 2].copy()
        heading /= np.linalg.norm(heading)
        horizontal_zero, _, _ = foot_path(0.0)
        maximum_error = 0.0
        for frame in range(FRAMES):
            horizontal, height, _ = foot_path(frame / FRAMES)
            targets = sites.copy()
            targets[:, :2] += (horizontal - horizontal_zero)[:, None] * heading
            targets[:, 2] += height
            data.qpos[:] = anchor
            jacobian = np.zeros((3, model.nv))
            for iteration in range(50):
                mujoco.mj_forward(model, data)
                matrices, errors = [], []
                for site, target in zip(SITES, targets):
                    mujoco.mj_jacSite(model, data, jacobian, None, model.site(site).id)
                    matrices.append(jacobian[:, 6:].copy())
                    errors.append(target - data.site(site).xpos)
                matrix = np.vstack(matrices)
                delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + np.eye(12) * 1e-5, np.concatenate(errors))
                data.qpos[7:] = np.clip(data.qpos[7:] + np.clip(delta, -0.03, 0.03), model.jnt_range[joints, 0], model.jnt_range[joints, 1])
            mujoco.mj_forward(model, data)
            maximum_error = max(maximum_error, float(np.max(np.linalg.norm(np.array([data.site(name).xpos for name in SITES]) - targets, axis=1))))
            table[anchor_id, frame] = data.qpos[7:] - anchor[7:]
        unwanted_penetration = 0.0
        for frame in np.arange(FRAMES * 4) / 4:
            lower = int(frame)
            blend = frame - lower
            data.qpos[:] = anchor
            data.qpos[7:] += (1 - blend) * table[anchor_id, lower] + blend * table[anchor_id, (lower + 1) % FRAMES]
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                normal_support = floor in (contact.geom1, contact.geom2) and (contact.geom1 in support_ids or contact.geom2 in support_ids)
                if not normal_support:
                    unwanted_penetration = max(unwanted_penetration, -float(contact.dist))
        velocities = (np.roll(table[anchor_id], -1, axis=0) - table[anchor_id]) / (PERIOD / FRAMES)
        accelerations = (np.roll(velocities, -1, axis=0) - velocities) / (PERIOD / FRAMES)
        diagnostics.append({"ik_max_error_m": maximum_error, "unwanted_penetration_m": unwanted_penetration,
                            "max_reference_speed_radps": float(np.max(np.abs(velocities))),
                            "max_discrete_acceleration_radps2": float(np.max(np.abs(accelerations))),
                            "periodic_last_step_rad": float(np.max(np.abs(table[anchor_id, -1] - table[anchor_id, 0]))),
                            "start_offset_rad": float(np.max(np.abs(table[anchor_id, 0])))})
        if maximum_error > 0.005 or unwanted_penetration > 0.003 or np.max(np.abs(velocities)) > 10:
            raise ValueError(f"IK geometry/continuity gate failed: {diagnostics[-1]}")
    manifest = json.loads(source.with_name("manifest.json").read_text())
    manifest["ik_tracking"] = {"protocol": "ik-soft-reference-v1", "period": PERIOD, "speed": SPEED, "lift": LIFT,
                               "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                               "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "model_sha256": model_hash, "diagnostics": diagnostics,
                               "limitations": "Geometric reference only; linear joint interpolation has finite velocity jumps at knots. No torque/contact feasibility claim, no velocity-limit specification validated for hardware."}
    digest = hashlib.sha256(model_bytes + json.dumps(manifest, sort_keys=True).encode() + table.tobytes()).hexdigest()[:16]
    destination = ROOT / "output/rl/models" / f"ik_tracking_{digest}"
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("flat.xml", model_bytes), ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())):
        path = destination / name
        if path.exists() and path.read_bytes() != value:
            raise ValueError("Immutable snapshot mismatch")
        if not path.exists():
            path.write_bytes(value)
    arrays = {"qpos": qpos, "qvel": qvel, "ctrl": ctrl, "offsets": table}
    path = destination / "ik_reference.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[name], value) for name, value in arrays.items()):
                raise ValueError("Immutable reference mismatch")
    else:
        np.savez_compressed(path, **arrays)
    print(json.dumps(manifest["ik_tracking"], indent=2))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("support_report", type=Path)
    print(build(parser.parse_args().support_report))
