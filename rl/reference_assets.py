import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from direct_task import prepare_direct
from prepare import ASSETS, ROOT, SUPPORTS


REFERENCE_COMMIT = "935e8b9c857bd494b952603c0453800a0cf5c9d2"


def joint_mapping(names, target_names):
    if len(names) != 23 or len(set(names)) != 23 or set(names) != set(target_names):
        raise ValueError("Expected the same 23 unique joint names")
    return np.array([target_names.index(name) for name in names], dtype=int)


def prepare_reference(pd_variant="current"):
    if pd_variant not in ("current", "reference"):
        raise ValueError("Unknown PD variant")
    audit = json.loads((ROOT / "output/rl/reference_audit.json").read_text())
    if audit["reference_commit"] != REFERENCE_COMMIT:
        raise ValueError("Reference audit must use the pinned commit")
    if audit["model_sha256"] != hashlib.sha256((ASSETS / "flat.xml").read_bytes()).hexdigest():
        raise ValueError("Model changed: rerun audit_reference.py")
    repository = ROOT.parent / "g1_crawl"
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain=v1"])
    content = subprocess.check_output(["git", "-C", str(repository), "show", f"{REFERENCE_COMMIT}:assets/animation_rc4.json"])
    animation = json.loads(content)
    if animation["schema"] != "gait_animation.v1" or animation["nq"] != 30:
        raise ValueError("Unexpected animation schema")
    source = prepare_direct(3)
    robot = ET.parse(source / "flat.xml").getroot()
    comparison = {row["joint"]: row for row in audit["actuator_comparison"]}
    if pd_variant == "reference":
        for actuator in robot.find("actuator"):
            if actuator.tag != "position":
                raise ValueError("Expected position actuators")
            row = comparison[actuator.get("joint")]
            actuator.set("kp", str(row["reference_training_kp"]))
            actuator.set("kv", str(row["reference_training_kd"]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    names = [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]]
    mapping = joint_mapping(audit["policy_order"], names)
    metadata = animation["metadata"]
    if metadata["base"] != {"pos_indices": [0, 1, 2], "quat_indices": [3, 4, 5, 6]}:
        raise ValueError("Unexpected root frame layout")
    joint_metadata = [item for item in metadata["joints"] if item["type"] == "hinge" and item["qposdim"] == 1]
    joint_mapping(names, [item["name"] for item in joint_metadata])
    indices = {item["name"]: item["qposadr"] for item in joint_metadata}
    frames = np.asarray(animation["frames"], dtype=float)
    if frames.ndim != 2 or frames.shape[1] != 30 or not len(frames) or not np.isfinite(frames).all():
        raise ValueError("Invalid animation frames")
    frames = frames[:, [*range(7), *(indices[name] for name in names)]]
    if not np.allclose(np.linalg.norm(frames[:, 3:7], axis=1), 1, atol=1e-5):
        raise ValueError("Animation quaternions are not normalized")
    sensors = robot.find("sensor")
    for index, name in enumerate(SUPPORTS):
        body = model.body(int(model.geom_bodyid[model.geom(name).id])).name
        ET.SubElement(sensors, "framelinvel", name=f"reference_support_vel_{index}", objtype="body", objname=body)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["reference"] = {
        "protocol": "reference-inspired-75-81-v1", "pd_variant": pd_variant,
        "commit": REFERENCE_COMMIT, "animation_sha256": hashlib.sha256(content).hexdigest(),
        "policy_order": audit["policy_order"], "policy_to_actuator": mapping.tolist(),
        "default_angles": [comparison[name]["reference_default_angle"] for name in names],
        "frame_count": len(frames), "original_model_sha256": audit["model_sha256"],
        "reset_semantics": "implemented intent; upstream function does not write state; joint velocities zero; root velocity noise only; floor projection",
        "reward_contact_semantics": "endpoint geom-found flags, grouped by body; not Isaac force-history threshold",
        "symmetry": "disabled in stage-0 simplified baseline",
        "physics_randomization": "disabled in stage-0 simplified baseline",
    }
    xml = ET.tostring(robot, encoding="unicode")
    mujoco.MjModel.from_xml_string(xml)
    metadata_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
    digest = hashlib.sha256(xml.encode() + metadata_bytes + frames.tobytes()).hexdigest()[:16]
    destination = ROOT / "output/rl/models" / f"reference_{digest}"
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in (("flat.xml", xml.encode()), ("manifest.json", metadata_bytes)):
        path = destination / name
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("Immutable model snapshot mismatch")
        if not path.exists():
            path.write_bytes(contents)
    frame_path = destination / "reset_frames.npy"
    if frame_path.exists():
        if not np.array_equal(np.load(frame_path, allow_pickle=False), frames):
            raise ValueError("Immutable animation snapshot mismatch")
    else:
        np.save(frame_path, frames, allow_pickle=False)
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain=v1"]) != status:
        raise RuntimeError("Reference worktree changed during preparation")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd", choices=("current", "reference"), default="current")
    print(prepare_reference(parser.parse_args().pd))
