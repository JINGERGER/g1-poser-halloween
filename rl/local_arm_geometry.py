import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from forearm_alignment import load_meshes
from prepare import ROOT
from target_load_ablation import digest


def capsule_distance(points, first, second, radius):
    axis = second - first
    fraction = np.clip((points - first) @ axis / max(float(axis @ axis), 1e-30), 0, 1)
    return np.linalg.norm(points - first - fraction[:, None] * axis, axis=1) - radius


def capsule_segment(model, data, name):
    geom = model.geom(name)
    axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2] * geom.size[1]
    return data.geom_xpos[geom.id] - axis, data.geom_xpos[geom.id] + axis, geom.size[0]


def evaluate(model, qpos, shell):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qpos[2] += 1
    mujoco.mj_forward(model, data)
    result = {}
    for side in ("left", "right"):
        body = model.body(f"{side}_wrist_roll_rubber_hand").id
        rotation = data.xmat[body].reshape(3, 3)
        world = shell @ rotation.T + data.xpos[body]
        upper = capsule_segment(model, data, f"{side}_upper_arm")
        distances = capsule_distance(world, *upper)
        endpoints = np.zeros(6)
        pair_distance = mujoco.mj_geomDistance(model, data, model.geom(f"{side}_upper_arm").id,
                                             model.geom(f"{side}_forearm").id, 1, endpoints)
        result[side] = {"capsule_pair_distance_m": float(pair_distance),
                        "cad_sample_to_upper_capsule_min_m": float(distances.min()),
                        "cad_samples_inside_upper_capsule_1mm": int(sum(distances < -.001)),
                        "closest_cad_sample_local_m": shell[np.argmin(distances)].tolist(),
                        "capsule_witness_points_forearm_local_m": ((endpoints.reshape(2, 3) - data.xpos[body]) @ rotation).tolist()}
    floor = model.geom("floor").id
    result["active_self_contacts"] = [{"pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name],
                                       "distance_m": float(contact.dist)} for contact in data.contact
                                      if floor not in (contact.geom1, contact.geom2) and contact.dist < -.001]
    return result


def run(source_path):
    source = json.loads(source_path.read_text())
    assert source["status"] == "completed"
    for key in ("alignment_report", "equilibrium_report", "base_model"):
        assert digest(source[key]) == source[key + "_sha256"]
    alignment = json.loads(Path(source["alignment_report"]).read_text())
    assert digest(alignment["source_report"]) == alignment["source_report_sha256"]
    parts = json.loads(Path(alignment["source_report"]).read_text())
    part = next(part for part in parts["parts"] if Path(part["file"]).name == "forearm.3mf")
    visual = parts["current_visual_forearm"]
    assert digest(part["file"]) == part["sha256"] and digest(visual["file"]) == visual["sha256"]
    vertices, triangles, unused = load_meshes(Path(part["file"]), Path(visual["file"]))
    transform = alignment["selected"]
    vertices = vertices @ np.array(transform["rotation_column_vector"]).T + transform["translation_m"]
    shell = np.concatenate((vertices, vertices[triangles].mean(axis=1)))
    variant = next(row for row in source["variants"] if row["variant"] == "reference_center")
    assert digest(variant["model"]) == variant["model_sha256"]
    model_path = Path(variant["model"])
    assert digest(model_path.with_name("manifest.json")) == variant["manifest_sha256"]
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(model_path.parent / "meshes" / name) == expected
    model = mujoco.MjModel.from_xml_path(str(model_path))
    nominal = json.loads(Path(source["base_model"]).with_name("manifest.json").read_text())["qpos"]
    equilibrium = json.loads(Path(source["equilibrium_report"]).read_text())["solution"]["qpos"]
    rows = []
    for label, pose in (("old_nominal", nominal), ("old_equilibrium", equilibrium)):
        rows.append({"pose": label, "joint": None, "offset_rad": 0, "result": evaluate(model, pose, shell)})
        for joint_id in model.actuator_trnid[:, 0]:
            name = model.joint(int(joint_id)).name
            if not any(part in name for part in ("shoulder", "elbow", "wrist")):
                continue
            address = model.jnt_qposadr[joint_id]
            for offset in (-.1, .1):
                changed = np.array(pose)
                changed[address] = np.clip(changed[address] + offset, *model.jnt_range[joint_id])
                rows.append({"pose": label, "joint": name, "offset_rad": float(changed[address] - pose[address]),
                             "result": evaluate(model, changed, shell)})
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body("left_wrist_roll_rubber_hand").id
    first, second, radius = capsule_segment(model, data, "left_forearm")
    segment = (np.array([first, second]) - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    coverage = capsule_distance(shell, segment[0], segment[1], radius)
    return {"status": "completed", "source_report": str(source_path.resolve()), "source_report_sha256": digest(source_path),
            "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in (Path(__file__).name, "forearm_alignment.py")},
            "model": str(model_path), "model_sha256": digest(model_path), "dynamic_rollouts": 0,
            "cad_surface_samples": len(shell), "cad_local_bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "forearm_capsule_axis_local_m": segment.tolist(), "forearm_capsule_radius_m": float(radius),
            "cad_samples_outside_forearm_capsule_1mm": int(sum(coverage > .001)),
            "maximum_sample_outside_forearm_capsule_m": float(coverage.max()), "rows": rows,
            "limitations": "CAD vertices and triangle centroids versus capsules, not solid-solid collision proof. Positive sampled clearance is not exhaustive clearance. Forearm capsule may cover elbow hardware as well as printed shell. Old poses are geometric probes, not validated new-model resets. Single-joint +/-0.1rad probes are not trajectories. No model changed."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_report", type=Path)
    args = parser.parse_args()
    report = run(args.source_report)
    folder = ROOT / "output/rl/local_arm_geometry" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(folder / "report.json")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    for row in report["rows"]:
        if row["joint"] is None:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
