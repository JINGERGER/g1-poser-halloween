import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from forearm_alignment import load_meshes
from local_arm_geometry import capsule_distance, capsule_segment
from prepare import ROOT
from target_load_ablation import digest


def mesh_samples(model, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]].astype(float)
    start = model.mesh_faceadr[mesh_id]
    faces = model.mesh_face[start:start + model.mesh_facenum[mesh_id]]
    assert faces.min() >= 0 and faces.max() < len(vertices)
    return np.concatenate((vertices, vertices[faces].mean(axis=1)))


def world_points(points, position, rotation):
    return points @ rotation.reshape(3, 3).T + position


def union_distances(model, data, side, points):
    distances = [capsule_distance(points, *capsule_segment(model, data, f"{side}_{part}"))
                 for part in ("upper_arm", "forearm")]
    ball = model.geom(f"{side}_hand_collision")
    distances.append(np.linalg.norm(points - data.geom_xpos[ball.id], axis=1) - ball.size[0])
    return np.stack(distances, axis=1)


def summarize(distances):
    outside = distances.min(axis=1)
    return {"sample_count": len(outside), "outside_1mm_count": int(sum(outside > .001)),
            "max_outside_m": float(outside.max()), "p95_signed_distance_m": float(np.quantile(outside, .95)),
            "worst_sample_index": int(np.argmax(outside))}


def run(local_path):
    local = json.loads(local_path.read_text())
    assert local["status"] == "completed"
    assert digest(local["source_report"]) == local["source_report_sha256"]
    source = json.loads(Path(local["source_report"]).read_text())
    for key in ("alignment_report", "equilibrium_report", "base_model"):
        assert digest(source[key]) == source[key + "_sha256"]
    model_path = Path(local["model"])
    assert digest(model_path) == local["model_sha256"]
    variant = next(row for row in source["variants"] if row["model"] == str(model_path))
    assert digest(model_path.with_name("manifest.json")) == variant["manifest_sha256"]
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(model_path.parent / "meshes" / name) == expected
    alignment = json.loads(Path(source["alignment_report"]).read_text())
    assert digest(alignment["source_report"]) == alignment["source_report_sha256"]
    parts = json.loads(Path(alignment["source_report"]).read_text())
    part = next(part for part in parts["parts"] if Path(part["file"]).name == "forearm.3mf")
    visual = parts["current_visual_forearm"]
    assert digest(part["file"]) == part["sha256"] and digest(visual["file"]) == visual["sha256"]
    vertices, triangles, unused = load_meshes(Path(part["file"]), Path(visual["file"]))
    transform = alignment["selected"]
    vertices = world_points(vertices, np.array(transform["translation_m"]), np.array(transform["rotation_column_vector"]))
    shell = np.concatenate((vertices, vertices[triangles].mean(axis=1)))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    components = []
    for side in ("left", "right"):
        for part_name, suffix in (("upper_arm_visual", "shoulder_yaw_link"), ("elbow_visual", "elbow_link"),
                                  ("printed_shell_cad", "wrist_roll_rubber_hand")):
            body = model.body(f"{side}_{suffix}").id
            geoms = np.flatnonzero(model.geom_bodyid == body)
            visual_geoms = [int(geom) for geom in geoms if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH]
            assert len(visual_geoms) == 1
            geom = visual_geoms[0]
            samples = shell if part_name == "printed_shell_cad" else mesh_samples(model, geom)
            local_points = samples if part_name == "printed_shell_cad" else world_points(
                samples, model.geom_pos[geom], quaternion_matrix(model.geom_quat[geom]))
            components.append({"side": side, "part": part_name, "body": body, "geom": geom, "samples": samples,
                               "body_bounds_m": [local_points.min(axis=0).tolist(), local_points.max(axis=0).tolist()],
                               "own_active_geoms": [model.geom(int(index)).name for index in geoms
                                                    if model.geom_contype[index] or model.geom_conaffinity[index]]})
    poses = {"old_nominal": json.loads(Path(source["base_model"]).with_name("manifest.json").read_text())["qpos"],
             "old_equilibrium": json.loads(Path(source["equilibrium_report"]).read_text())["solution"]["qpos"]}
    data = mujoco.MjData(model)
    rows = []
    for specification in local["rows"]:
        data.qpos[:] = poses[specification["pose"]]
        data.qpos[2] += 1
        if specification["joint"] is not None:
            joint = model.joint(specification["joint"])
            data.qpos[joint.qposadr[0]] += specification["offset_rad"]
        mujoco.mj_forward(model, data)
        entries = []
        for component in components:
            body, geom = component["body"], component["geom"]
            is_cad = component["part"] == "printed_shell_cad"
            points = world_points(component["samples"], data.xpos[body] if is_cad else data.geom_xpos[geom],
                                  data.xmat[body] if is_cad else data.geom_xmat[geom])
            distances = union_distances(model, data, component["side"], points)
            summary = summarize(distances)
            worst = points[summary["worst_sample_index"]]
            summary["worst_sample_body_local_m"] = ((worst - data.xpos[body]) @ data.xmat[body].reshape(3, 3)).tolist()
            summary["individual_primitive_max_outside_m"] = distances.max(axis=0).tolist()
            entries.append({"side": component["side"], "part": component["part"], **summary})
        rows.append({key: specification[key] for key in ("pose", "joint", "offset_rad")} | {"components": entries})
    return {"status": "completed", "source_report": str(local_path.resolve()), "source_report_sha256": digest(local_path),
            "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                              (Path(__file__).name, "local_arm_geometry.py", "forearm_alignment.py")},
            "model": str(model_path), "model_sha256": digest(model_path), "dynamic_rollouts": 0,
            "components": [{key: value for key, value in component.items() if key != "samples"} for component in components],
            "rows": rows, "primitive_order": ["upper_arm", "forearm", "hand_collision"],
            "limitations": "Coverage of CAD/visual surface samples by same-side three-primitive union only, not full-robot collision coverage, surface area, solid intersection or hardware measurement. Mesh compiled coordinates use geom transforms including compiler mesh recentering. No collision model, exclusions, initial states or controller changed."}


def quaternion_matrix(quaternion):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("local_report", type=Path)
    args = parser.parse_args()
    report = run(args.local_report)
    folder = ROOT / "output/rl/arm_collision_coverage" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(folder / "report.json")
    for component in report["components"]:
        entries = [entry for row in report["rows"] for entry in row["components"]
                   if entry["side"] == component["side"] and entry["part"] == component["part"]]
        print(json.dumps(component | {"sample_count": entries[0]["sample_count"],
                                    "outside_count_range": [min(entry["outside_1mm_count"] for entry in entries), max(entry["outside_1mm_count"] for entry in entries)],
                                    "worst_outside_m": max(entry["max_outside_m"] for entry in entries)}))


if __name__ == "__main__":
    main()
