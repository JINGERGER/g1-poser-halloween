import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from ball_contact_assets import write_once
from forearm_alignment import load_meshes
from prepare import ROOT
from servo_force import validate_position_servos
from target_load_ablation import digest


def hull_obj(vertices):
    hull = ConvexHull(vertices)
    lines = ["v " + " ".join(format(value, ".17g") for value in vertex) for vertex in vertices]
    for face, equation in zip(hull.simplices, hull.equations):
        face = face.copy()
        points = vertices[face]
        if np.cross(points[1] - points[0], points[2] - points[0]) @ equation[:3] < 0:
            face[[1, 2]] = face[[2, 1]]
        lines.append("f " + " ".join(str(int(index) + 1) for index in face))
    return ("\n".join(lines) + "\n").encode()


def build(source_path):
    source = json.loads(source_path.read_text())
    assert source["status"] == "completed"
    assert digest(source["model"]) == source["model_sha256"]
    assert digest(source["source_report"]) == source["source_report_sha256"]
    local = json.loads(Path(source["source_report"]).read_text())
    assert digest(local["source_report"]) == local["source_report_sha256"]
    balls = json.loads(Path(local["source_report"]).read_text())
    assert digest(balls["alignment_report"]) == balls["alignment_report_sha256"]
    alignment = json.loads(Path(balls["alignment_report"]).read_text())
    assert digest(alignment["source_report"]) == alignment["source_report_sha256"]
    parts = json.loads(Path(alignment["source_report"]).read_text())
    part = next(part for part in parts["parts"] if Path(part["file"]).name == "forearm.3mf")
    visual = parts["current_visual_forearm"]
    assert digest(part["file"]) == part["sha256"] and digest(visual["file"]) == visual["sha256"]
    vertices, unused, unused_stl = load_meshes(Path(part["file"]), Path(visual["file"]))
    transform = alignment["selected"]
    vertices = vertices @ np.array(transform["rotation_column_vector"]).T + transform["translation_m"]
    base_path = Path(source["model"])
    variant = next(row for row in balls["variants"] if row["model"] == str(base_path))
    assert digest(base_path.with_name("manifest.json")) == variant["manifest_sha256"]
    old_manifest = json.loads(base_path.with_name("manifest.json").read_text())
    meshes = {}
    for name, expected in old_manifest["mesh_sha256"].items():
        path = base_path.parent / "meshes" / name
        assert digest(path) == expected
        meshes[name] = path.read_bytes()
    meshes["printed_shell_hull.obj"] = hull_obj(vertices)
    root = ET.parse(base_path).getroot()
    root.set("model", "g1_component_hulls_v1")
    root.find("compiler").set("inertiafromgeom", "false")
    ET.SubElement(root.find("asset"), "mesh", name="printed_shell_hull", file="printed_shell_hull.obj", maxhullvert="-1")
    for side in ("left", "right"):
        for name, mesh in ((f"{side}_upper_arm", f"{side}_shoulder_yaw_link"), (f"{side}_forearm", "printed_shell_hull")):
            geom = root.find(f".//geom[@name='{name}']")
            geom.attrib.clear()
            geom.attrib.update(name=name, type="mesh", mesh=mesh, density="0", **{"class": "collision"})
        elbow = root.find(f".//body[@name='{side}_elbow_link']")
        ET.SubElement(elbow, "geom", name=f"{side}_elbow_collision", type="mesh", mesh=f"{side}_elbow_link", density="0", **{"class": "collision"})
        for mesh_name in (f"{side}_shoulder_yaw_link", f"{side}_elbow_link"):
            root.find(f"./asset/mesh[@name='{mesh_name}']").set("maxhullvert", "-1")
    contact = root.find("contact")
    removed = []
    for exclude in list(contact.findall("exclude")):
        if any(exclude.get("body1") == f"{side}_shoulder_yaw_link" and exclude.get("body2") == f"{side}_wrist_roll_rubber_hand" for side in ("left", "right")):
            removed.append(dict(exclude.attrib))
            contact.remove(exclude)
    assert len(removed) == 2
    xml = ET.tostring(root, encoding="utf-8")
    manifest = {"source_report": str(source_path.resolve()), "source_report_sha256": digest(source_path),
                "base_model": str(base_path), "base_model_sha256": digest(base_path),
                "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in (Path(__file__).name, "forearm_alignment.py", "ball_contact_assets.py")},
                "mesh_sha256": {name: hashlib.sha256(contents).hexdigest() for name, contents in meshes.items()},
                "removed_exclusions": removed, "candidate_only": True,
                "limitations": "Per-component conservative convex hulls fill cavities and mounting recesses. Default adjacent-body filtering retained and explicitly probed. Not calibrated hardware, not a valid reset. No inertia, actuator or ball changes."}
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode()
    folder = ROOT / "output/rl/models" / ("component_hulls_v1_" + hashlib.sha256(xml + encoded).hexdigest()[:16])
    (folder / "meshes").mkdir(parents=True, exist_ok=True)
    for name, contents in meshes.items():
        write_once(folder / "meshes" / name, contents)
    write_once(folder / "flat.xml", xml)
    write_once(folder / "manifest.json", encoded)
    return folder, local, balls


def inspect_pose(model, pose):
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    data.qpos[2] += 1
    mujoco.mj_forward(model, data)
    floor = model.geom("floor").id
    active = [{"pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name], "distance_m": float(contact.dist)}
              for contact in data.contact if floor not in (contact.geom1, contact.geom2) and contact.dist < -.001]
    direct = []
    for side in ("left", "right"):
        for first, second in (("upper_arm", "elbow_collision"), ("elbow_collision", "forearm"), ("upper_arm", "forearm")):
            names = [f"{side}_{first}", f"{side}_{second}"]
            distance = mujoco.mj_geomDistance(model, data, model.geom(names[0]).id, model.geom(names[1]).id, 1, None)
            direct.append({"pair": names, "distance_m": float(distance)})
    return {"active_self_contacts": active, "direct_component_distances": direct}


def evaluate(folder, local, balls):
    model = mujoco.MjModel.from_xml_path(str(folder / "flat.xml"))
    base = mujoco.MjModel.from_xml_path(local["model"])
    validate_position_servos(model)
    fields = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "jnt_range", "jnt_actfrcrange",
              "dof_armature", "dof_damping", "actuator_trnid", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "actuator_gear")
    for field in fields:
        np.testing.assert_array_equal(getattr(base, field), getattr(model, field))
    assert model.nu == 23 and model.nexclude == base.nexclude - 2
    assert digest(balls["equilibrium_report"]) == balls["equilibrium_report_sha256"]
    nominal = json.loads(Path(balls["base_model"]).with_name("manifest.json").read_text())["qpos"]
    equilibrium = json.loads(Path(balls["equilibrium_report"]).read_text())["solution"]["qpos"]
    rows = []
    for row in local["rows"]:
        pose = np.array(nominal if row["pose"] == "old_nominal" else equilibrium)
        if row["joint"] is not None:
            pose[model.joint(row["joint"]).qposadr[0]] += row["offset_rad"]
        rows.append({key: row[key] for key in ("pose", "joint", "offset_rad")} | inspect_pose(model, pose))
    sweep = []
    for joint in model.actuator_trnid[:, 0]:
        for fraction in np.linspace(0, 1, 5):
            pose = np.array(nominal)
            pose[model.jnt_qposadr[joint]] = model.jnt_range[joint] @ np.array([1 - fraction, fraction])
            sweep.append({"joint": model.joint(int(joint)).name, "fraction": float(fraction)} | inspect_pose(model, pose))
    return {"unchanged_fields": list(fields), "local": rows, "full_limit_sweep": sweep,
            "local_active_collision_samples": sum(bool(row["active_self_contacts"]) for row in rows),
            "full_limit_active_collision_samples": sum(bool(row["active_self_contacts"]) for row in sweep)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_report", type=Path)
    args = parser.parse_args()
    folder, local, balls = build(args.coverage_report)
    result = evaluate(folder, local, balls)
    report = {"status": "completed", "model": str(folder / "flat.xml"), "model_sha256": digest(folder / "flat.xml"),
              "manifest_sha256": digest(folder / "manifest.json"), "source_report": str(args.coverage_report.resolve()),
              "source_report_sha256": digest(args.coverage_report), "dynamic_rollouts": 0, **result}
    destination = ROOT / "output/rl/arm_hull_candidate" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination / "report.json")
    print(json.dumps({key: value for key, value in report.items() if key not in ("local", "full_limit_sweep")}))


if __name__ == "__main__":
    main()
