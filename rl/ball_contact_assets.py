import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from load_transfer import physical_metrics
from prepare import ROOT
from servo_force import validate_position_servos
from target_load_ablation import digest


VARIANTS = {"reference_center": .18, "visual_center": .1822}


def write_once(path, contents):
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f"Immutable snapshot mismatch: {path}")
    else:
        path.write_bytes(contents)


def build_variant(base, alignment_path, variant):
    alignment = json.loads(alignment_path.read_text())
    assert alignment["status"] == "completed" and alignment["stl_identical_to_reference_commit"]
    assert abs(alignment["fitted_sphere"]["radius_m"] - .02) < 1e-6
    np.testing.assert_allclose(alignment["fitted_sphere"]["center_stl_m"], [.1822, 0, 0], atol=1e-6, rtol=0)
    for name, expected in alignment["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(alignment["source_report"]) == alignment["source_report_sha256"]
    parts = json.loads(Path(alignment["source_report"]).read_text())
    for part in parts["parts"]:
        assert digest(part["file"]) == part["sha256"]
    assert digest(parts["current_visual_forearm"]["file"]) == parts["current_visual_forearm"]["sha256"]
    root = ET.parse(base).getroot()
    meshdir = Path(root.find("compiler").get("meshdir", "."))
    if not meshdir.is_absolute():
        meshdir = base.parent / meshdir
    meshes = {}
    for mesh in root.findall("./asset/mesh"):
        path = meshdir / mesh.get("file")
        assert path.name not in meshes
        meshes[path.name] = path
        mesh.set("file", path.name)
    root.find("compiler").set("meshdir", "meshes")
    edits = []
    for side in ("left", "right"):
        geom = root.find(f".//geom[@name='{side}_hand_collision']")
        before = dict(geom.attrib)
        geom.set("size", "0.02")
        geom.set("pos", f"{VARIANTS[variant]} 0 0")
        edits.append({"geom": geom.get("name"), "before": before, "after": dict(geom.attrib)})
    root.set("model", f"g1_ball_contact_v1_{variant}")
    xml = ET.tostring(root, encoding="utf-8")
    manifest = {"protocol": "ball-contact-v1", "variant": variant, "base_model": str(base.resolve()), "base_model_sha256": digest(base),
                "alignment_report": str(alignment_path.resolve()), "alignment_report_sha256": digest(alignment_path),
                "builder_sha256": digest(__file__), "edits": edits,
                "mesh_sha256": {name: digest(path) for name, path in meshes.items()},
                "provenance": {"radius_m": "20mm supported by reference XML and visual sphere fit",
                               "center_m": "reference XML" if variant == "reference_center" else "visual mesh fit rounded to micrometers; not measured mounting"},
                "unvalidated": ["training-only upper-arm and forearm capsule approximations retained", "collision exclusions retained and separately probed",
                                "mass/inertia retained, printed assembly coverage unconfirmed", "contact friction/compliance not calibrated",
                                "motor speed/power/thermal limits not calibrated", "no valid loaded initial state supplied"],
                "limitations": "Ball-geometry correction only, not full hardware model. Source snapshots and default app scene unchanged. All meshes copied and hashed; old loaded states are diagnostic probes, never valid reset states."}
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode()
    identifier = hashlib.sha256(xml + encoded).hexdigest()[:16]
    destination = ROOT / "output/rl/models" / f"ball_contact_v1_{variant}_{identifier}"
    (destination / "meshes").mkdir(parents=True, exist_ok=True)
    for name, path in meshes.items():
        write_once(destination / "meshes" / name, path.read_bytes())
    write_once(destination / "flat.xml", xml)
    write_once(destination / "manifest.json", encoded)
    model = mujoco.MjModel.from_xml_path(str(destination / "flat.xml"))
    validate_position_servos(model)
    return destination


def compare_models(base, updated):
    fields = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
              "jnt_range", "jnt_actfrcrange", "dof_armature", "dof_damping", "actuator_ctrlrange",
              "actuator_gainprm", "actuator_biasprm", "actuator_gear", "geom_type", "geom_quat",
              "geom_contype", "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp", "exclude_signature")
    for field in fields:
        np.testing.assert_array_equal(getattr(base, field), getattr(updated, field))
    changed = []
    for index in range(base.ngeom):
        if not np.array_equal(base.geom_size[index], updated.geom_size[index]) or not np.array_equal(base.geom_pos[index], updated.geom_pos[index]):
            changed.append(base.geom(index).name)
    assert sorted(changed) == ["left_hand_collision", "right_hand_collision"]
    return {"unchanged_numeric_fields": list(fields), "changed_geoms": changed}


def probe(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    row = physical_metrics(model, data)
    return {"gaps_m": row["gaps_m"], "floor_contacts": row["floor_contacts"], "max_penetration_m": row["penetration"]["depth_m"],
            "contacts": [{"pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name], "distance_m": float(contact.dist)} for contact in data.contact]}


def self_collision_sweep(model, nominal):
    data = mujoco.MjData(model)
    active = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
    pairs = []
    for side in ("left", "right"):
        first = model.body(f"{side}_shoulder_yaw_link").id
        second = model.body(f"{side}_wrist_roll_rubber_hand").id
        pairs.extend((int(left), int(right)) for left in active for right in active
                     if model.geom_bodyid[left] == first and model.geom_bodyid[right] == second)
    floor = model.geom("floor").id
    rows = []
    for joint in model.actuator_trnid[:, 0]:
        for fraction in np.linspace(0, 1, 5):
            data.qpos[:] = nominal
            data.qpos[2] += 1
            data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, 0] * (1 - fraction) + model.jnt_range[joint, 1] * fraction
            mujoco.mj_forward(model, data)
            contacts = [{"pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name], "distance_m": float(contact.dist)}
                        for contact in data.contact if floor not in (contact.geom1, contact.geom2) and contact.dist < -.001]
            excluded = [{"pair": [model.geom(first).name, model.geom(second).name], "distance_m": float(mujoco.mj_geomDistance(model, data, first, second, 1, None))}
                        for first, second in pairs]
            rows.append({"joint": model.joint(int(joint)).name, "fraction": float(fraction), "active_penetrations": contacts,
                         "excluded_pair_distances": excluded})
    return {"samples": len(rows), "active_collision_samples": sum(bool(row["active_penetrations"]) for row in rows),
            "excluded_overlap_samples": sum(any(pair["distance_m"] < -.001 for pair in row["excluded_pair_distances"]) for row in rows),
            "rows": rows, "limitations": "One-joint-at-a-time full-limit geometric probes, not trajectories, joint-limit combinations or hardware-safe commands. Explicit distance checks bypass the named exclusion pairs only; other filtering is not exhaustively audited."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alignment_report", type=Path)
    parser.add_argument("equilibrium_report", type=Path)
    args = parser.parse_args()
    equilibrium = json.loads(args.equilibrium_report.read_text())
    assert equilibrium["status"] == "completed"
    chain = equilibrium
    for key in ("audit_report", "source_report", "baseline_report", "geometry_report", "source_report"):
        assert digest(chain[key]) == chain[key + "_sha256"]
        chain = json.loads(Path(chain[key]).read_text())
    base_path = Path(chain["model"])
    assert digest(base_path) == chain["model_sha256"]
    before = digest(base_path)
    base = mujoco.MjModel.from_xml_path(str(base_path))
    manifest = json.loads(base_path.with_name("manifest.json").read_text())
    destination = ROOT / "output/rl/ball_contact_assets" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "variants": [], "training_updates": 0, "dynamic_rollouts": 0,
              "source_sha256": {Path(__file__).name: digest(__file__)}, "base_model": str(base_path), "base_model_sha256": before,
              "equilibrium_report": str(args.equilibrium_report.resolve()), "equilibrium_report_sha256": digest(args.equilibrium_report),
              "alignment_report": str(args.alignment_report.resolve()), "alignment_report_sha256": digest(args.alignment_report),
              "old_equilibrium_probe": probe(base, equilibrium["solution"]["qpos"]),
              "base_sweep": self_collision_sweep(base, manifest["qpos"])}
    try:
        for variant in VARIANTS:
            folder = build_variant(base_path, args.alignment_report, variant)
            model = mujoco.MjModel.from_xml_path(str(folder / "flat.xml"))
            comparison = compare_models(base, model)
            row = {"variant": variant, "model": str(folder / "flat.xml"), "model_sha256": digest(folder / "flat.xml"),
                   "manifest_sha256": digest(folder / "manifest.json"), "comparison": comparison,
                   "old_equilibrium_geometry_probe": probe(model, equilibrium["solution"]["qpos"]),
                   "nominal_geometry_probe": probe(model, manifest["qpos"]), "sweep": self_collision_sweep(model, manifest["qpos"])}
            report["variants"].append(row)
            print(json.dumps({"variant": variant, "model": row["model"], "old_pose_gaps_m": row["old_equilibrium_geometry_probe"]["gaps_m"],
                              "active_collision_samples": row["sweep"]["active_collision_samples"], "excluded_overlap_samples": row["sweep"]["excluded_overlap_samples"]}), flush=True)
        assert digest(base_path) == before
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
