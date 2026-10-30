import argparse
from datetime import datetime
import json
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from prepare import ROOT
from target_load_ablation import digest


NAMESPACE = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
UNITS = {"meter": 1., "millimeter": .001, "centimeter": .01, "micron": 1e-6, "inch": .0254, "foot": .3048}
REFERENCE_COMMIT = "935e8b9c857bd494b952603c0453800a0cf5c9d2"


def mesh_summary(vertices, triangles):
    assert vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
    assert triangles.ndim == 2 and triangles.shape[1] == 3 and triangles.min() >= 0 and triangles.max() < len(vertices)
    facets = vertices[triangles]
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    balanced = np.bincount(inverse, weights=signs)
    closed = bool(np.all(counts == 2) and np.all(balanced == 0))
    signed_volume = float(np.sum(np.einsum("ij,ij->i", facets[:, 0], np.cross(facets[:, 1], facets[:, 2]))) / 6)
    return {"vertex_count": len(vertices), "triangle_count": len(triangles), "bounds_min_m": vertices.min(axis=0).tolist(),
            "bounds_max_m": vertices.max(axis=0).tolist(), "axis_aligned_dimensions_mm": (np.ptp(vertices, axis=0) * 1000).tolist(),
            "closed_consistently_oriented_edges": closed, "non_two_incident_edges": int(np.sum(counts != 2)),
            "signed_volume_m3": signed_volume, "enclosed_volume_cm3": abs(signed_volume) * 1e6 if closed else None}


def read_part(path):
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
        names = archive.namelist()
    unit = root.get("unit", "millimeter")
    scale = UNITS[unit]
    objects = {item.get("id"): item for item in root.findall("m:resources/m:object", NAMESPACE)}
    summaries = []
    for item in root.findall("m:build/m:item", NAMESPACE):
        obj = objects[item.get("objectid")]
        if obj.find("m:components", NAMESPACE) is not None:
            raise ValueError("Component assemblies require explicit recursive transforms")
        mesh = obj.find("m:mesh", NAMESPACE)
        vertices = np.array([[float(vertex.get(axis)) for axis in ("x", "y", "z")]
                             for vertex in mesh.findall("m:vertices/m:vertex", NAMESPACE)])
        transform = np.fromstring(item.get("transform", "1 0 0 0 1 0 0 0 1 0 0 0"), sep=" ")
        assert transform.shape == (12,)
        vertices = (vertices @ transform[:9].reshape(3, 3) + transform[9:]) * scale
        triangles = np.array([[int(triangle.get(axis)) for axis in ("v1", "v2", "v3")]
                              for triangle in mesh.findall("m:triangles/m:triangle", NAMESPACE)])
        summaries.append({"object_name": obj.get("name"), "build_transform": transform.tolist(), **mesh_summary(vertices, triangles)})
    metadata = [{"attributes": dict(item.attrib), "text": item.text} for item in root.iter() if item.tag.rsplit("}", 1)[-1] == "metadata"]
    return {"file": str(path.resolve()), "sha256": digest(path), "declared_unit": unit, "archive_entries": names,
            "metadata": metadata, "objects": summaries,
            "mass_material_inertia": "No explicit mass, material density, print infill or inertia values found in these inspected archives; geometry volume is not printed mass."}


def stl_bounds(path):
    contents = path.read_bytes()
    count = struct.unpack_from("<I", contents, 80)[0]
    assert len(contents) == 84 + count * 50
    facets = np.frombuffer(contents, dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), offset=84)
    vertices = facets["vertices"].reshape(-1, 3).astype(float)
    return {"file": str(path.resolve()), "sha256": digest(path), "triangles": count,
            "bounds_min_raw": vertices.min(axis=0).tolist(), "bounds_max_raw": vertices.max(axis=0).tolist(),
            "dimensions_raw": np.ptp(vertices, axis=0).tolist(), "units": "STL has no declared unit; use consuming MJCF mesh scale, not implicit print millimeters."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", type=Path)
    args = parser.parse_args()
    repository = args.parts.resolve().parent
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain=v1"])
    parts = [read_part(path) for path in sorted(args.parts.glob("*.3mf"))]
    assert len(parts) == 4
    xml_path = ROOT / "g1_description/g1.xml"
    root = ET.parse(xml_path).getroot()
    scene_bytes = subprocess.check_output(["git", "-C", str(repository), "show", f"{REFERENCE_COMMIT}:sim2sim_mj/scene.xml"])
    scene = ET.fromstring(scene_bytes)
    reference_arms = [{"body": body.get("name"), "inertial": dict(body.find("inertial").attrib),
                       "collision": [dict(geom.attrib) for geom in body.findall("geom") if "hand_collision" in geom.get("name", "")]}
                      for body in scene.findall(".//body") if body.get("name") in ("left_forearm", "right_forearm")]
    report = {"status": "completed", "parts": parts, "readme": str(repository / "README.md"), "readme_sha256": digest(repository / "README.md"),
              "source_xml": str(xml_path), "source_xml_sha256": digest(xml_path), "source_sha256": {Path(__file__).name: digest(__file__)},
              "current_visual_forearm": stl_bounds(ROOT / "g1_description/meshes/forearm.stl"),
              "current_forearm_mesh_definition": dict(root.find("./asset/mesh[@name='forearm']").attrib),
              "reference_commit": REFERENCE_COMMIT, "reference_arm_definitions": reference_arms,
              "reference_scene_sha256": __import__("hashlib").sha256(scene_bytes).hexdigest(),
              "limitations": "CAD geometry and repository model values are evidence, not as-built mass/inertia/material calibration. Build-frame bounds are not robot-frame assembly transforms. Squash ball and mounting hardware are separate; no collision or mass parameters changed."}
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain=v1"]) == before
    destination = ROOT / "output/rl/printed_parts_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for part in parts:
        print(json.dumps({"file": Path(part["file"]).name, "unit": part["declared_unit"], "objects": part["objects"]}), flush=True)
    print(json.dumps(report["current_visual_forearm"]))
    print(destination / "report.json")


if __name__ == "__main__":
    main()
