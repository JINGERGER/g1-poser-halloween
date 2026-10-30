import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from prepare import ROOT
from printed_parts_audit import NAMESPACE, REFERENCE_COMMIT
from target_load_ablation import digest


def point_surface_distances(points, facets):
    origin = facets[:, 0]
    edge_one, edge_two = facets[:, 1] - origin, facets[:, 2] - origin
    normal = np.cross(edge_one, edge_two)
    squared_normal = np.sum(normal * normal, axis=1)
    first_squared = np.sum(edge_one * edge_one, axis=1)
    second_squared = np.sum(edge_two * edge_two, axis=1)
    product = np.sum(edge_one * edge_two, axis=1)
    determinant = first_squared * second_squared - product * product
    result = []
    for start in range(0, len(points), 64):
        relative = points[start:start + 64, None, :] - origin
        first_dot = np.einsum("bti,ti->bt", relative, edge_one)
        second_dot = np.einsum("bti,ti->bt", relative, edge_two)
        first_barycentric = (second_squared * first_dot - product * second_dot) / np.maximum(determinant, 1e-30)
        second_barycentric = (first_squared * second_dot - product * first_dot) / np.maximum(determinant, 1e-30)
        inside = (determinant > 1e-25) & (first_barycentric >= 0) & (second_barycentric >= 0) & (first_barycentric + second_barycentric <= 1)
        plane_squared = np.einsum("bti,ti->bt", relative, normal) ** 2 / np.maximum(squared_normal, 1e-30)
        minimum = np.where(inside, plane_squared, np.inf)
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = facets[:, second] - facets[:, first]
            delta = points[start:start + 64, None, :] - facets[:, first]
            fraction = np.clip(np.einsum("bti,ti->bt", delta, edge) / np.maximum(np.sum(edge * edge, axis=1), 1e-30), 0, 1)
            remainder = delta - fraction[:, :, None] * edge
            minimum = np.minimum(minimum, np.sum(remainder * remainder, axis=2))
        result.extend(np.sqrt(np.min(minimum, axis=1)))
    return np.array(result)


def statistics(distances):
    return {"count": len(distances), "median_mm": float(np.median(distances) * 1000),
            "p95_mm": float(np.quantile(distances, .95) * 1000), "max_mm": float(max(distances) * 1000),
            "fraction_within_0.1mm": float(np.mean(distances <= .0001))}


def load_meshes(cad, stl):
    with zipfile.ZipFile(cad) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
    assert root.get("unit") == "meter"
    build = root.findall("m:build/m:item", NAMESPACE)
    assert len(build) == 1 and "transform" not in build[0].attrib
    obj = root.find(f"m:resources/m:object[@id='{build[0].get('objectid')}']", NAMESPACE)
    vertices = np.array([[float(vertex.get(axis)) for axis in ("x", "y", "z")] for vertex in obj.findall("m:mesh/m:vertices/m:vertex", NAMESPACE)])
    triangles = np.array([[int(triangle.get(axis)) for axis in ("v1", "v2", "v3")] for triangle in obj.findall("m:mesh/m:triangles/m:triangle", NAMESPACE)])
    contents = stl.read_bytes()
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    count = int.from_bytes(contents[80:84], "little")
    assert len(contents) == 84 + 50 * count
    facets = np.frombuffer(contents, dtype=dtype, offset=84)["vertices"].astype(float)
    return vertices, triangles, facets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts_audit", type=Path)
    args = parser.parse_args()
    source = json.loads(args.parts_audit.read_text())
    for name, expected in source["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    part = next(part for part in source["parts"] if Path(part["file"]).name == "forearm.3mf")
    cad, stl = Path(part["file"]), Path(source["current_visual_forearm"]["file"])
    assert digest(cad) == part["sha256"] and digest(stl) == source["current_visual_forearm"]["sha256"]
    repository = cad.parent.parent
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain=v1"])
    archived = subprocess.check_output(["git", "-C", str(repository), "show", f"{REFERENCE_COMMIT}:sim2sim_mj/meshes/forearm.stl"])
    vertices, triangles, facets = load_meshes(cad, stl)
    stl_vertices = np.unique(facets.reshape(-1, 3), axis=0)
    distal = stl_vertices[stl_vertices[:, 0] > stl_vertices[:, 0].max() - .012]
    fitted, residuals, rank, singular = np.linalg.lstsq(np.c_[2 * distal, np.ones(len(distal))], np.sum(distal * distal, axis=1), rcond=None)
    assert rank == 4
    center = fitted[:3]
    radius = float(np.sqrt(fitted[3] + center @ center))
    radial_error = np.abs(np.linalg.norm(distal - center, axis=1) - radius)
    assert max(radial_error) < 1e-7
    spherical = np.max(np.abs(np.linalg.norm(facets - center, axis=2) - radius), axis=1) < 1e-7
    assert np.any(spherical) and np.any(~spherical)
    shell = facets[~spherical]
    shell_points = np.unique(shell.reshape(-1, 3), axis=0)
    translation = np.array([shell_points[:, 0].min() - vertices[:, 2].min(), 0., 0.])
    base_rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    comparisons = []
    for turn in range(4):
        angle = turn * np.pi / 2
        axial = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        rotation = axial @ base_rotation
        transformed = vertices @ rotation.T + translation
        forward = point_surface_distances(transformed, shell)
        backward = point_surface_distances(shell_points, transformed[triangles])
        comparisons.append({"quarter_turns": turn, "rotation_column_vector": rotation.tolist(), "translation_m": translation.tolist(),
                            "cad_vertices_to_stl_shell": statistics(forward), "stl_shell_vertices_to_cad": statistics(backward),
                            "score_p95_sum_mm": float((np.quantile(forward, .95) + np.quantile(backward, .95)) * 1000)})
    best = min(comparisons, key=lambda candidate: candidate["score_p95_sum_mm"])
    transformed = vertices @ np.array(best["rotation_column_vector"]).T + translation
    forward_centroids = point_surface_distances(transformed[triangles].mean(axis=1), shell)
    backward_centroids = point_surface_distances(shell.mean(axis=1), transformed[triangles])
    best["cad_centroids_to_stl_shell"] = statistics(forward_centroids)
    best["stl_shell_centroids_to_cad"] = statistics(backward_centroids)
    original_xml = Path(source["source_xml"])
    assert digest(original_xml) == source["source_xml_sha256"]
    root = ET.parse(original_xml).getroot()
    collisions = [dict(root.find(f".//geom[@name='{side}_hand_collision']").attrib) for side in ("left", "right")]
    report = {"status": "completed", "training_updates": 0, "source_report": str(args.parts_audit.resolve()), "source_report_sha256": digest(args.parts_audit),
              "source_sha256": {**source["source_sha256"], Path(__file__).name: digest(__file__)},
              "stl_identical_to_reference_commit": stl.read_bytes() == archived, "reference_stl_sha256": hashlib.sha256(archived).hexdigest(),
              "fitted_sphere": {"center_stl_m": center.tolist(), "radius_m": radius, "max_radial_error_m": float(max(radial_error)),
                                "fit_points": len(distal), "classified_triangles": int(sum(spherical)), "fit_cutoff_m": float(stl_vertices[:, 0].max() - .012)},
              "shell_dimensions_mm": (np.ptp(shell_points, axis=0) * 1000).tolist(), "candidates": comparisons, "selected": best,
              "source_collision_geoms": collisions,
              "limitations": "Four proper axis-alignment hypotheses, no scale fit. Vertex/centroid to triangle distances are sampled surface comparisons, not exact Hausdorff distance, proof of identical solids or unique mounting orientation. Sphere fit identifies geometry in the visual STL, not measured ball compliance, mass or installed location. No physics parameters changed."}
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain=v1"]) == status
    destination = ROOT / "output/rl/forearm_alignment" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("stl_identical_to_reference_commit", "fitted_sphere", "shell_dimensions_mm", "selected")}, indent=2))
    print(destination / "report.json")


if __name__ == "__main__":
    main()
