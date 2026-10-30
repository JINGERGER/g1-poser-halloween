import argparse
import json
from pathlib import Path

import numpy as np

from forearm_alignment import load_meshes, point_surface_distances, statistics
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    facets = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
    points = np.array([[.2, .2, 1], [.2, .2, 0], [1, 1, 0], [-1, 0, 0], [.5, 0, -.2]])
    expected = np.array([1, 0, np.sqrt(.5), 1, .2])
    np.testing.assert_allclose(point_surface_distances(points, facets), expected, atol=1e-12)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(point_surface_distances(points @ rotation.T + 3, facets @ rotation.T + 3), expected, atol=1e-12)
    degenerate = np.array([[[0., 0, 0], [1, 0, 0], [1, 0, 0]]])
    np.testing.assert_allclose(point_surface_distances(np.array([[.5, 1, 0]]), degenerate), [1], atol=1e-12)
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert digest(report["source_report"]) == report["source_report_sha256"]
    for name, expected_hash in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected_hash
    source = json.loads(Path(report["source_report"]).read_text())
    part = next(part for part in source["parts"] if Path(part["file"]).name == "forearm.3mf")
    cad, stl = Path(part["file"]), Path(source["current_visual_forearm"]["file"])
    assert digest(cad) == part["sha256"] and digest(stl) == source["current_visual_forearm"]["sha256"]
    assert digest(stl) == report["reference_stl_sha256"] and report["stl_identical_to_reference_commit"]
    vertices, triangles, facets = load_meshes(cad, stl)
    sphere = report["fitted_sphere"]
    center, radius = np.array(sphere["center_stl_m"]), sphere["radius_m"]
    spherical = np.max(np.abs(np.linalg.norm(facets - center, axis=2) - radius), axis=1) < 1e-7
    assert sum(spherical) == sphere["classified_triangles"]
    assert abs(radius - .02) < 1e-6
    shell = facets[~spherical]
    np.testing.assert_allclose(np.ptp(shell.reshape(-1, 3), axis=0) * 1000, report["shell_dimensions_mm"], atol=1e-12)
    assert len(report["candidates"]) == 4
    selected = report["selected"]
    assert selected["score_p95_sum_mm"] == min(candidate["score_p95_sum_mm"] for candidate in report["candidates"])
    rotation = np.array(selected["rotation_column_vector"])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-12)
    transformed = vertices @ rotation.T + selected["translation_m"]
    for points, target, key in ((transformed, shell, "cad_vertices_to_stl_shell"),
                                (np.unique(shell.reshape(-1, 3), axis=0), transformed[triangles], "stl_shell_vertices_to_cad"),
                                (transformed[triangles].mean(axis=1), shell, "cad_centroids_to_stl_shell"),
                                (shell.mean(axis=1), transformed[triangles], "stl_shell_centroids_to_cad")):
        actual = statistics(point_surface_distances(points, target))
        for name, value in actual.items():
            np.testing.assert_allclose(value, selected[key][name], atol=1e-10)
    print("PASS: triangle distances, rigid-transform invariance, degenerate triangles, input hashes, sphere split, proper alignment and bidirectional sampled surface checks")


if __name__ == "__main__":
    main()
