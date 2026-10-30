import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from artifact_utils import digest
from audit_transition_collision import world_vertices
from forearm_alignment import load_meshes
from reference_policy_baseline import ROOT


def crossing(first, second):
    lower, upper = second.min(axis=1), second.max(axis=1)
    for first_index, triangle in enumerate(first):
        candidates = np.flatnonzero(np.all(upper >= triangle.min(axis=0), axis=1)
                                   & np.all(lower <= triangle.max(axis=0), axis=1))
        for second_index in candidates:
            other = second[second_index]
            for reverse, edges, surface in ((False, triangle, other), (True, other, triangle)):
                for edge in range(3):
                    start, end = edges[edge], edges[(edge + 1) % 3]
                    direction = end - start
                    matrix = np.column_stack((direction, -(surface[1] - surface[0]),
                                              -(surface[2] - surface[0])))
                    norms = np.linalg.norm(matrix, axis=0)
                    if np.any(norms < 1e-12):
                        continue
                    if abs(np.linalg.det(matrix / norms)) < 1e-9:
                        continue
                    fraction, bary_first, bary_second = np.linalg.solve(matrix, surface[0] - start)
                    if (1e-7 < fraction < 1. - 1e-7 and bary_first > 1e-7
                            and bary_second > 1e-7 and bary_first + bary_second < 1. - 1e-7):
                        point = start + fraction * direction
                        residual = point - (surface[0] + bary_first * (surface[1] - surface[0])
                                            + bary_second * (surface[2] - surface[0]))
                        if np.linalg.norm(residual) > 1e-10:
                            raise ValueError("Invalid crossing witness")
                        return {"first_triangle": int(first_index), "second_triangle": int(second_index),
                                "edge_on_second": reverse, "edge": edge, "point_m": point.tolist(),
                                "fraction": float(fraction), "barycentric": [float(bary_first), float(bary_second)],
                                "first_vertices_m": triangle.tolist(), "second_vertices_m": other.tolist()}
    return None


def main():
    source = ROOT / "output/rl/transition_collision_audit/2026-10-23_18-31-18-576249/report.json"
    previous = json.loads(source.read_text())
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    assets = {str(path.relative_to(model_path.parent)): digest(path)
              for path in sorted(model_path.parent.rglob("*")) if path.is_file()}
    assert assets == previous["assets"]
    alignment_path = ROOT / "output/rl/forearm_alignment/2026-10-19_20-03-10-217402/report.json"
    alignment = json.loads(alignment_path.read_text())
    assert digest(alignment["source_report"]) == alignment["source_report_sha256"]
    parts = json.loads(Path(alignment["source_report"]).read_text())
    part = next(part for part in parts["parts"] if Path(part["file"]).name == "forearm.3mf")
    visual = parts["current_visual_forearm"]
    assert digest(part["file"]) == part["sha256"] and digest(visual["file"]) == visual["sha256"]
    vertices, faces, unused = load_meshes(Path(part["file"]), Path(visual["file"]))
    transform = alignment["selected"]
    vertices = vertices @ np.array(transform["rotation_column_vector"]).T + transform["translation_m"]
    obj = model_path.parent / "meshes/printed_shell_hull.obj"
    hull_vertices = np.array([[float(value) for value in line.split()[1:]]
                              for line in obj.read_text().splitlines() if line.startswith("v ")])
    np.testing.assert_allclose(vertices, hull_vertices, rtol=0, atol=1e-12)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    cases = []
    for case in previous["cases"]:
        trajectory = ROOT / case["trajectory"]
        assert digest(trajectory) == case["sha256"]
        saved = np.load(trajectory)
        data = mujoco.MjData(model)
        data.qpos[:] = saved["qpos"][-1]
        mujoco.mj_forward(model, data)
        body = model.body("right_wrist_roll_rubber_hand").id
        shell = vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
        comparisons = []
        for comparison in case["comparisons"]:
            if comparison["arm"] != "right_forearm" or comparison["separated"]:
                continue
            geom = model.geom(comparison["leg_visual"]).id
            mesh = model.geom_dataid[geom]
            start = model.mesh_faceadr[mesh]
            leg_faces = model.mesh_face[start:start + model.mesh_facenum[mesh]]
            leg_vertices = world_vertices(model, data, geom)
            leg_path = model_path.parent / "meshes" / (comparison["body"] + ".STL")
            unused_vertices, unused_faces, raw_leg = load_meshes(Path(part["file"]), leg_path)
            leg_body = model.geom_bodyid[geom]
            raw_world = raw_leg.reshape(-1, 3) @ data.xmat[leg_body].reshape(3, 3).T + data.xpos[leg_body]
            assert cKDTree(leg_vertices).query(raw_world)[0].max() < 1e-7
            witness = crossing(shell[faces], leg_vertices[leg_faces])
            comparisons.append({"body": comparison["body"], "transverse_surface_crossing": witness is not None,
                                "witness": witness})
        cases.append({"trajectory": case["trajectory"], "sha256": case["sha256"], "comparisons": comparisons})
        print(case["trajectory"], [(entry["body"], entry["transverse_surface_crossing"]) for entry in comparisons], flush=True)
    output = ROOT / "output/rl/transition_surface_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"source_sha256": digest(source), "script_sha256": digest(Path(__file__)),
              "alignment_sha256": digest(alignment_path), "cad_sha256": part["sha256"], "assets": assets,
              "method": "Original aligned 3MF triangles versus compiled visual leg triangles; checks raw STL vertex mapping. "
              "Positive interior segment/triangle witness proves transverse surface intersection at saved pose. "
              "Negative result is inconclusive: coplanarity, tangency, containment and numerical degeneracies are not certified. "
              "Only previously unresolved right-side pairs; no collision model or physics parameters changed. "
              "CAD/visual geometry evidence, not as-built hardware calibration or a penetration depth estimate.", "cases": cases}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
