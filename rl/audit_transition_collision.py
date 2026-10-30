import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from artifact_utils import digest
from reference_policy_baseline import ROOT


def separating_plane(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    origin = first.mean(axis=0)
    centered_first, centered_second = first - origin, second - origin
    reduced_first = centered_first[ConvexHull(centered_first).vertices]
    reduced_second = centered_second[ConvexHull(centered_second).vertices]
    constraints = np.vstack([
        np.c_[reduced_first, -np.ones(len(reduced_first)), np.ones(len(reduced_first))],
        np.c_[-reduced_second, np.ones(len(reduced_second)), np.ones(len(reduced_second))],
    ])
    result = linprog([0., 0., 0., 0., -1.], A_ub=constraints,
                     b_ub=np.zeros(len(constraints)),
                     bounds=[(-1., 1.)] * 3 + [(None, None), (0., None)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    normal = result.x[:3]
    length = np.linalg.norm(normal)
    if length < 1e-10:
        return {"separated": False, "gap_lower_bound_m": 0., "normal": None}
    normal /= length
    gap = float(np.min(centered_second @ normal) - np.max(centered_first @ normal))
    return {"separated": gap > 1e-6, "gap_lower_bound_m": max(0., gap), "normal": normal.tolist()}


def world_vertices(model, data, geom):
    mesh = model.geom_dataid[geom]
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError("Expected mesh geometry")
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(float)
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def main():
    source = ROOT / "output/rl/closed_loop_transition/2026-10-23_16-27-00-393302/report.json"
    original = json.loads(source.read_text())
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    assets = {str(path.relative_to(model_path.parent)): digest(path)
              for path in sorted(model_path.parent.rglob("*")) if path.is_file()}
    if assets != original["assets"]:
        raise ValueError("Frozen model assets differ from experiment")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.geom_solref[:, 0] = .005
    cases = []
    for case in original["trained"]:
        trajectory = Path(case["trajectory"])
        saved = np.load(trajectory)
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = saved["qpos"][-1], saved["qvel"][-1]
        mujoco.mj_forward(model, data)
        contacts = [{"pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name],
                     "distance_m": float(contact.dist)} for contact in data.contact if contact.dist < -.001]
        comparisons = []
        for side in ("left", "right"):
            arm = model.geom(f"{side}_forearm").id
            arm_vertices = world_vertices(model, data, arm)
            for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
                body = model.body(f"{side}_{part}_link").id
                geoms = [geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == body
                         and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
                         and model.geom_group[geom] == 2]
                if not geoms:
                    raise ValueError("Missing visual mesh")
                for geom in geoms:
                    comparisons.append({"arm": model.geom(arm).name, "leg_visual": model.geom(geom).name,
                                        "body": model.body(body).name,
                                        **separating_plane(arm_vertices, world_vertices(model, data, geom))})
        cases.append({"trajectory": str(trajectory.relative_to(ROOT)), "sha256": digest(trajectory),
                      "time_s": float(saved["samples"][-1, 0]), "contacts_beyond_1mm": contacts,
                      "comparisons": comparisons})
    output = ROOT / "output/rl/transition_collision_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"source_sha256": digest(source), "script_sha256": digest(Path(__file__)), "assets": assets,
              "method": "Separating planes on compiled mesh vertices transformed by geom world pose. "
              "Positive gap proves separation of complete convex hulls, hence included mesh triangles. "
              "No separating plane is inconclusive for concave solids. Only same-side forearm versus four leg components, "
              "at eight final poses; not whole-robot clearance, hardware validation, or dynamic success. "
              "Geometry-only mj_forward reconstruction; contact forces are not evaluated.", "cases": cases}
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(path)
    for case in cases:
        print(case["time_s"], [(entry["body"], round(entry["gap_lower_bound_m"] * 1000, 3))
                               for entry in case["comparisons"]])


if __name__ == "__main__":
    main()
