import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from arm_hull_candidate import build, evaluate, hull_obj
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.1, .1, .1]])
    lines = hull_obj(points).decode().splitlines()
    faces = [np.array([int(index) - 1 for index in line.split()[1:]]) for line in lines if line.startswith("f ")]
    assert len(faces) == 4
    for face in faces:
        triangle = points[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        assert np.max((points - triangle[0]) @ normal) < 1e-12
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and report["dynamic_rollouts"] == 0
    assert digest(report["source_report"]) == report["source_report_sha256"]
    folder = Path(report["model"]).parent
    assert digest(folder / "flat.xml") == report["model_sha256"]
    assert digest(folder / "manifest.json") == report["manifest_sha256"]
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(folder / "meshes" / name) == expected
    rebuilt, local, balls = build(Path(report["source_report"]))
    assert rebuilt == folder
    actual = evaluate(folder, local, balls)
    for key, value in actual.items():
        assert value == report[key]
    assert len(actual["local"]) == 42 and len(actual["full_limit_sweep"]) == 115
    model = mujoco.MjModel.from_xml_path(str(folder / "flat.xml"))
    base = mujoco.MjModel.from_xml_path(manifest["base_model"])
    for name in ("timestep", "gravity", "iterations", "solver", "integrator", "cone", "tolerance"):
        np.testing.assert_array_equal(getattr(model.opt, name), getattr(base.opt, name))
    for side in ("left", "right"):
        for suffix in ("upper_arm", "forearm", "elbow_collision"):
            geom = model.geom(f"{side}_{suffix}")
            assert geom.type == mujoco.mjtGeom.mjGEOM_MESH
            assert geom.contype == geom.conaffinity == 1
        assert model.geom(f"{side}_elbow_collision").bodyid == model.body(f"{side}_elbow_link").id
        for field in ("size", "pos", "friction", "solref", "solimp"):
            np.testing.assert_array_equal(getattr(model.geom(f"{side}_hand_collision"), field), getattr(base.geom(f"{side}_hand_collision"), field))
    raw = np.array([[float(value) for value in line.split()[1:]] for line in
                    (folder / "meshes/printed_shell_hull.obj").read_text().splitlines() if line.startswith("v ")])
    mesh = model.mesh("printed_shell_hull")
    compiled = model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]].astype(float)
    geom = model.geom("left_forearm")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, geom.quat)
    restored = compiled @ rotation.reshape(3, 3).T + geom.pos
    hull = ConvexHull(restored)
    assert np.max(raw @ hull.equations[:, :3].T + hull.equations[:, 3]) < 1e-7
    assert digest(manifest["base_model"]) == manifest["base_model_sha256"]
    print("PASS: hull orientation/coverage, immutable regeneration, body mapping, preserved physics and all 157 geometric probes")


if __name__ == "__main__":
    main()
