import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import LinearConstraint, minimize

from load_transfer import physical_metrics
from prepare import ROOT, SITES, SUPPORTS
from target_load_ablation import digest


STANCE = (0, 1, 3)


def polygon_margin(points, position):
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    distances = -(hull.equations[:, :2] @ position + hull.equations[:, 2])
    return float(np.min(distances))


def snapshot(model, data):
    floor = model.geom("floor").id
    ids = [model.geom(name).id for name in SUPPORTS]
    patches = [[] for name in SUPPORTS]
    for contact in data.contact:
        if floor not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor else contact.geom1
        if other in ids and contact.dist <= 0:
            patches[ids.index(other)].append(contact.pos[:2].tolist())
    sites = np.array([data.site(name).xpos.copy() for name in SITES])
    com = data.subtree_com[1].copy()
    return {"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(), "ctrl": data.ctrl.tolist(),
            "sites_xyz_m": sites.tolist(), "com_xyz_m": com.tolist(), "contact_patches_xy_m": patches,
            "four_support_margin_m": polygon_margin([point for patch in patches for point in patch], com[:2]),
            "remaining_support_margin_m": polygon_margin([point for limb in STANCE for point in patches[limb]], com[:2]),
            "metrics": physical_metrics(model, data)}


def nearest_inset_target(points, position, inset=.005):
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    position = np.asarray(position, dtype=float)
    assert position.shape == (2,) and inset >= 0
    if polygon_margin(points, position) is None:
        return {"passed": False, "reason": "degenerate_support_patch"}
    hull = ConvexHull(points)
    normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
    result = minimize(lambda target: .5 * np.sum((target - position) ** 2), position,
                      jac=lambda target: target - position, method="SLSQP",
                      constraints=[LinearConstraint(normals, -np.inf, -offsets - inset)],
                      options={"maxiter": 100, "ftol": 1e-12})
    margin = polygon_margin(points, result.x)
    passed = bool(result.success and margin >= inset - 1e-8)
    return {"passed": passed, "message": str(result.message), "iterations": int(result.nit),
            "inset_m": inset, "target_xy_m": result.x.tolist(), "shift_xy_m": (result.x - position).tolist(),
            "shift_norm_m": float(np.linalg.norm(result.x - position)), "margin_m": margin}


def compare(model, start, end):
    start_sites, end_sites = np.array(start["sites_xyz_m"]), np.array(end["sites_xyz_m"])
    delta = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, delta, 1, np.array(start["qpos"]), np.array(end["qpos"]))
    source, target = end_sites[list(STANCE), :2], start_sites[list(STANCE), :2]
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, singular, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    aligned = (source - source_center) @ rotation + target_center
    ranked = sorted(range(6, model.nv), key=lambda index: abs(delta[index]), reverse=True)
    return {"com_delta_xyz_m": (np.array(end["com_xyz_m"]) - start["com_xyz_m"]).tolist(),
            "base_delta_xyz_m": (np.array(end["qpos"][:3]) - start["qpos"][:3]).tolist(),
            "site_delta_xyz_m": (end_sites - start_sites).tolist(),
            "stance_horizontal_displacement_m": np.linalg.norm((end_sites - start_sites)[list(STANCE), :2], axis=1).tolist(),
            "stance_residual_after_planar_rigid_alignment_m": np.linalg.norm(aligned - target, axis=1).tolist(),
            "stance_pair_distance_changes_m": [float(np.linalg.norm(end_sites[first, :2] - end_sites[second, :2]) - np.linalg.norm(start_sites[first, :2] - start_sites[second, :2]))
                                                for first, second in ((0, 1), (0, 3), (1, 3))],
            "generalized_displacement": delta.tolist(),
            "largest_joint_changes": [{"joint": model.joint(int(model.dof_jntid[index])).name, "delta_rad": float(delta[index])} for index in ranked[:6]]}


def main():
    parser = argparse.ArgumentParser(description="Endpoint and support-geometry audit; no live-state editing or optimization")
    parser.add_argument("ablation_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.ablation_report.read_text())
    assert source["status"] == "completed" and mujoco.__version__ == source["mujoco"]
    assert digest(source["model"]) == source["model_sha256"]
    for name, expected in source["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["matrix_report"]) == source["matrix_report_sha256"]
    matrix = json.loads(Path(source["matrix_report"]).read_text())
    terminal = next(run for run in matrix["runs"] if run["anchor"] == 2 and run["swing"] == 2)
    hold = next(run for run in terminal["replays"] if run["branch"] == "hold")
    assert hold["passed"] and terminal["model_sha256"] == source["model_sha256"]
    model = mujoco.MjModel.from_xml_path(source["model"])
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, np.array(source["initial_physics_state"]), mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = source["initial_ctrl"]
    mujoco.mj_forward(model, data)
    initial = snapshot(model, data)
    inset_target = nearest_inset_target([point for limb in STANCE for point in initial["contact_patches_xy_m"][limb]], initial["com_xyz_m"][:2])
    with np.load(hold["trajectory"], allow_pickle=False) as saved:
        data.qpos[:] = saved["qpos"][-1]
        data.qvel[:] = saved["qvel"][-1]
        data.ctrl[:] = saved["ctrl"][-1]
    mujoco.mj_forward(model, data)
    final = snapshot(model, data)
    np.testing.assert_allclose(final["metrics"]["loads_n"], hold["trace"][-1]["loads_n"], atol=1e-7, rtol=0)
    branches = []
    for run in source["runs"]:
        with np.load(run["trajectory"], allow_pickle=False) as saved:
            data.qpos[:] = saved["qpos"][-1]
            data.qvel[:] = saved["qvel"][-1]
            data.ctrl[:] = saved["ctrl"][-1]
        mujoco.mj_forward(model, data)
        current = snapshot(model, data)
        np.testing.assert_allclose(current["metrics"]["loads_n"], run["final_loads_n"], atol=1e-7, rtol=0)
        branches.append({"branch": run["branch"], "snapshot": current, "change": compare(model, initial, current)})
    trajectories = [hold["trajectory"]] + [run["trajectory"] for run in source["runs"]]
    report = {"status": "completed", "training_updates": 0, "dynamic_optimization_runs": 0,
              "geometric_target": inset_target,
              "source_report": str(args.ablation_report.resolve()), "source_report_sha256": digest(args.ablation_report),
              "source_sha256": {**source["source_sha256"], Path(__file__).name: digest(__file__)},
              "trajectory_sha256": {name: digest(name) for name in trajectories},
              "initial": initial, "terminal": final, "terminal_change": compare(model, initial, final), "branches": branches,
              "limitations": "Read-only scratch reconstruction, not dynamic replay. Margins use instantaneous floor contact patches and vertical COM projection: quasi-static geometry only, not friction/moment or dynamic feasibility. Site displacement does not distinguish rolling, sliding or lift. Static terminal may require different stance geometry; no reachability claim."}
    destination = ROOT / "output/rl/transition_geometry" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"geometric_target": inset_target, "initial_margin_m": initial["remaining_support_margin_m"], "terminal_margin_m": final["remaining_support_margin_m"],
                      "terminal_change": report["terminal_change"], "branches": [{"branch": run["branch"], "margin_m": run["snapshot"]["remaining_support_margin_m"], "com_delta_m": run["change"]["com_delta_xyz_m"]} for run in branches]}, indent=2))
    print(destination / "report.json")


if __name__ == "__main__":
    main()
