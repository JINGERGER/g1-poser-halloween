import json
from datetime import datetime

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from artifact_utils import digest
from check_standing_endpoint import standing_pose
from reference_policy_baseline import ROOT


def support_polygon(model, data, feet):
    points = []
    for geom in feet:
        rotation = data.geom_xmat[geom].reshape(3, 3)
        for forward in (-1., 1.):
            for lateral in (-1., 1.):
                corner = model.geom_size[geom] * [forward, lateral, -1.]
                points.append((data.geom_xpos[geom] + rotation @ corner)[:2])
    return ConvexHull(np.array(points)).equations


def geometry(model, data, feet, floor, foot_positions, foot_rotations, polygon):
    penetration, self_penetration, other_penetration = 0., 0., 0.
    deepest = None
    supports = feet + [model.geom(name).id for name in ("left_hand_collision", "right_hand_collision")]
    for contact in data.contact:
        if -contact.dist > penetration:
            penetration = float(-contact.dist)
            deepest = [model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name]
        if floor not in (contact.geom1, contact.geom2):
            self_penetration = max(self_penetration, float(-contact.dist))
        elif (contact.geom2 if contact.geom1 == floor else contact.geom1) not in supports:
            other_penetration = max(other_penetration, float(-contact.dist))
    joints = model.actuator_trnid[:, 0]
    angles = data.qpos[model.jnt_qposadr[joints]]
    excess = max(0., float(np.max(model.jnt_range[joints, 0] - angles)),
                 float(np.max(angles - model.jnt_range[joints, 1])))
    com = data.subtree_com[model.body("pelvis").id, :2]
    margin = float(np.min(-(polygon[:, :2] @ com + polygon[:, 2])))
    errors = [np.arccos(np.clip((np.trace(original.T @ data.geom_xmat[geom].reshape(3, 3)) - 1) / 2, -1., 1.))
              for geom, original in zip(feet, foot_rotations)]
    metrics = {"penetration_m": penetration, "self_penetration_m": self_penetration,
               "non_support_penetration_m": other_penetration, "deepest_pair": deepest,
               "foot_position_error_m": float(np.max(np.linalg.norm(data.geom_xpos[feet] - foot_positions, axis=1))),
               "foot_orientation_error_rad": float(max(errors)), "joint_excess_rad": excess,
               "com_support_margin_m": margin}
    gates = {"penetration_5mm": penetration <= .005, "self_penetration_1mm": self_penetration <= .001,
             "non_support_penetration_01mm": other_penetration <= .0001,
             "foot_position_2mm": metrics["foot_position_error_m"] <= .002,
             "foot_orientation_001rad": metrics["foot_orientation_error_rad"] <= .01,
             "joint_limits": excess <= .001, "com_margin_5mm": margin >= .005}
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def main():
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    initial_data = standing_pose(model, .6)
    initial = initial_data.qpos.copy()
    feet = [model.geom(name).id for name in ("left_foot", "right_foot")]
    hands = [model.geom(name).id for name in ("left_hand_collision", "right_hand_collision")]
    floor = model.geom("floor").id
    foot_positions = initial_data.geom_xpos[feet].copy()
    foot_rotations = [initial_data.geom_xmat[geom].reshape(3, 3).copy() for geom in feet]
    hand_positions = initial_data.geom_xpos[hands].copy()
    polygon = support_polygon(model, initial_data, feet)
    center = foot_positions.mean(axis=0)
    joints = model.actuator_trnid[:, 0]
    assert np.array_equal(model.jnt_qposadr[joints], np.arange(7, model.nq))
    assert np.array_equal(model.jnt_dofadr[joints], np.arange(6, model.nv))
    lower = np.r_[[-.4, -.2, -.4, -.5, -.2, -.5], model.jnt_range[joints, 0] - initial[7:]]
    upper = np.r_[[.4, .2, .1, .5, 1.4, .5], model.jnt_range[joints, 1] - initial[7:]]
    output = ROOT / "output/rl/stand_contact_routes" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(__file__),
              "standing_source_sha256": digest(ROOT / "rl/check_standing_endpoint.py"),
              "assets": {str(path.relative_to(model_path.parent)): digest(path)
                         for path in sorted(model_path.parent.rglob("*")) if path.is_file()},
              "protocol": "Geometry-only first-hand-contact screening, not a complete transition or physics execution. "
                          "Two routes crossed with hand targets .25/.35/.45m forward of foot center. "
                          "Four continuation nodes plus initial, max 120 least-squares evaluations each. "
                          "Feet fixed, COM targeted over two-foot polygon even at first contact. "
                          "Collision checked after solving, not optimized away. No parameter retries. "
                          "Twenty interpolation samples per segment; samples do not certify continuous collision freedom.",
              "limitations": ["No force balance, torque or dynamics proof", "No PD switching tested",
                              "No connection from first contact to crawl endpoint", "No right-hand-first route",
                              "COM-in-polygon is a quasi-static screen, not a general dynamic necessity",
                              "Existing adjacent-body collision filtering remains"], "routes": []}
    data = mujoco.MjData(model)

    def set_pose(values):
        data.qpos[:] = initial
        mujoco.mj_integratePos(model, data.qpos, values, 1.)
        mujoco.mj_forward(model, data)

    try:
        for route in ("both_hands", "left_hand_first"):
            for forward in (.25, .35, .45):
                values = np.zeros(model.nv)
                nodes = [{"phase": 0., "qpos": initial.tolist()}]
                target = hand_positions.copy()
                target[0] = [center[0] + forward, foot_positions[0, 1], model.geom_size[hands[0], 0]]
                if route == "both_hands":
                    target[1] = [center[0] + forward, foot_positions[1, 1], model.geom_size[hands[1], 0]]
                else:
                    target[1] = [center[0] + forward, hand_positions[1, 1], .15]
                for phase in (.25, .5, .75, 1.):
                    desired = hand_positions + phase * (target - hand_positions)

                    def residual(candidate):
                        set_pose(candidate)
                        rotations = np.concatenate([(data.geom_xmat[geom].reshape(3, 3) - original).ravel()
                                                    for geom, original in zip(feet, foot_rotations)])
                        com = data.subtree_com[model.body("pelvis").id, :2]
                        return np.r_[(data.geom_xpos[feet] - foot_positions).ravel() / .001,
                                     rotations / .01, (data.geom_xpos[hands] - desired).ravel() / .005,
                                     (com - center[:2]) / .01, candidate * .05]

                    solved = least_squares(residual, values, bounds=(lower, upper), max_nfev=120)
                    values = solved.x
                    set_pose(values)
                    checked = geometry(model, data, feet, floor, foot_positions, foot_rotations, polygon)
                    error = float(np.max(np.linalg.norm(data.geom_xpos[hands] - desired, axis=1)))
                    nodes.append({"phase": phase, "qpos": data.qpos.tolist(), "nfev": solved.nfev,
                                  "solver_success": bool(solved.success), "hand_target_error_m": error, **checked})
                samples = []
                for start, finish in zip(nodes, nodes[1:]):
                    delta = np.zeros(model.nv)
                    mujoco.mj_differentiatePos(model, delta, 1., np.array(start["qpos"]), np.array(finish["qpos"]))
                    for fraction in np.linspace(0., 1., 21):
                        data.qpos[:] = start["qpos"]
                        mujoco.mj_integratePos(model, data.qpos, delta, float(fraction))
                        mujoco.mj_forward(model, data)
                        samples.append({"phase": start["phase"] + fraction * .25,
                                        **geometry(model, data, feet, floor, foot_positions, foot_rotations, polygon)})
                passed = all(node["passed"] and node["hand_target_error_m"] <= .005 for node in nodes[1:]) and all(point["passed"] for point in samples)
                result = {"route": route, "forward_target_m": forward, "geometry_passed": passed,
                          "nodes": nodes, "samples": samples}
                report["routes"].append(result)
                print(json.dumps({"route": route, "forward": forward, "passed": passed,
                                  "final_hand_error": nodes[-1]["hand_target_error_m"],
                                  "final_gates": nodes[-1]["gates"],
                                  "worst_penetration": max(point["metrics"]["penetration_m"] for point in samples)}), flush=True)
        report["status"] = "completed"
    finally:
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
