import hashlib
import json

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from artifact_utils import digest
from check_reverse_landing_path import ENVELOPE
from check_transition_route_envelope import static_capacity
from reference_policy_baseline import ROOT
from run_arm_preparation import REFERENCE
from stand_transition_task import StandTransitionTask
from standing_capture_routes import integration_state


def main():
    for pitch in (1.4, np.pi / 2. - 1e-4, np.pi / 2. + 1e-4, 1.8):
        original = Rotation.from_euler("XYZ", [.05, pitch, -.03])
        coordinates = original.as_rotvec()
        restored = Rotation.from_rotvec(coordinates)
        assert (restored.inv() * original).magnitude() < 1e-12
        assert abs(coordinates[0]) < .7 and -.6 < coordinates[1] < 1.9 and abs(coordinates[2]) < .7
    task = StandTransitionTask()
    task.make_states([0])
    before = integration_state(task)
    report = json.loads(ENVELOPE.read_text())
    assert report["status"] == "completed" and not report["full_transition_validated"]
    assert hashlib.sha256(report["source_snapshot"].encode()).hexdigest() == report["source_sha256"]
    assert digest(ROOT / report["source_report"]) == report["source_report_sha256"]
    assert len(report["cases"]) == 12
    data = mujoco.MjData(task.model)
    classifications = []
    for case in report["cases"]:
        data.qpos[:] = case["qpos"]
        mujoco.mj_forward(task.model, data)
        assert static_capacity(task, data, False)["feasible"] == case["feet_only"]["feasible"]
        if not case["geometry_passed"]:
            continue
        np.testing.assert_allclose(data.geom_xpos[task.supports[2:]], case["foot_goals_xyz"], atol=.001, rtol=0.)
        assert np.max(np.linalg.norm(data.geom_xpos[task.supports[:2]] - report["hand_goals_xyz"], axis=1)) < .005
        np.testing.assert_allclose(data.geom_xmat[task.supports[2:]].reshape(2, 3, 3), case["foot_rotation_goals"], atol=.005, rtol=0.)
        corners = []
        for geom in task.supports[2:]:
            size = task.model.geom_size[geom].copy()
            size[:2] -= .005
            rotation = data.geom_xmat[geom].reshape(3, 3)
            for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                corners.append((data.geom_xpos[geom] + rotation @ (size * [sign_x, sign_y, -1.]))[:2])
        hull = ConvexHull(corners)
        com = data.subtree_com[task.model.body("pelvis").id, :2]
        outside = float(np.max(hull.equations[:, :2] @ com + hull.equations[:, 2]))
        if case["feet_only"]["feasible"]:
            assert outside <= 1e-6
        classifications.append({"layout": case["layout"], "balance": case["balance"],
                                "feet_only_static": case["feet_only"]["feasible"], "com_support_polygon_violation_m": outside})
    planned = json.loads(REFERENCE.read_text())
    for name, expected in planned["sources"].items():
        assert hashlib.sha256(planned["source_snapshots"][name].encode()).hexdigest() == expected
    route = next(route for route in planned["routes"] if route["family"] == "front_then_down")
    assert route["nodes_passed"]
    poses = np.array([node["qpos"] for node in route["nodes"]])
    assert np.max(np.abs(np.diff(poses[:, task.addresses], axis=0))) <= .200001
    for lower, upper in zip(poses[:-1], poses[1:], strict=True):
        for fraction in np.linspace(0., 1., 11):
            data.qpos[:] = (1. - fraction) * lower + fraction * upper
            data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
            mujoco.mj_forward(task.model, data)
            for contact in data.contact:
                if task.floor not in (contact.geom1, contact.geom2):
                    assert contact.dist >= -.001
                else:
                    geom = contact.geom2 if contact.geom1 == task.floor else contact.geom1
                    assert geom in task.supports or contact.dist >= 0.
    np.testing.assert_array_equal(integration_state(task), before)
    print(json.dumps({"passed": True, "checks": ["90-degree rotation branch", "archived source hashes",
                      "static support classification", "COM support polygon", "bounded joint increments",
                      "dense interpolated collision screening", "real state immutability"],
                      "classifications": classifications, "full_transition_validated": False}))


if __name__ == "__main__":
    main()
