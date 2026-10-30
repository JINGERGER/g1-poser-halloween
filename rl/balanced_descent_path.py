import json
from datetime import datetime

import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


class BalancedDescentPath:
    suffixes = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
                "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")

    def __init__(self, task):
        self.task = task
        self.model = task.model
        self.data = mujoco.MjData(self.model)
        self.initial = task.data.qpos.copy()
        self.data.qpos[:] = self.initial
        mujoco.mj_forward(self.model, self.data)
        self.feet = self.data.geom_xpos[task.supports[2:]].copy()
        self.rotations = self.data.geom_xmat[task.supports[2:]].copy()
        self.hands = self.data.geom_xpos[task.supports[:2]].copy()
        self.pelvis = self.model.body("pelvis").id
        self.initial_com = self.data.subtree_com[self.pelvis].copy()
        self.initial_pitch = float(2. * np.arctan2(self.initial[5], self.initial[3]))
        self.left = [self.model.joint(f"left_{suffix}_joint").id for suffix in self.suffixes]
        self.right = [self.model.joint(f"right_{suffix}_joint").id for suffix in self.suffixes]
        self.signs = np.array([-1. if "roll" in suffix or "yaw" in suffix else 1. for suffix in self.suffixes])
        self.lower = np.r_[-.5, .18, -.1, self.model.jnt_range[self.left, 0] + .03]
        self.upper = np.r_[.6, .85, 2., self.model.jnt_range[self.left, 1] - .03]
        self.values = np.r_[self.initial[0], self.initial[2], self.initial_pitch, self.initial[self.model.jnt_qposadr[self.left]]]

    def pose(self, values):
        self.data.qpos[:] = self.initial
        self.data.qpos[:3] = [values[0], 0., values[1]]
        self.data.qpos[3:7] = [np.cos(values[2] / 2.), 0., np.sin(values[2] / 2.), 0.]
        self.data.qpos[self.model.jnt_qposadr[self.left]] = values[3:]
        self.data.qpos[self.model.jnt_qposadr[self.right]] = self.signs * values[3:]
        mujoco.mj_forward(self.model, self.data)

    def depths(self):
        self_depth, other_depth = 0., 0.
        for contact in self.data.contact:
            if self.task.floor not in (contact.geom1, contact.geom2):
                self_depth = max(self_depth, -contact.dist)
            elif (contact.geom2 if contact.geom1 == self.task.floor else contact.geom1) not in self.task.supports:
                other_depth = max(other_depth, -contact.dist)
        return self_depth, other_depth

    def equilibrium(self):
        model, data, task = self.model, self.data, self.task
        force_jacs = []
        for geom in task.supports[2:]:
            rotation = data.geom_xmat[geom].reshape(3, 3)
            for sign_x, sign_y in ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
                point = data.geom_xpos[geom] + rotation @ (model.geom_size[geom] * [sign_x, sign_y, -1.])
                jac = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jac, None, point, model.geom_bodyid[geom])
                force_jacs.append(jac.T)
        matrix = np.zeros((model.nv, model.nu + 24))
        matrix[model.jnt_dofadr[task.joints], np.arange(model.nu)] = 1.
        matrix[:, model.nu:] = np.hstack(force_jacs)
        inequalities = np.zeros((32, model.nu + 24))
        for index in range(8):
            inequalities[index * 4:index * 4 + 4, model.nu + index * 3:model.nu + index * 3 + 3] = [
                [1., 0., -.4], [-1., 0., -.4], [0., 1., -.4], [0., -1., -.4]]
        limits = .7 * np.minimum(-model.jnt_actfrcrange[task.joints, 0], model.jnt_actfrcrange[task.joints, 1])
        bounds = list(zip(-limits, limits)) + [(None, None), (None, None), (0., 400.)] * 8
        result = linprog(np.zeros(model.nu + 24), A_ub=inequalities, b_ub=np.zeros(32),
                         A_eq=matrix, b_eq=data.qfrc_bias - data.qfrc_passive, bounds=bounds, method="highs")
        return {"feasible": bool(result.success), "torque_nm": result.x[:model.nu].tolist() if result.success else None}

    def solve(self, hand_x, final_pitch, spread=0., yaw=0., endpoint=False):
        nodes = []
        for phase in ([1.] if endpoint else np.linspace(0., 1., 31)):
            previous = self.values.copy()
            goals = self.hands.copy()
            goals[:, 0] += phase * (hand_x - goals[:, 0])
            goals[:, 1] += phase * (np.array([.32, -.32]) - goals[:, 1])
            goals[:, 2] += phase * (.02 - goals[:, 2])
            feet = self.feet.copy()
            feet[:, 1] += phase * np.array([spread, -spread])
            rotations = self.rotations.reshape(2, 3, 3).copy()
            for index, sign in enumerate((1., -1.)):
                angle = phase * yaw * sign
                rotations[index] = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]) @ rotations[index]

            def residual(values):
                self.pose(values)
                self_depth, other_depth = self.depths()
                return np.r_[500. * (self.data.geom_xpos[self.task.supports[2:]] - feet).ravel(),
                             50. * (self.data.geom_xmat[self.task.supports[2:]].reshape(2, 3, 3) - rotations).ravel(),
                             100. * (self.data.subtree_com[self.pelvis, :2] -
                                     (self.initial_com[:2] + phase * (np.array([self.feet[:, 0].mean(), 0.]) - self.initial_com[:2]))),
                             30. * (self.data.geom_xpos[self.task.supports[:2]] - goals).ravel(),
                             (.01 if endpoint else .5) * (values - previous),
                             (.1 if endpoint else 2.) * (values[2] - self.initial_pitch - phase * (final_pitch - self.initial_pitch)),
                             2. * values[8], 1000. * max(0., self_depth - .0005), 1000. * other_depth]

            solution = least_squares(residual, self.values, bounds=(self.lower, self.upper), max_nfev=150,
                                     ftol=1e-7, xtol=1e-7, gtol=1e-7)
            self.values = solution.x
            self.pose(self.values)
            self_depth, other_depth = self.depths()
            feet_error = float(np.max(np.linalg.norm(self.data.geom_xpos[self.task.supports[2:]] - feet, axis=1)))
            hand_error = float(np.max(np.linalg.norm(self.data.geom_xpos[self.task.supports[:2]] - goals, axis=1)))
            orientation_error = float(np.max(np.abs(self.data.geom_xmat[self.task.supports[2:]].reshape(2, 3, 3) - rotations)))
            equilibrium = self.equilibrium()
            node = {"phase": float(phase), "qpos": self.data.qpos.tolist(), "feet_error_m": feet_error,
                    "hand_error_m": hand_error, "foot_rotation_matrix_error": orientation_error,
                    "self_penetration_m": float(self_depth), "other_penetration_m": float(other_depth),
                    "hand_gaps_m": [float(mujoco.mj_geomDistance(self.model, self.data, geom, self.task.floor, 2., None)) for geom in self.task.supports[:2]],
                    "equilibrium": equilibrium, "solver_success": bool(solution.success)}
            node["passed"] = bool(feet_error < .001 and orientation_error < .005 and hand_error < .01
                                  and self_depth < .001 and other_depth == 0. and equilibrium["feasible"])
            nodes.append(node)
        return nodes


def main():
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    output = ROOT / "output/rl/balanced_descent_path" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "source_sha256": digest(__file__), "cases": [],
              "protocol": "Planner-only symmetric free-root path, feet fixed, COM over feet, .03rad joint margin, 70% static torque limits. Four predeclared endpoint goals. Static equilibrium uses hypothetical flat-foot corners and is only a screening condition, never dynamic acceptance."}
    try:
        for hand_x, pitch in ((.35, .8), (.5, .8), (.35, 1.4), (.5, 1.4)):
            planner = BalancedDescentPath(task)
            nodes = planner.solve(hand_x, pitch)
            case = {"hand_x_m": hand_x, "pitch_goal_rad": pitch, "nodes": nodes,
                    "passed": all(node["passed"] for node in nodes)}
            report["cases"].append(case)
            print(hand_x, pitch, sum(node["passed"] for node in nodes), nodes[-1]["hand_gaps_m"], flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json")


if __name__ == "__main__":
    main()
