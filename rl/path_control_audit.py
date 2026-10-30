import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from load_transfer import physical_metrics
from prepare import ROOT
from servo_force import position_servo_force, validate_position_servos
from target_load_ablation import digest


def acceleration(model, data):
    return {"base_linear_mps2": float(np.linalg.norm(data.qacc[:3])),
            "angular_joint_max_radps2": float(np.max(np.abs(data.qacc[3:])))}


class FixedPoseFit:
    def __init__(self, model, qpos, ctrl):
        self.model, self.data = model, mujoco.MjData(model)
        self.qpos, self.initial_ctrl = np.array(qpos), np.array(ctrl)
        self.calls = 0

    def residual(self, ctrl):
        self.calls += 1
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        return np.r_[self.data.qacc / np.r_[[9.81] * 3, [10.] * 26], (ctrl - self.initial_ctrl) * .001]

    def solve(self):
        self.residual(self.initial_ctrl)
        before = acceleration(self.model, self.data)
        result = least_squares(self.residual, self.initial_ctrl,
                               bounds=(self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]),
                               max_nfev=100, diff_step=1e-6, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        self.residual(result.x)
        after = acceleration(self.model, self.data)
        metrics = physical_metrics(self.model, self.data)
        equilibrium = after["base_linear_mps2"] <= .01 and after["angular_joint_max_radps2"] <= .1
        safe = metrics["penetration"]["depth_m"] <= .005 and metrics["other_total_n"] <= 1 and max(metrics["torque_ratios"]) < .95
        return {"before": before, "after": after, "ctrl": result.x.tolist(), "qpos": self.qpos.tolist(),
                "solver_success": bool(result.success), "message": str(result.message), "nfev": result.nfev,
                "residual_calls_including_finite_differences": self.calls, "native_metrics": metrics,
                "equilibrium_passed": bool(equilibrium and safe), "max_ctrl_change_rad": float(np.max(np.abs(result.x - self.initial_ctrl)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    assert source["status"] == "completed"
    for name, expected in source["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["baseline_report"]) == source["baseline_report_sha256"]
    baseline = json.loads(Path(source["baseline_report"]).read_text())
    assert digest(baseline["geometry_report"]) == baseline["geometry_report_sha256"]
    geometry = json.loads(Path(baseline["geometry_report"]).read_text())
    assert digest(geometry["source_report"]) == geometry["source_report_sha256"]
    original = json.loads(Path(geometry["source_report"]).read_text())
    assert digest(original["model"]) == original["model_sha256"] and mujoco.__version__ == original["mujoco"]
    model = mujoco.MjModel.from_xml_path(original["model"])
    validate_position_servos(model)
    run = next(run for run in source["runs"] if run["branch"] == "shift_then_lift_rotation0")
    path = next(path for path in source["paths"] if path["branch"] == "shift_then_lift" and not path["rotating"])
    assert run["physical_safety_passed"] and run["duration_s"] == 6 and path["passed"]
    initial = np.array(geometry["initial"]["qpos"])
    initial_ctrl = np.array(original["initial_ctrl"])
    bias = initial_ctrl - initial[7:]
    actual, desired = mujoco.MjData(model), mujoco.MjData(model)
    rows, fits = [], []
    with np.load(run["trajectory"], allow_pickle=False) as saved:
        for node in path["nodes"]:
            index = round(node["time_s"] / .02)
            actual.qpos[:], actual.qvel[:], actual.ctrl[:] = saved["qpos"][index], saved["qvel"][index], saved["ctrl"][index - 1]
            desired.qpos[:], desired.qvel[:], desired.ctrl[:] = node["qpos"], 0, np.array(node["qpos"])[7:] + bias
            mujoco.mj_forward(model, actual)
            mujoco.mj_forward(model, desired)
            np.testing.assert_allclose(physical_metrics(model, actual)["loads_n"], run["trace"][index - 1]["loads_n"], atol=1e-7, rtol=0)
            force = position_servo_force(model, actual.ctrl, actual.qpos[7:], actual.qvel[6:])
            np.testing.assert_allclose(force, actual.qfrc_actuator[6:], atol=1e-9, rtol=0)
            errors = actual.qpos[7:] - desired.qpos[7:]
            worst = int(np.argmax(np.abs(errors)))
            rows.append({"time_s": node["time_s"], "actual_com_xyz_m": actual.subtree_com[1].tolist(),
                         "desired_com_xyz_m": desired.subtree_com[1].tolist(),
                         "com_xy_error_m": float(np.linalg.norm(actual.subtree_com[1, :2] - desired.subtree_com[1, :2])),
                         "joint_tracking_max_rad": float(np.max(np.abs(errors))),
                         "worst_joint": model.joint(int(model.actuator_trnid[worst, 0])).name,
                         "joint_errors_rad": errors.tolist(), "actual_torques_nm": force.tolist(),
                         "actual_metrics": physical_metrics(model, actual), "desired_static_metrics": physical_metrics(model, desired),
                         "desired_static_acceleration": acceleration(model, desired),
                         "actual_velocity_norm": float(np.linalg.norm(actual.qvel)),
                         "actual_acceleration": acceleration(model, actual)})
        for elapsed in (0., 3., 5.):
            qpos = initial if elapsed == 0 else np.array(next(node for node in path["nodes"] if node["time_s"] == elapsed)["qpos"])
            result = FixedPoseFit(model, qpos, qpos[7:] + bias).solve()
            result["time_s"] = elapsed
            fits.append(result)
            print(json.dumps({key: value for key, value in result.items() if key not in ("qpos", "ctrl", "native_metrics")}), flush=True)
    destination = ROOT / "output/rl/path_control_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "completed", "training_updates": 0, "source_report": str(args.report.resolve()),
              "source_report_sha256": digest(args.report), "trajectory": run["trajectory"], "trajectory_sha256": digest(run["trajectory"]),
              "source_sha256": {**source["source_sha256"], Path(__file__).name: digest(__file__)}, "tracking": rows, "fixed_pose_fits": fits,
              "protocol": {"fixed_path": "shift_then_lift_rotation0", "static_times_s": [0, 3, 5], "max_nfev_per_fit": 100, "restarts": 0},
              "limitations": "Audit only: fixed-pose fits reset scratch velocity to zero, never the live trajectory. Native contact forces depend on pose and control; no external force allocation. Failure of bounded local control-only fitting is not proof no equilibrium exists. Even an equilibrium pass needs native hold and original-state transfer. No new dynamic rollout or successful lift demonstration."}
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"final_tracking": {key: rows[-1][key] for key in ("com_xy_error_m", "joint_tracking_max_rad", "worst_joint", "desired_static_acceleration")}}))
    print(destination / "report.json")


if __name__ == "__main__":
    main()
