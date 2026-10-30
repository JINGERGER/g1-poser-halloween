import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from prepare import ROOT, SUPPORTS
from reference_gait import paired_cycles


class PeriodicReference:
    def __init__(self, joints):
        self.knots = np.r_[np.arange(80) / 100, 1.0]
        self.values = np.vstack((joints, joints[0]))
        widths = np.diff(self.knots)
        slopes = np.diff(self.values, axis=0) / widths[:, None]
        previous = np.roll(slopes, 1, axis=0)
        previous_widths = np.roll(widths, 1)
        weight_left = (2 * widths + previous_widths)[:, None]
        weight_right = (widths + 2 * previous_widths)[:, None]
        same = previous * slopes > 0
        denominator = weight_left / np.where(same, previous, 1) + weight_right / np.where(same, slopes, 1)
        derivatives = np.where(same, (weight_left + weight_right) / np.where(same, denominator, 1), 0)
        self.derivatives = np.vstack((derivatives, derivatives[0]))

    def __call__(self, phase, derivative=0):
        phase = float(phase) % 1.0
        index = min(np.searchsorted(self.knots, phase, side="right") - 1, 79)
        width = self.knots[index + 1] - self.knots[index]
        fraction = (phase - self.knots[index]) / width
        left, right = self.values[index:index + 2]
        slope_left, slope_right = self.derivatives[index:index + 2]
        if derivative == 0:
            return ((2 * fraction ** 3 - 3 * fraction ** 2 + 1) * left
                    + (fraction ** 3 - 2 * fraction ** 2 + fraction) * width * slope_left
                    + (-2 * fraction ** 3 + 3 * fraction ** 2) * right
                    + (fraction ** 3 - fraction ** 2) * width * slope_right)
        if derivative == 1:
            return ((6 * fraction ** 2 - 6 * fraction) * left / width
                    + (3 * fraction ** 2 - 4 * fraction + 1) * slope_left
                    + (-6 * fraction ** 2 + 6 * fraction) * right / width
                    + (3 * fraction ** 2 - 2 * fraction) * slope_right)
        raise ValueError("Only reference position and velocity are supported")


def trajectory(frames):
    joints = np.asarray(frames)[:, 7:]
    if joints.shape != (80, 23) or not np.isfinite(joints).all():
        raise ValueError("Expected the pinned 80-frame, 23-joint reference")
    return PeriodicReference(joints)


def initial_pose(model, frames):
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    candidates = []
    for index, frame in enumerate(frames):
        data.qpos[:] = frame
        data.qpos[:2] = 0
        data.qpos[7:] = np.clip(data.qpos[7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1])
        mujoco.mj_forward(model, data)
        distances = [mujoco.mj_geomDistance(model, data, geom, floor, 2, None)
                     for geom in range(model.ngeom) if geom != floor and (model.geom_contype[geom] or model.geom_conaffinity[geom])]
        projection = 0.002 - min(distances)
        data.qpos[2] += projection
        mujoco.mj_forward(model, data)
        gaps = np.array([mujoco.mj_geomDistance(model, data, model.geom(name).id, floor, 2, None) for name in SUPPORTS])
        candidates.append((int(np.sum(gaps < 0.012)), -float(gaps.sum()), index, data.qpos.copy(), projection))
    selected = max(candidates, key=lambda row: row[:2])
    return selected[2], selected[3], selected[4]


def phase_clock(elapsed, period):
    moving = max(0.0, elapsed - 0.5)
    ramp = min(moving, 1.0)
    return (ramp ** 3 - 0.5 * ramp ** 4 + max(0.0, moving - 1.0)) / period, (3 * ramp ** 2 - 2 * ramp ** 3) / period


def run_candidate(model_path, frames, period, feedforward, output):
    from evaluate_support import metrics

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    start, pose, projection = initial_pose(model, frames)
    curve = trajectory(frames)
    data.qpos[:] = pose
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    heading = rotation.reshape(3, 3)[:2, 2].copy()
    heading /= np.linalg.norm(heading)
    lateral = np.array([-heading[1], heading[0]])
    origin = data.qpos[:2].copy()
    timestep = float(model.opt.timestep)
    substeps = round(0.02 / timestep)
    if not np.isclose(substeps * timestep, 0.02):
        raise ValueError("Expected exact 20ms control period")
    duration = 1.0 + 2 * period
    states, velocities, controls, trace = [data.qpos.copy()], [data.qvel.copy()], [], []
    failure = None
    max_penetration = 0.0
    peak_ratio = 0.0
    clipping = 0
    joint_ids = model.actuator_trnid[:, 0]
    if not np.array_equal(model.jnt_qposadr[joint_ids], np.arange(7, 30)):
        raise ValueError("Actuator order differs from reference frame joint order")
    limits = model.jnt_actfrcrange[joint_ids]
    started = time.perf_counter()
    for step in range(round(duration / 0.02)):
        phase, rate = phase_clock(data.time, period)
        target = curve((start / 100 + phase) % 1.0)
        if feedforward:
            target = target + (-model.actuator_biasprm[:, 2] / model.actuator_gainprm[:, 0]) * curve((start / 100 + phase) % 1.0, 1) * rate
        bounded = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        clipping += int(np.count_nonzero(bounded != target))
        data.ctrl[:] = bounded
        controls.append(bounded.copy())
        for substep in range(substeps):
            if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise AssertionError("External assistance is forbidden")
            mujoco.mj_step(model, data)
            max_penetration = max(max_penetration, max((-float(contact.dist) for contact in data.contact), default=0.0))
            forces = data.qfrc_actuator[model.jnt_dofadr[joint_ids]]
            peak_ratio = max(peak_ratio, float(np.max(np.abs(forces) / np.where(forces >= 0, limits[:, 1], -limits[:, 0]))))
        mujoco.mj_forward(model, data)
        max_penetration = max(max_penetration, max((-float(contact.dist) for contact in data.contact), default=0.0))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or abs(data.time - (step + 1) * 0.02) > 1e-7:
            raise ValueError("Invalid simulation state or implicit simulator reset")
        row = metrics(model, data)
        row.update(elapsed_s=(step + 1) * 0.02, progress_m=float((data.qpos[:2] - origin) @ heading),
                   lateral_m=float((data.qpos[:2] - origin) @ lateral))
        trace.append(row)
        states.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        gravity_error = float(np.sum((-rotation.reshape(3, 3)[2] - [1, 0, 0]) ** 2))
        if data.qpos[2] < 0.08 or data.qpos[2] > 1 or gravity_error > 3.5:
            failure = "base_height_or_orientation"
            break
        if max_penetration > 0.03:
            failure = "penetration_over_3cm"
            break
    protocol = json.loads(Path(__file__).with_name("reference_eval.json").read_text())
    settled = [row for row in trace if row["elapsed_s"] > 0.5]
    cycles = paired_cycles(settled, protocol, 0.02)
    loaded = [row for row in settled if sum(row["loads_n"]) > 1]
    slip = float(np.mean([np.average(row["slip_mps"], weights=row["loads_n"]) for row in loaded])) if loaded else None
    other = float(np.mean([row["other_load_n"] > 1 for row in settled])) if settled else None
    coverage = len(loaded) / len(settled) if settled else None
    saturation = float(np.mean([row["max_torque_ratio"] >= 0.95 for row in settled])) if settled else None
    gates = {"completed": failure is None, "each_limb_two_cycles": all(row["count"] >= 2 for row in cycles),
             "progress_20cm": trace[-1]["progress_m"] >= 0.2,
             "lateral_under_15cm": max(abs(row["lateral_m"]) for row in trace) <= 0.15,
             "non_support_fraction": other is not None and other <= 0.01,
             "loaded_slip": slip is not None and slip <= 0.02,
             "support_coverage": coverage is not None and coverage >= 0.8,
             "saturation_fraction": saturation is not None and saturation <= 0.1,
             "penetration_under_5mm": max_penetration <= 0.005,
             "force_limits": peak_ratio <= 1.000001}
    np.savez_compressed(output, qpos=np.array(states), qvel=np.array(velocities), ctrl=np.array(controls), dt=0.02)
    return {"period_s": period, "velocity_feedforward": feedforward, "initial_frame": start,
            "initial_floor_projection_m": projection, "model": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "duration_s": trace[-1]["elapsed_s"], "planned_duration_s": duration, "failure": failure,
            "progress_m": trace[-1]["progress_m"], "cycles": cycles, "mean_loaded_slip_mps": slip,
            "non_support_fraction": other, "max_penetration_m": max_penetration, "peak_torque_ratio": peak_ratio,
            "support_coverage": coverage, "saturation_fraction": saturation,
            "clipped_target_components": clipping, "gates": gates, "passed": all(gates.values()),
            "trajectory": str(output), "trace": trace, "wall_seconds": time.perf_counter() - started}


def record(result, output):
    import imageio.v2 as imageio

    model = mujoco.MjModel.from_xml_path(result["model"])
    data = mujoco.MjData(model)
    saved = np.load(result["trajectory"], allow_pickle=False)
    data.qpos[:] = saved["qpos"][0]
    data.qvel[:] = saved["qvel"][0]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance, camera.azimuth, camera.elevation = 2.3, 135, -22
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(output, mode="I", duration=100, loop=0) as writer:
        for step in range(len(saved["ctrl"]) + 1):
            if step:
                data.ctrl[:] = saved["ctrl"][step - 1]
                for substep in range(round(0.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if not np.allclose(data.qpos, saved["qpos"][step], atol=1e-8, rtol=0):
                    raise AssertionError("Recorded controls do not reproduce the saved physical trajectory")
            if step % 5 == 0 or step == len(saved["ctrl"]):
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())


def main():
    from reference_assets import prepare_reference

    parser = argparse.ArgumentParser(description="Bounded physical reference-tracking construction; never a kinematic playback")
    parser.add_argument("--record", type=Path, help="Replay a completed report and render its explicitly selected illustration")
    args = parser.parse_args()
    if args.record:
        report = json.loads(args.record.read_text())
        record(report["runs"][report["illustration_index"]], args.record.with_suffix(".gif"))
        return
    output = ROOT / "output/rl/short_crawl" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "runs": [], "training_transitions": 0,
              "protocol": "short-reference-tracking-v1", "max_candidates": 12,
              "limitations": "Constructed periodic joint reference with a synthetic 21%-period closing segment; not an executable expert motion. No base pinning, external force, state overwrite after reset, model/force-limit changes or learned policy. 2mm initial floor projection only. Substep penetration/torque checks; 20ms gait/contact metrics. A successful example would establish nominal simulation execution, not robustness or hardware feasibility.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        for variant in ("current", "reference"):
            assets = prepare_reference(variant)
            frames = np.load(assets / "reset_frames.npy", allow_pickle=False)
            report["animation_joint_seam_rad"] = float(np.max(np.abs(frames[-1, 7:] - frames[0, 7:])))
            report["frames_sha256"] = hashlib.sha256((assets / "reset_frames.npy").read_bytes()).hexdigest()
            for period in (2.0, 4.0, 6.0):
                for feedforward in (False, True):
                    result = run_candidate(assets / "flat.xml", frames, period, feedforward, output / f"{variant}_{period}_{int(feedforward)}.npz")
                    result["pd"] = variant
                    report["runs"].append(result)
                    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    print(json.dumps({key: result[key] for key in ("pd", "period_s", "velocity_feedforward", "duration_s", "progress_m", "failure", "passed")}), flush=True)
        report["status"] = "completed"
        report["success_count"] = sum(row["passed"] for row in report["runs"])
        report["illustration_selection"] = "Most total completed limb cycles, then longest duration; illustration only, all candidates reported"
        report["illustration_index"] = max(range(len(report["runs"])), key=lambda index: (sum(row["count"] for row in report["runs"][index]["cycles"]), report["runs"][index]["duration_s"]))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
