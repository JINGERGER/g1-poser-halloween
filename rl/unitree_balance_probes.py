import numpy as np


CASES = [("native_hold", "native_hold", "none"), ("official_zero", "official_zero", "none"),
         ("push_forward", "official_zero", "push_forward"), ("push_lateral", "official_zero", "push_lateral"),
         ("lower_native", "native_hold", "lower"), ("lower_policy", "official_zero", "lower")]


def inputs(kind, step, names):
    force = np.zeros(3)
    offset = np.zeros(len(names))
    if kind not in ("none", "push_forward", "push_lateral", "lower"):
        raise ValueError("Unknown probe")
    if kind.startswith("push") and 1000 <= step < 1100:
        force[0 if kind == "push_forward" else 1] = 15.
    if kind == "lower":
        phase = float(np.clip((step - 1000) / 500, 0., 1.))
        phase = phase * phase * (3. - 2. * phase)
        for index, name in enumerate(names):
            if name.endswith("knee_joint"):
                offset[index] = .2 * phase
            elif name.endswith(("hip_pitch_joint", "ankle_pitch_joint")):
                offset[index] = -.1 * phase
    return force, offset


def assess(kind, samples, baseline, gates, positions, baseline_positions):
    physical = {name: gates[name] for name in ("completed_5s", "replay", "height_055_to_095m", "tilt_below_015rad",
                "penetration_5mm", "self_penetration_1mm", "nonfoot_load_1n", "torque_below_95percent", "joint_limits_1mrad")}
    tail = samples[samples[:, 0] >= 4. - 1e-8]
    base_tail = baseline[baseline[:, 0] >= 4. - 1e-8]
    result = {"physical_gates": physical, "physical_passed": all(physical.values()), "probe_passed": False}
    if len(tail) < 500 or len(base_tail) < 500 or len(tail) != len(base_tail):
        result["reason"] = "Missing full final-second comparison"
        return result
    rms = lambda values: float(np.sqrt(np.mean(values ** 2)))
    result.update(height_reduction_m=float(base_tail[:, 1].mean() - tail[:, 1].mean()),
                  linear_speed_rms=rms(tail[:, 3]), baseline_linear_speed_rms=rms(base_tail[:, 3]),
                  angular_speed_rms=rms(tail[:, 4]), baseline_angular_speed_rms=rms(base_tail[:, 4]),
                  tilt_rms=rms(tail[:, 2]), baseline_tilt_rms=rms(base_tail[:, 2]))
    if kind.startswith("push"):
        drift = np.linalg.norm(positions[-len(tail):, :2] - baseline_positions[-len(base_tail):, :2], axis=1)
        recovery = {"tail_linear_rms": result["linear_speed_rms"] <= result["baseline_linear_speed_rms"] + .05,
                    "tail_angular_rms": result["angular_speed_rms"] <= result["baseline_angular_speed_rms"] + .1,
                    "tail_tilt_rms": result["tilt_rms"] <= result["baseline_tilt_rms"] + .03,
                    "tail_displacement_vs_baseline_15cm": bool(np.max(drift) <= .15)}
        result.update(recovery_gates=recovery, max_tail_position_difference_m=float(np.max(drift)),
                      probe_passed=bool(all(physical.values()) and all(recovery.values())))
    elif kind == "lower":
        result["probe_passed"] = bool(all(physical.values()) and result["height_reduction_m"] >= .01)
    else:
        result["probe_passed"] = all(physical.values())
    return result
