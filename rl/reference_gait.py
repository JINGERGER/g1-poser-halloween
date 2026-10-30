import math

import numpy as np


def paired_cycles(trace, config, dt):
    air_samples = max(1, math.ceil(config["min_air_s"] / dt - 1e-9))
    support_samples = max(1, math.ceil(config["min_support_s"] / dt - 1e-9))
    results = []
    for limb in range(4):
        armed = False
        flying = False
        loaded_run = 0
        air_run = 0
        takeoff = None
        cycles = []
        for row in trace:
            loaded = (row["loads_n"][limb] > config["loaded_n"] and row["floor_contacts"][limb]
                      and row["gaps_m"][limb] <= config["landing_gap_m"])
            airborne = (row["loads_n"][limb] <= config["unloaded_n"] and not row["floor_contacts"][limb]
                        and row["gaps_m"][limb] >= config["clearance_m"])
            loaded_run = loaded_run + 1 if loaded else 0
            if flying:
                if loaded_run >= support_samples:
                    cycles.append({"takeoff_s": takeoff, "landing_s": row["elapsed_s"] - (support_samples - 1) * dt,
                                   "confirmed_s": row["elapsed_s"]})
                    flying = False
                    armed = True
                    air_run = 0
            elif loaded_run >= support_samples:
                armed = True
                air_run = 0
            elif armed:
                air_run = air_run + 1 if airborne else 0
                if air_run >= air_samples:
                    takeoff = row["elapsed_s"] - (air_samples - 1) * dt
                    flying = True
                    armed = False
                    loaded_run = 0
        results.append({"limb": limb, "count": len(cycles), "cycles": cycles, "unfinished_flight": flying})
    return results


def longest_run_seconds(flags, dt):
    longest = 0
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest * dt


def summarize_episode(trace, survived, config, dt):
    if not trace:
        raise ValueError("An empty trajectory cannot be evaluated")
    times = np.array([row["elapsed_s"] for row in trace])
    if not np.allclose(times, np.arange(1, len(trace) + 1) * dt, rtol=0, atol=1e-7):
        raise ValueError("Evaluation requires consecutive control-step samples")
    settled = [row for row in trace if row["elapsed_s"] > config["warmup_s"]]
    cycles = paired_cycles(settled, config, dt)
    result = {"duration_s": len(trace) * dt, "survived": bool(survived), "cycles": cycles,
              "progress_m": trace[-1]["progress_m"], "max_lateral_drift_m": max(abs(row["lateral_m"]) for row in trace),
              "max_penetration_m": max(row["penetration_m"] for row in trace),
              "max_non_support_load_n": max(row["other_load_n"] for row in trace),
              "max_torque_ratio": max(row["max_torque_ratio"] for row in trace),
              "speed_mae_mps": None, "turn_mae_radps": None, "loaded_slip_mps": None,
              "support_coverage": None, "non_support_fraction": None, "saturation_fraction": None,
              "longest_saturation_s": None}
    if settled:
        loads = np.asarray([row["loads_n"] for row in settled])
        slips = np.asarray([row["slip_mps"] for row in settled])
        total = loads.sum(axis=1)
        supported = total > config["loaded_n"]
        near_limit = [row["max_torque_ratio"] >= config["torque_ratio"] for row in settled]
        result.update(
            speed_mae_mps=float(np.mean([row["speed_error_mps"] for row in settled])),
            turn_mae_radps=float(np.mean([row["turn_error_radps"] for row in settled])),
            loaded_slip_mps=float(np.mean((loads[supported] * slips[supported]).sum(axis=1) / total[supported])) if supported.any() else None,
            support_coverage=float(supported.mean()),
            non_support_fraction=float(np.mean([row["other_load_n"] > config["non_support_load_n"] for row in settled])),
            saturation_fraction=float(np.mean(near_limit)), longest_saturation_s=longest_run_seconds(near_limit, dt),
        )
    gates = {
        "completed": survived and result["duration_s"] >= config["duration_s"] - dt / 2,
        "four_limb_cycles": all(row["count"] >= config["cycles_per_limb"] for row in cycles),
        "forward_progress": result["progress_m"] >= config["command_mps"] * config["duration_s"] * config["progress_fraction"],
        "lateral_drift": result["max_lateral_drift_m"] <= config["lateral_drift_m"],
        "penetration": result["max_penetration_m"] <= config["penetration_m"],
    }
    for name in ("speed_mae_mps", "turn_mae_radps", "loaded_slip_mps", "non_support_fraction", "saturation_fraction"):
        gates[name] = result[name] is not None and result[name] <= config[name]
    gates["support_coverage"] = result["support_coverage"] is not None and result["support_coverage"] >= config["support_coverage"]
    result["gates"] = {name: bool(value) for name, value in gates.items()}
    result["passed"] = all(gates.values())
    return result
