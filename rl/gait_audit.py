import numpy as np


def summarize_gait(trace):
    limbs = []
    for limb in range(4):
        peak = [row for row in trace if 0.825 < row["phase"][limb] < 0.925]
        landing = [row for row in trace if row["phase"][limb] < 0.1]
        limbs.append({
            "peak_samples": len(peak),
            "mean_peak_geom_gap_m": float(np.mean([row["gaps_m"][limb] for row in peak])) if peak else None,
            "peak_contact_fraction": float(np.mean([row["floor_contacts"][limb] for row in peak])) if peak else None,
            "peak_clear_15mm_unloaded_fraction": float(np.mean([row["gaps_m"][limb] >= 0.015 and not row["floor_contacts"][limb] and row["loads_n"][limb] <= 1 for row in peak])) if peak else None,
            "landing_samples": len(landing),
            "landing_loaded_fraction": float(np.mean([row["loads_n"][limb] > 1 for row in landing])) if landing else None,
        })
    return {"per_limb": limbs, "final_progress_m": trace[-1]["progress_m"] if trace else None,
            "mean_command_mps": float(np.mean([row["command_mps"] for row in trace])) if trace else None,
            "speed_mae_mps": float(np.mean([abs(row["forward_speed_mps"] - row["command_mps"]) for row in trace])) if trace else None,
            "note": "Independent phase-window samples, not paired complete-cycle success; geom distance and native floor contact/load, not site height. No locomotion pass label."}
