import numpy as np


def summarize_windows(trace):
    summaries = []
    for start, end in ((0, 1), (1, 5), (5, 10), (10, 20)):
        rows = [row for row in trace if start < row["elapsed_s"] <= end]
        summary = {"start_s": start, "end_s": end, "samples": len(rows)}
        if rows:
            loads = np.array([row["loads_n"] for row in rows])
            slips = np.array([row["slip_mps"] for row in rows])
            loaded = loads > 1.0
            total = loads.sum(axis=1)
            supported = total > 1.0
            speeds = (loads * slips).sum(axis=1) / np.maximum(total, 1e-12)
            summary.update({
                "loaded_samples": int(supported.sum()),
                "mean_force_weighted_slip_mps": float(speeds[supported].mean()) if supported.any() else None,
                "limb_mean_slip_when_loaded_mps": [float(slips[loaded[:, limb], limb].mean()) if loaded[:, limb].any() else None for limb in range(4)],
                "limb_loaded_fraction": loaded.mean(axis=0).tolist(),
                "all_four_loaded_fraction": float(loaded.all(axis=1).mean()),
                "mean_base_speed_mps": float(np.mean([row["base_speed_mps"] for row in rows])),
                "max_base_drift_m": max(row["drift_m"] for row in rows),
            })
        summaries.append(summary)
    return summaries
