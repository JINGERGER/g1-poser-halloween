import argparse
import json
from pathlib import Path

import numpy as np

from artifact_utils import digest


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def reconstruct(samples, positions):
    times = samples[:, 0]
    events = []
    for limb in range(4):
        load = samples[:, 4 + limb]
        gap = samples[:, 8 + limb]
        limb_events = []
        for start, end in intervals((load <= 1.) & (gap > 0.)):
            clear_times = [times[start + stop - 1] - times[start + begin]
                           for begin, stop in intervals(gap[start:end] >= .01)]
            clearance_duration = max(clear_times, default=0.)
            advance = float(positions[end - 1, limb, 0] - positions[start, limb, 0])
            landing = np.flatnonzero((times >= times[end - 1]) & (times <= times[end - 1] + .02) & (load > 5.))
            loaded = False
            land_time = None
            if len(landing):
                land = int(landing[0])
                land_time = float(times[land])
                finish = int(np.searchsorted(times, times[land] + .04))
                loaded = finish < len(times) and bool(np.all(load[land:finish + 1] > 5.))
            limb_events.append({
                "start_s": float(times[start]), "air_end_s": float(times[end - 1]),
                "land_s": land_time, "continuous_10mm_s": float(clearance_duration),
                "max_gap_m": float(gap[start:end].max()),
                "airborne_forward_m": advance, "loaded_40ms": loaded,
                "qualified": bool(clearance_duration >= .08 - 1e-10 and advance >= .02 and loaded),
            })
        events.append(limb_events)
    cycles = []
    steps = [[event for event in limb if event["max_gap_m"] >= .01] for limb in events]
    for index, (first, next_first) in enumerate(zip(steps[0], steps[0][1:])):
        start, finish = first["start_s"], next_first["start_s"]
        period = finish - start
        selected = [[event for event in limb if start <= event["start_s"] < finish] for limb in steps]
        if not all(len(group) == 1 and group[0]["qualified"] for group in selected):
            continue
        chosen = [group[0] for group in selected]
        phase = [(event["start_s"] - start) / period for event in chosen]
        diagonal_alternation = (
            phase[3] <= .25 and abs(phase[1] - phase[2]) <= .25
            and .25 <= phase[1] <= .75 and .25 <= phase[2] <= .75
        )
        if diagonal_alternation and all(event["land_s"] < finish for event in chosen):
            cycles.append({"anchor_index": index, "start_s": start, "end_s": finish, "phases": phase})
    consecutive = any(right["anchor_index"] == left["anchor_index"] + 1 for left, right in zip(cycles, cycles[1:]))
    return events, cycles, consecutive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    audit_path = args.report.with_name("physics_audit_v3.json")
    audit = json.loads(audit_path.read_text())
    assert audit["report_sha256"] == digest(args.report)
    assert report["status"] == "completed" and len(report["runs"]) == len(audit["runs"]) == 6
    results = []
    for trial, checked in zip(report["runs"], audit["runs"]):
        assert trial["trajectory"] == checked["trajectory"] and checked["replay_passed"]
        assert digest(trial["trajectory"]) == trial["trajectory_sha256"]
        assert digest(checked["samples"]) == checked["samples_sha256"]
        saved = np.load(checked["samples"], allow_pickle=False)
        samples, positions = saved["samples"], saved["support_positions"]
        assert np.isfinite(samples).all() and np.isfinite(positions).all()
        events, cycles, consecutive = reconstruct(samples, positions)
        gates = {
            "completed_10s": trial["failure"] is None and trial["elapsed_s"] >= 10. - 1e-8,
            "forward_6cm": trial["base_displacement_m"][0] >= .06,
            "penetration_5mm": max(float(saved["penetrations"].max()), checked["initial_penetration_m"]) <= .005,
            "self_penetration_1mm": max(float(saved["self_penetrations"].max()), checked["initial_self_penetration_m"]) <= .001,
            "other_floor_load_1n": float(samples[:, 12].max()) <= 1.,
            "torque_below_95percent": float(samples[:, 13].max()) < .95,
            "joint_limits": float(samples[:, 14].max()) <= .001,
            "base_height": bool(np.all((samples[:, 3] >= .16) & (samples[:, 3] <= .65))),
            "two_consecutive_alternating_cycles": consecutive,
        }
        result = {"case_index": trial["case_index"], "passed": all(gates.values()), "gates": gates,
                  "qualified_events": [sum(event["qualified"] for event in limb) for limb in events],
                  "events": events, "cycles": cycles}
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("events", "cycles")}), flush=True)
    result = {
        "source_sha256": digest(__file__), "report_sha256": digest(args.report), "audit_sha256": digest(audit_path),
        "nominal_three_passed": all(trial["passed"] for trial in results[:3]),
        "numerical_and_yaw_checks_passed": all(trial["passed"] for trial in results[3:]),
        "scope": "Continuous diagonal crawl functional audit on the explicitly recorded contact/controller profile. "
                 "Not the original static short-step target/hold task, not hardware safety or general terrain robustness. "
                 "Airborne movement and clearance reconstructed independently of diagnostic event counters.",
        "runs": results,
    }
    output = args.report.with_name("crawl_verification_v2.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
