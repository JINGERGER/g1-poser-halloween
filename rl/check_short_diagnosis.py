import argparse
import copy
import json
from pathlib import Path

import numpy as np

from diagnose_short_crawl import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path)
    args = parser.parse_args()
    base = {"time_s": 0.0, "penetration": {"depth_m": 0.005}, "other_total_n": 1,
            "tracking_error_max_rad": 0.1, "near_limit_joints": []}
    assert all(value is None for value in summarize([base]).values())
    failed = copy.deepcopy(base)
    failed.update(time_s=0.02, other_total_n=2, tracking_error_max_rad=0.2, near_limit_joints=["joint"])
    failed["penetration"]["depth_m"] = 0.006
    assert all(value["time_s"] == 0.02 for value in summarize([base, failed, failed]).values())
    assert all(value is None for value in summarize([]).values())
    print("PASS: strict event thresholds, first-event selection and absent events")
    if args.diagnosis:
        diagnosis = json.loads(args.diagnosis.read_text())
        assert len(diagnosis["runs"]) == 12 and len(diagnosis["hold_controls"]) == 2
        for hold in diagnosis["hold_controls"]:
            assert hold["verified_state_error"] <= 1e-8
            for run in diagnosis["runs"]:
                if run["pd"] != hold["pd"]:
                    continue
                assert run["verified_state_error"] <= 1e-8
                for actual, fixed in zip(run["trace"][:26], hold["trace"][:26]):
                    for key in ("time_s", "base_height_m", "loads_n", "tracking_error_rad", "torque_ratios"):
                        np.testing.assert_allclose(actual[key], fixed[key], atol=1e-10, rtol=0)
        print("PASS: all saved controls replay exactly and every candidate shares its PD-matched pre-motion trajectory")


if __name__ == "__main__":
    main()
