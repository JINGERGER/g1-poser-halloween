import argparse
import json
from datetime import datetime

import numpy as np

from artifact_utils import digest
from balanced_descent_path import BalancedDescentPath
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true")
    args = parser.parse_args()
    task = StandTransitionTask()
    task.reset(task.make_states([0])[0])
    output = ROOT / "output/rl/descent_endpoints" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "sources": {name: digest(ROOT / "rl" / name) for name in
              ("balanced_descent_path.py", "check_descent_endpoints.py")},
              "protocol": "Endpoint-only architecture screening, not executable paths: fixed narrow feet versus widened/yawed feet. Two arm seeds crossed with three leg seeds. No robot state is executed or altered by planning.", "cases": []}
    try:
        configurations = ((.26, 1.4), (.26, 1.8)) if args.wide else ((0., 0.), (.12, .4), (.20, .8))
        for spread, yaw in configurations:
            for arm_seed in ("standing", "crawl"):
                for leg_seed in (0, 1, 2):
                    planner = BalancedDescentPath(task)
                    if arm_seed == "crawl":
                        planner.values[9:] = task.nominal[task.model.jnt_qposadr[planner.left[6:]]]
                    if leg_seed:
                        planner.values[:3] = [-.05, .4, 1.2]
                        planner.values[3:9] = [-1.5, .4, .4, 1.8, -.4, 0.] if leg_seed == 1 else [-1.6, 2., 1., 2., -.1, 0.]
                    planner.values = np.clip(planner.values, planner.lower + 1e-6, planner.upper - 1e-6)
                    node = planner.solve(.4, 1.3, spread, yaw, endpoint=True)[0]
                    case = {"spread_m": spread, "yaw_rad": yaw, "arm_seed": arm_seed, "leg_seed": leg_seed, "node": node}
                    report["cases"].append(case)
                    print(spread, yaw, arm_seed, leg_seed, node["passed"], round(node["hand_error_m"], 3), round(node["self_penetration_m"], 4), flush=True)
                    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output / "report.json")


if __name__ == "__main__":
    main()
