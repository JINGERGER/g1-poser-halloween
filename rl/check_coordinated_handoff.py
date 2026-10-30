import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from artifact_utils import digest
from reference_policy_baseline import ROOT
from search_coordinated_contact import rollout
from stand_transition_task import StandTransitionTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source_path = args.report.resolve()
    source = json.loads(source_path.read_text())
    best = next(family["verified_best"] for family in source["families"] if family["family"] == "fold_leads")
    assert best["replay_passed"] and digest(ROOT / best["trajectory"]) == best["trajectory_sha256"]
    torch.set_num_threads(1)
    task = StandTransitionTask()
    assert digest(task.policy_path) == source["policy_sha256"] and digest(task.bundle_path) == source["bundle_sha256"]
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/coordinated_handoff" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_coordinated_handoff.py", "search_coordinated_contact.py", "near_handoff_task.py", "transition_execution.py")
    report = {"status": "running", "full_transition_validated": False, "cases": [],
              "source_report": str(source_path.relative_to(ROOT)), "source_report_sha256": digest(source_path),
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names},
              "protocol": "True standing in each case; unchanged joint trajectory. Handoff after bilateral loads >5N for 0/40/80/120/160/200ms. Original physical and 2s frozen-crawl success gates unchanged; no state reset at handoff."}
    for delay in (0., .04, .08, .12, .16, .2):
        result = rollout(task, state, np.array(best["parameters"]), output / f"delay_{delay:.2f}.npz", handoff_delay=delay)
        result["handoff_delay_s"] = delay
        report["cases"].append(result)
        report["full_transition_validated"] |= result["success"]
        print("delay", delay, "success", result["success"], "failure", result["failure"], "handoff", result["handoff"], flush=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "completed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
