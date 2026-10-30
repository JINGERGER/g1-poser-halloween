import json
from datetime import datetime

import numpy as np
import torch

from artifact_utils import digest
from coordinated_blend_task import CoordinatedBlendTask
from reference_policy_baseline import ROOT
from search_coordinated_contact import rollout


SOURCE = ROOT / "output/rl/coordinated_contact_search/2026-10-30_15-51-20-975178/report.json"


def main():
    source = json.loads(SOURCE.read_text())
    best = next(family["verified_best"] for family in source["families"] if family["family"] == "fold_leads")
    assert digest(ROOT / best["trajectory"]) == best["trajectory_sha256"]
    torch.set_num_threads(1)
    task = CoordinatedBlendTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/coordinated_blend" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("check_coordinated_blend.py", "coordinated_blend_task.py", "search_coordinated_contact.py",
             "position_torque_guard.py", "transition_execution.py", "near_handoff_task.py")
    report = {"status": "running", "full_transition_validated": False, "cases": [],
              "source_report": str(SOURCE.relative_to(ROOT)), "source_report_sha256": digest(SOURCE),
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    for delay in (0., .04, .08):
        for duration in (.1, .2, .4):
            task.blend_duration = duration
            result = rollout(task, state, np.array(best["parameters"]), output / f"delay_{delay:.2f}_blend_{duration:.1f}.npz", handoff_delay=delay)
            result["handoff_delay_s"] = delay
            report["cases"].append(result)
            report["full_transition_validated"] |= result["success"]
            print(delay, duration, "success", result["success"], "failure", result["failure"], "handoff", result["handoff"], flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "completed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
