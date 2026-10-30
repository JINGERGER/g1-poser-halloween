import argparse
import json
from datetime import datetime

import numpy as np
import torch

from artifact_utils import digest
from coordinated_blend_task import CoordinatedBlendTask
from reference_policy_baseline import ROOT
from search_coordinated_contact import rollout


CONFIG = ROOT / "rl/coordinated_transition_v1.json"


def main():
    parser = argparse.ArgumentParser(description="Nominal standing-to-crawl simulator candidate; not hardware validated")
    parser.add_argument("--evaluate-perturbations", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    torch.set_num_threads(1)
    task = CoordinatedBlendTask()
    for name in ("policy", "bundle", "model"):
        assert digest(getattr(task, name + "_path")) == config[name + "_sha256"]
    search = json.loads((ROOT / config["search_report"]).read_text())
    for path, expected in search["model_assets"].items():
        assert digest(task.model_path.parent / path) == expected
    task.blend_duration = config["blend_duration_s"]
    output = ROOT / "output/rl/coordinated_transition_v1" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    names = ("run_coordinated_transition.py", "coordinated_blend_task.py", "search_coordinated_contact.py",
             "position_torque_guard.py", "stand_transition_task.py", "near_handoff_task.py", "transition_execution.py")
    report = {"status": "running", "full_transition_validated": False, "cases": [],
              "validation_scope": "nominal_zero_velocity_only", "model_assets": search["model_assets"],
              "config": config, "config_sha256": digest(CONFIG),
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in names}}
    perturbations = [("nominal", 0, 0.)]
    if args.evaluate_perturbations:
        perturbations += [(f"velocity_{axis}_{speed:+.2f}", axis, speed)
                          for axis in (0, 1) for speed in (-.03, -.01, .01, .03)]
    for name, axis, speed in perturbations:
        state = task.make_states([0])[0]
        state["qvel"] = np.zeros(task.model.nv).tolist()
        state["qvel"][axis] = speed
        result = rollout(task, state, np.array(config["parameters"]), output / f"{name}.npz", config["handoff_delay_s"])
        result.update(label=name, initial_qvel=state["qvel"])
        report["cases"].append(result)
        if name == "nominal":
            report["full_transition_validated"] = result["success"]
        print(name, "success", result["success"], "failure", result["failure"], "elapsed", result["elapsed_s"], flush=True)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report.update(status="completed", perturbations_tested=len(perturbations) - 1,
                  perturbations_passed=sum(case["success"] for case in report["cases"][1:]))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
