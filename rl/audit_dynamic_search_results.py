import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    if source["status"] != "completed":
        raise ValueError("Search must be completed before final audit")
    torch.set_num_threads(1)
    task = StandTransitionTask()
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    if source["assets"] != assets or source["policy_sha256"] != digest(task.policy_path):
        raise ValueError("Replay assets changed")
    candidates = [candidate for run in source["runs"] for candidate in run["candidates"]]
    heldout = [candidate for run in source["runs"] for candidate in run["heldout"]]
    assert len(candidates) == source["budget"]["search_rollouts"]
    assert len(heldout) == source["budget"]["heldout_rollouts"]
    assert all(candidate["replay_passed"] for candidate in candidates + heldout)
    audit = {"source_sha256": digest(args.report), "audit_source_sha256": digest(__file__),
             "search_count": len(candidates), "heldout_count": len(heldout),
             "search_successes": sum(candidate["success"] for candidate in candidates),
             "heldout_successes": sum(candidate["success"] for candidate in heldout),
             "failure_counts": dict(Counter(reason for candidate in candidates for reason in candidate["failure"])),
             "selected_replays": []}
    for run in source["runs"]:
        selected = next(candidate for candidate in run["candidates"]
                        if candidate["generation"] == run["selected"]["generation"] and candidate["index"] == run["selected"]["index"])
        path = ROOT / selected["trajectory"]
        assert digest(path) == selected["trajectory_sha256"]
        with np.load(path) as trajectory:
            task.reset(source["search_initial_state"])
            for index, control in enumerate(trajectory["ctrl"]):
                if trajectory["phase"][index] and task.execution.phase != "crawl":
                    task.execution.handoff()
                task.data.ctrl[:] = control
                mujoco.mj_step(task.model, task.data)
                mujoco.mj_forward(task.model, task.data)
                np.testing.assert_allclose(task.data.qpos, trajectory["qpos"][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(task.data.qvel, trajectory["qvel"][index + 1], atol=1e-10, rtol=0)
        contacts = []
        for index, contact in enumerate(task.data.contact):
            if task.floor not in (contact.geom1, contact.geom2):
                continue
            geom = contact.geom2 if contact.geom1 == task.floor else contact.geom1
            force = np.zeros(6)
            mujoco.mj_contactForce(task.model, task.data, index, force)
            contacts.append({"geom": task.model.geom(geom).name,
                             "body": task.model.body(int(task.model.geom_bodyid[geom])).name,
                             "designated_support": geom in task.supports, "normal_force_n": float(force[0])})
        audit["selected_replays"].append({"family": run["family"], "seed": run["seed"],
                                          "final_base_position_m": task.data.qpos[:3].tolist(),
                                          "final_base_velocity": task.data.qvel[:6].tolist(),
                                          "final_floor_contacts": contacts, "replay_passed": True})
    output = args.report.with_name("audit.json")
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2))
    print(output)


if __name__ == "__main__":
    main()
