import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from artifact_utils import digest
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask
from verify_crawl_candidate import reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    case = next(case for case in report["cases"] if case.get("label") == "nominal")
    assert case["success"] and case["replay_passed"] and report["full_transition_validated"]
    trajectory = ROOT / case["trajectory"]
    assert digest(trajectory) == case["trajectory_sha256"]
    task = StandTransitionTask()
    for name in ("policy", "bundle", "model"):
        assert digest(getattr(task, name + "_path")) == report["config"][name + "_sha256"]
    for path, expected in report.get("model_assets", {}).items():
        assert digest(task.model_path.parent / path) == expected
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    task.reset(state)
    pure_start = case["handoff"]["pure_crawl_start_index"]
    with np.load(trajectory, allow_pickle=False) as saved:
        positions, velocities, controls = saved["qpos"], saved["qvel"], saved["ctrl"]
        rows, phases, penetrations = saved["samples"], saved["phase"], saved["penetrations"]
        assert len(positions) == len(controls) + 1 and len(rows) == len(controls)
        np.testing.assert_array_equal(positions[0], state["qpos"])
        np.testing.assert_array_equal(velocities[0], state["qvel"])
        first_crawl = int(np.flatnonzero(phases)[0])
        assert np.all(phases[:first_crawl] == 0) and np.all(phases[first_crawl:] == 1)
        assert pure_start - first_crawl == round(report["config"]["blend_duration_s"] / .002)
        assert len(controls) - pure_start == 1000
        assert np.all(rows[first_crawl - 1, 4:6] > 5.)
        for index, control in enumerate(controls):
            if index == first_crawl:
                task.phase = "crawl"
                task.execution.handoff()
                task.execution.pending_control = False
            elif index > first_crawl and (index - first_crawl) % 10 == 0:
                task.execution.controller.update(task.data)
            if index >= pure_start:
                np.testing.assert_allclose(task.data.ctrl, control, atol=1e-10, rtol=0.)
            assert not task.data.qfrc_applied.any() and not task.data.xfrc_applied.any()
            task.data.ctrl[:] = control
            mujoco.mj_step(task.model, task.data)
            mujoco.mj_forward(task.model, task.data)
            task._observe_step(task.model, task.data)
            np.testing.assert_allclose(task.data.qpos, positions[index + 1], atol=1e-10, rtol=0.)
            np.testing.assert_allclose(task.data.qvel, velocities[index + 1], atol=1e-10, rtol=0.)
            np.testing.assert_allclose(task.samples[-1], rows[index], atol=1e-9, rtol=0.)
            np.testing.assert_allclose(task.penetrations[-1], penetrations[index], atol=1e-9, rtol=0.)
        events, cycles, consecutive = reconstruct(np.asarray(task.samples[pure_start:]), np.asarray(task.support_positions[pure_start:]))
        forward = float(task.data.qpos[0] - positions[pure_start, 0])
        assert consecutive and forward >= .06 and not task.failure
    print(json.dumps({"passed": True, "duration_s": float(task.data.time), "forward_m": forward,
                      "cycles": len(cycles), "pure_policy_steps": 1000, "full_transition_validated": True,
                      "checks": ["true standing", "no state reset", "physical gates every 2ms", "original frozen policy controls after bridge", "qualified crawl cycles"]}))


if __name__ == "__main__":
    main()
