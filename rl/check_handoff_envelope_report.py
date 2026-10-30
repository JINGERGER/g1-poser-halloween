import argparse
import json
from pathlib import Path

import numpy as np

from artifact_utils import digest
from check_handoff_envelope import specifications
from reference_policy_baseline import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    assert report["specifications"] == specifications()
    assert len(report["cases"]) == len(report["specifications"]) == 44
    for name, expected in report["sources"].items():
        assert digest(ROOT / "rl" / name) == expected
    nominal = np.array(report["cases"][0]["state"]["qpos"])
    for case, specification in zip(report["cases"], specifications(), strict=True):
        assert case["name"] == specification["name"]
        pose = np.array(case["state"]["qpos"])
        np.testing.assert_allclose(np.linalg.norm(pose[3:7]), 1., atol=1e-7)
        np.testing.assert_array_equal(pose[7:], nominal[7:])
        np.testing.assert_allclose(pose[2] - nominal[2], case["floor_alignment_m"] + specification["rise_m"], atol=1e-12)
        np.testing.assert_array_equal(case["state"]["qvel"][:6], specification["velocity_delta"])
        if not case["executed"]:
            assert not case["success"] and not all(case["initial_gates"].values())
            continue
        assert all(case["initial_gates"].values()) and case["replay_passed"]
        trajectory = ROOT / case["trajectory"]
        assert digest(trajectory) == case["trajectory_sha256"]
        saved = np.load(trajectory)
        np.testing.assert_array_equal(saved["qpos"][0], pose)
        np.testing.assert_array_equal(saved["qvel"][0], case["state"]["qvel"])
        assert np.all(saved["phase"] == 1)
        assert len(saved["qpos"]) == len(saved["ctrl"]) + 1
        loads = saved["samples"][:, 4:8]
        np.testing.assert_allclose(case["sampled_peak_loads_n"], loads.max(axis=0))
        np.testing.assert_allclose(case["sampled_normal_impulses_ns"], loads.sum(axis=0) * .002)
        np.testing.assert_allclose(case["first_200ms_peak_hand_load_n"], loads[:100, :2].max())
        assert case["success"] == all(case["result"]["gates"].values())
    np.testing.assert_allclose(report["cases"][0]["result"]["forward_m"], .820855278360121, atol=1e-10, rtol=0)
    assert report["summary"]["successful"] == sum(case["success"] for case in report["cases"])
    print("PASS: fixed budget, source/trajectory hashes, perturbations, resets, load metrics, acceptance and baseline")


if __name__ == "__main__":
    main()
