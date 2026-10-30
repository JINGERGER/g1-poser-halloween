import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from support_handoff import leg_offset, pulse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    assert pulse(0) == 0 and pulse(0.5) == 0 and pulse(1.25) == 1 and pulse(1.75) == 1 and pulse(2.5) == 0 and pulse(3) == 0
    for elapsed in np.linspace(0, 3, 301):
        assert 0 <= pulse(elapsed) <= 1
    for boundary in (0.5, 1.25, 1.75, 2.5):
        assert abs((pulse(boundary + 1e-5) - pulse(boundary - 1e-5)) / 2e-5) < 1e-6
    print("PASS: bounded pulse, exact zero/plateau/landing and smooth boundaries")
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and len(report["runs"]) == 6
        for index in (0, 3):
            group = report["runs"][index:index + 3]
            assert [row["branch"] for row in group] == ["continue", "freeze", "right_foot_probe"]
            assert len({row["anchor"]["paired_state_sha256"] for row in group}) == 1
            for row in group:
                assert row["anchor"]["ready"] and row["max_native_state_error"] < 2e-4
                model = mujoco.MjModel.from_xml_path(row["model"])
                anchor = np.array(row["anchor"]["qpos"])
                original = anchor.copy()
                offset, error = leg_offset(model, anchor)
                np.testing.assert_array_equal(anchor, original)
                assert error < 0.001 and max(abs(offset)) <= 0.15000001
                for actuator, joint in enumerate(model.actuator_trnid[:, 0]):
                    name = model.joint(int(joint)).name
                    if not (name.startswith("right_") and any(part in name for part in ("hip", "knee", "ankle"))):
                        assert offset[actuator] == 0
                anchor[:2] += [0.3, -0.2]
                translated, translated_error = leg_offset(model, anchor)
                np.testing.assert_allclose(translated, offset, atol=1e-9)
                assert translated_error < 0.001
                assert len(row["trace"]) == 650 and row["summary"]["completed_3s"]
        print("PASS: paired physical/observation states, exact model replay, isolated bounded IK, input immutability and translation invariance")


if __name__ == "__main__":
    main()
