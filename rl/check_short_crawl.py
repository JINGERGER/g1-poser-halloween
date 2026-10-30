import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from short_crawl import phase_clock, trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    frames = np.zeros((80, 30))
    frames[:, 7:] = np.sin(np.arange(80)[:, None] / 80 * 2 * np.pi) * np.linspace(0.1, 1, 23)
    curve = trajectory(frames)
    for index in range(80):
        np.testing.assert_allclose(curve(index / 100), frames[index, 7:], atol=1e-12)
    np.testing.assert_allclose(curve(0), curve(1), atol=1e-12)
    np.testing.assert_allclose(curve(1 - 1e-8, 1), curve(1e-8, 1), atol=1e-4)
    for phase in np.linspace(0.001, 0.999, 100):
        derivative = (curve(phase + 1e-7) - curve(phase - 1e-7)) / 2e-7
        np.testing.assert_allclose(curve(phase, 1), derivative, atol=1e-5)
        assert np.all(curve(phase) <= frames[:, 7:].max(axis=0) + 1e-12)
        assert np.all(curve(phase) >= frames[:, 7:].min(axis=0) - 1e-12)
    for elapsed in (0.1, 0.6, 1.0, 1.4, 2.0, 5.0):
        phase, rate = phase_clock(elapsed, 4)
        derivative = (phase_clock(elapsed + 1e-6, 4)[0] - phase_clock(elapsed - 1e-6, 4)[0]) / 2e-6
        assert abs(derivative - rate) < 1e-8
        assert 0 <= rate <= 0.25 and phase >= 0
    assert phase_clock(0.5, 4) == (0, 0)
    for invalid in (np.zeros((79, 30)), np.full((80, 30), np.nan)):
        try:
            trajectory(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid trajectory accepted")
    print("PASS: exact knots, periodic C1 seam, bounded interpolation, analytic velocity, smooth phase ramp and invalid inputs")
    if args.report:
        report = json.loads(args.report.read_text())
        assert report["status"] == "completed" and len(report["runs"]) == 12
        for result in report["runs"]:
            model = mujoco.MjModel.from_xml_path(result["model"])
            data = mujoco.MjData(model)
            saved = np.load(result["trajectory"], allow_pickle=False)
            data.qpos[:] = saved["qpos"][0]
            data.qvel[:] = saved["qvel"][0]
            mujoco.mj_forward(model, data)
            for step, control in enumerate(saved["ctrl"]):
                data.ctrl[:] = control
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                for substep in range(round(0.02 / model.opt.timestep)):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.qpos, saved["qpos"][step + 1], atol=1e-8, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][step + 1], atol=1e-8, rtol=0)
            assert result["passed"] == all(result["gates"].values())
        print("PASS: all 12 saved controls reproduce full position/velocity traces without resets or external assistance")


if __name__ == "__main__":
    main()
