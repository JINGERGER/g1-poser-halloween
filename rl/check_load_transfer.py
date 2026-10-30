import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from load_transfer import bounded_vector, posture, smooth, unloaded
from prepare import SITES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    np.testing.assert_array_equal(bounded_vector([0, 0], 0.08), [0, 0])
    np.testing.assert_allclose(bounded_vector([3, 4], 0.1), [0.06, 0.08])
    assert smooth(-1) == 0 and smooth(2) == 1
    assert unloaded([6, 6, 6, 5])
    assert not unloaded([6, 6, 6, 5.001]) and not unloaded([5, 6, 6, 0])
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and len(report["runs"]) == 4
    for index in (0, 2):
        freeze, transfer = report["runs"][index:index + 2]
        assert freeze["anchor_hash"] == transfer["anchor_hash"]
        np.testing.assert_array_equal(freeze["initial_physics_state"], transfer["initial_physics_state"])
        np.testing.assert_array_equal(freeze["initial_ctrl"], transfer["initial_ctrl"])
    for run in report["runs"]:
        model = mujoco.MjModel.from_xml_path(run["model"])
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, np.array(run["initial_physics_state"]), mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = run["initial_ctrl"]
        mujoco.mj_forward(model, data)
        anchor = data.qpos.copy()
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = anchor
        sites = np.array([data.site(name).xpos.copy() for name in SITES])
        offset, residual = posture(model, scratch, anchor, sites, np.array([0.005, 0.005]), 0)
        assert residual < 0.005 and np.max(np.abs(offset)) <= 0.25000001
        np.testing.assert_array_equal(data.qpos, anchor)
        saved = np.load(run["trajectory"], allow_pickle=False)
        for step, control in enumerate(saved["ctrl"]):
            assert np.max(np.abs(control - data.ctrl)) <= 0.010000001
            data.ctrl[:] = control
            assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
            for substep in range(round(0.02 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, saved["qpos"][step + 1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(data.qvel, saved["qvel"][step + 1], atol=1e-9, rtol=0)
        if run["unload_trigger_s"] is None:
            assert all(row["requested_site_lift_m"] == 0 for row in run["trace"])
        assert all(np.linalg.norm(row["commanded_root_shift"]) <= 0.08000001 for row in run["trace"])
    print("PASS: load gate, paired anchors, bounded shift/IK/control rate, scratch isolation, no ungated lift, and exact physical position/velocity replay for all four runs")


if __name__ == "__main__":
    main()
