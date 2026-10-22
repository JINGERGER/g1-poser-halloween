import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from reference_policy_baseline import materialize
from artifact_utils import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    assert digest(report["policy"]) == report["policy_sha256"]
    assert digest(report["asset_manifest"]) == report["asset_manifest_sha256"]
    folder, unused_constants = materialize()
    manifest = json.loads((folder / "reference_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest(folder / name) == expected
    for trial in report["runs"]:
        assert digest(trial["trajectory"]) == trial["trajectory_sha256"]
        saved = np.load(trial["trajectory"], allow_pickle=False)
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        model.actuator_gainprm[:, 0] *= 1.4
        model.actuator_biasprm[:, 1:3] *= 1.4
        assert model.na == 0
        data = mujoco.MjData(model)
        data.qpos[:] = saved["qpos"][0]
        data.qvel[:] = saved["qvel"][0]
        mujoco.mj_forward(model, data)
        assert len(saved["ctrl"]) == len(trial["trace"]) == len(saved["qpos"]) - 1
        for index, control in enumerate(saved["ctrl"]):
            data.ctrl[:] = control
            steps = round((trial["trace"][index]["time_s"] - data.time) / model.opt.timestep)
            assert 1 <= steps <= trial["control_substeps"]
            for unused in range(steps):
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-10, rtol=0)
        print(f"replay passed: {trial['trajectory']}", flush=True)
    print("Reference physics replay verified; this does not certify locomotion or target-model compatibility.")


if __name__ == "__main__":
    main()
