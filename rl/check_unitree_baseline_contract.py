import os
import runpy
import sys
import types
from unittest.mock import patch

import mujoco.viewer
import numpy as np
import torch

from check_unitree_standing_baseline import REFERENCE, load_reference, rollout


class HeadlessViewer:
    def __init__(self, model, data, steps):
        self.data = data
        self.steps = steps
        self.positions = []
        self.controls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def is_running(self):
        return len(self.positions) < self.steps

    def sync(self):
        self.positions.append(self.data.qpos.copy())
        self.controls.append(self.data.ctrl.copy())


def main():
    config, functions = load_reference()
    torch.set_num_threads(1)
    steps = 500
    unused, saved, stopped = rollout(config, functions, "policy", np.array(config["cmd_init"], dtype=np.float32), steps)
    assert stopped is None
    viewers = []

    def launch(model, data):
        viewer = HeadlessViewer(model, data, steps)
        viewers.append(viewer)
        return viewer

    stub = types.ModuleType("legged_gym")
    stub.LEGGED_GYM_ROOT_DIR = str(REFERENCE)
    cwd = os.getcwd()
    try:
        os.chdir(REFERENCE)
        with patch.dict(sys.modules, {"legged_gym": stub}), patch.object(sys, "argv", ["deploy_mujoco.py", "g1.yaml"]), \
                patch("mujoco.viewer.launch_passive", launch), patch("time.sleep", lambda unused: None):
            runpy.run_path(str(REFERENCE / "deploy/deploy_mujoco/deploy_mujoco.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    assert len(viewers) == 1 and len(viewers[0].positions) == steps
    np.testing.assert_allclose(viewers[0].positions, saved["qpos"][1:], atol=1e-10, rtol=0)
    np.testing.assert_allclose(viewers[0].controls, saved["ctrl"], atol=1e-10, rtol=0)
    print("Official unmodified deployment vs headless adapter: 500 physics steps, qpos/ctrl match within 1e-10")


if __name__ == "__main__":
    main()
