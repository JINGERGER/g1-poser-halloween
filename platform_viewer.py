from __future__ import annotations

from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path
import os
import subprocess
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


_CAMERA_FIELDS = ("type", "fixedcamid", "trackbodyid", "lookat", "distance", "azimuth", "elevation")
_EDITABLE_FIELDS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "qfrc_applied", "xfrc_applied")
_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
_FRAME_INTERVAL = 1.0 / 60.0
_TIMEOUT = 30.0


def _camera_values(camera):
    return {name: np.array(getattr(camera, name), copy=True) for name in _CAMERA_FIELDS}


def _apply_camera(camera, values):
    for name, value in values.items():
        if name == "lookat":
            camera.lookat[:] = value
        else:
            setattr(camera, name, value.item())


class ProcessViewer:
    def __init__(self, model, data, xml_path):
        self.model = model
        self.data = data
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self._camera = _camera_values(self.cam)
        self._state = np.empty(mujoco.mj_stateSize(model, _STATE))
        self._last_sync = 0.0
        self._running = False
        self._process = None
        self._connection, child_connection = Pipe()
        executable = Path(sys.executable).with_name("mjpython")
        environment = os.environ.copy()
        environment["G1_VIEWER_FD"] = str(child_connection.fileno())
        environment["G1_VIEWER_XML"] = str(Path(xml_path).resolve())
        try:
            self._process = subprocess.Popen(
                [str(executable), str(Path(__file__).resolve())],
                env=environment,
                pass_fds=(child_connection.fileno(),),
            )
            child_connection.close()
            if not self._connection.poll(_TIMEOUT):
                raise RuntimeError("MuJoCo viewer did not start within 30 seconds")
            self._camera = self._connection.recv()
            _apply_camera(self.cam, self._camera)
            self._running = True
        except BaseException:
            child_connection.close()
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def is_running(self):
        if self._process.poll() not in (None, 0):
            raise RuntimeError(f"MuJoCo viewer exited with code {self._process.returncode}")
        return self._running and self._process.poll() is None

    def sync(self):
        now = time.monotonic()
        if now - self._last_sync < _FRAME_INTERVAL or not self.is_running():
            return
        camera = _camera_values(self.cam)
        changes = {
            name: value for name, value in camera.items()
            if not np.array_equal(value, self._camera[name])
        }
        mujoco.mj_getState(self.model, self.data, self._state, _STATE)
        try:
            self._connection.send((self._state, changes))
            if not self._connection.poll(_TIMEOUT):
                raise RuntimeError("MuJoCo viewer stopped responding")
            response = self._connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self._running = False
            if self._process.wait(timeout=5) != 0:
                raise RuntimeError("MuJoCo viewer crashed; see its error output") from None
            return
        if response is None:
            self._running = False
            return
        self._camera, edits = response
        _apply_camera(self.cam, self._camera)
        for name, value in edits.items():
            getattr(self.data, name)[:] = value
        self._last_sync = now

    def close(self):
        self._running = False
        self._connection.close()
        if self._process is not None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()


def launch_viewer(model, data, xml_path):
    if sys.platform == "darwin":
        return ProcessViewer(model, data, xml_path)
    return mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=True,
    )


def _run_viewer():
    connection = Connection(int(os.environ.pop("G1_VIEWER_FD")))
    model = mujoco.MjModel.from_xml_path(os.environ.pop("G1_VIEWER_XML"))
    data = mujoco.MjData(model)
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            connection.send(_camera_values(viewer.cam))
            while viewer.is_running():
                if not connection.poll(_FRAME_INTERVAL):
                    continue
                try:
                    state, camera = connection.recv()
                except EOFError:
                    break
                with viewer.lock():
                    mujoco.mj_setState(model, data, state, _STATE)
                    mujoco.mj_forward(model, data)
                    _apply_camera(viewer.cam, camera)
                    before = {name: getattr(data, name).copy() for name in _EDITABLE_FIELDS}
                viewer.sync()
                with viewer.lock():
                    edits = {
                        name: getattr(data, name).copy() for name in _EDITABLE_FIELDS
                        if not np.array_equal(before[name], getattr(data, name))
                    }
                    camera = _camera_values(viewer.cam)
                connection.send((camera, edits))
            connection.send(None)
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        connection.close()


if __name__ == "__main__":
    _run_viewer()
