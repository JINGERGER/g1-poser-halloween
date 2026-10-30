import json

import mujoco
from mujoco import batch_env
import numpy as np

from crawl_task import ROOT


def main():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body><joint name="hinge" type="hinge"/>
        <geom type="sphere" size="0.1" mass="1" pos="0.2 0 0"/>
      </body></worldbody>
      <actuator><position joint="hinge" kp="40" kv="3"/></actuator>
      <sensor><jointactuatorfrc joint="hinge" name="torque"/></sensor>
    </mujoco>''')
    initial = np.zeros((1, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)))
    command = np.full((1, 10, 1), 0.7)
    with batch_env.BatchEnvPool(model, nbatch=1, nthread=0) as pool:
        state, sensors = pool.step(initial, nstep=10, control=command, return_sensor=True, post_step_forward_sensor=True)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = 0.7
    mujoco.mj_forward(model, data)
    correct = float(data.sensor("torque").data[0])
    analytical = float(40 * (0.7 - data.qpos[0]) - 3 * data.qvel[0])
    assert np.isclose(correct, analytical)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    zero_control = float(data.sensor("torque").data[0])
    native = mujoco.MjData(model)
    native.ctrl[:] = 0.7
    for step in range(10):
        mujoco.mj_step(model, native)
    native_state = np.zeros(initial.shape[1])
    mujoco.mj_getState(model, native, native_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    assert np.allclose(state[0], native_state, atol=1e-10)
    report = {
        "mujoco_version": mujoco.__version__,
        "batch_refreshed_torque": float(sensors[0, 0]),
        "correct_final_state_torque": correct,
        "zero_control_torque": zero_control,
        "refresh_matches_zero_control": bool(np.isclose(sensors[0, 0], zero_control)),
        "refresh_matches_command": bool(np.isclose(sensors[0, 0], correct)),
        "batch_dynamics_match_native": True,
    }
    output = ROOT / "output/rl/force_refresh_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
