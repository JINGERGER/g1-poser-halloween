from types import SimpleNamespace

import mujoco
import numpy as np

from check_transition_route_envelope import static_capacity


def make_task(pitch, height):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>
    <body pos="0 0 {height}"><freejoint/><geom name="foot" type="box" size=".09 .03 .008" euler="0 {pitch} 0" mass="1"/></body>
    <body pos="0 0 2"><joint name="hinge" actuatorfrcrange="-10 10"/><geom name="hand" type="sphere" size=".02"/></body>
    </worldbody><actuator><motor joint="hinge"/></actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    task = SimpleNamespace(model=model, floor=model.geom("floor").id,
                           supports=[model.geom("hand").id, model.geom("hand").id,
                                     model.geom("foot").id, model.geom("foot").id],
                           joints=np.array([model.joint("hinge").id]))
    return task, data


def main():
    task, data = make_task(0., .00795)
    assert static_capacity(task, data, True, active_only=True)["support_point_counts"] == [0, 0, 4, 4]
    task, data = make_task(30., .09 * np.sin(np.pi / 6) + .008 * np.cos(np.pi / 6) - .00005)
    assert static_capacity(task, data, True, active_only=True)["support_point_counts"] == [0, 0, 2, 2]
    task, data = make_task(0., .02)
    result = static_capacity(task, data, True, active_only=True)
    assert result["support_point_counts"] == [0, 0, 0, 0] and not result["feasible"]
    print("PASS: airborne spheres excluded; flat feet four corners; tilted feet two corners; airborne feet excluded")


if __name__ == "__main__":
    main()
