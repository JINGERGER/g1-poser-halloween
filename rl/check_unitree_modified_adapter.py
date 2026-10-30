import mujoco
import numpy as np

from check_standing_endpoint import standing_pose
from check_unitree_standing_baseline import REFERENCE, load_reference
from reference_policy_baseline import ROOT
from unitree_modified_adapter import LegAdapter


def main():
    config, functions = load_reference()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"))
    reference = mujoco.MjModel.from_xml_path(str(REFERENCE / "resources/robots/g1_description/scene.xml"))
    adapter = LegAdapter(model, reference)
    untouched = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_damping", "dof_frictionloss",
                 "dof_armature", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
                 "geom_solimp", "geom_contype", "geom_conaffinity", "jnt_range", "jnt_limited",
                 "jnt_actfrcrange", "jnt_actfrclimited", "actuator_gear", "actuator_forcelimited", "actuator_forcerange")
    before = {name: getattr(model, name).copy() for name in untouched}
    upper = np.setdiff1d(np.arange(model.nu), adapter.actuators)
    upper_fields = ("actuator_gainprm", "actuator_biasprm", "actuator_gaintype", "actuator_biastype", "actuator_ctrllimited", "actuator_ctrlrange")
    upper_before = {name: getattr(model, name)[upper].copy() for name in upper_fields}
    adapter.torque_interface(model)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(model, name), original)
    for name, original in upper_before.items():
        np.testing.assert_array_equal(getattr(model, name)[upper], original)
    data = standing_pose(model, .6)
    all_joints = model.actuator_trnid[:, 0]
    data.ctrl[:] = data.qpos[model.jnt_qposadr[all_joints]]
    request = np.array([1000., -1000.] * 6)
    data.ctrl[adapter.actuators] = request
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[adapter.qvel],
                               np.clip(request, model.jnt_actfrcrange[adapter.joints, 0], model.jnt_actfrcrange[adapter.joints, 1]),
                               atol=1e-10, rtol=0)
    data.ctrl[adapter.actuators] = 12.5
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[adapter.qvel], 12.5, atol=1e-10, rtol=0)
    reference.actuator_trnid[[0, 11]] = reference.actuator_trnid[[11, 0]]
    shuffled = LegAdapter(model, reference)
    assert shuffled.actuators[0] == adapter.actuators[11] and shuffled.actuators[11] == adapter.actuators[0]
    data.qvel[6:] = np.arange(model.nv - 6) * .01
    observation = shuffled.observation(data, np.zeros(12, dtype=np.float32), 10, np.zeros(3), config,
                                       functions["get_gravity_orientation"])
    expected = np.array([data.qvel[model.jnt_dofadr[model.joint(name).id]] for name in shuffled.names])
    np.testing.assert_allclose(observation[21:33], expected * config["dof_vel_scale"], atol=1e-8, rtol=0)
    model.actuator_gear[0, 0] = 2.
    try:
        LegAdapter(model, reference)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported transmission was accepted")
    print("Modified adapter: name mapping, raw torque units, retained saturation, unchanged upper body/physics and invalid gear rejection passed")


if __name__ == "__main__":
    main()
