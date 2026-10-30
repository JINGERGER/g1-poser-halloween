import json
from datetime import datetime

import mujoco
import numpy as np
import torch

from artifact_utils import digest
from check_standing_endpoint import standing_pose
from reference_policy_baseline import ROOT, materialize, run
from transition_execution import TransitionExecution, physical_state


def main():
    folder, constants = materialize()
    torch.set_num_threads(1)
    policy = torch.jit.load(str(folder / "policy_crawl.pt"), map_location="cpu").eval()
    model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    bundle = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
    initial = np.array(json.loads(bundle.read_text())["initial_qpos"])
    reference = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    output = ROOT / "output/rl/transition_execution" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    trial = run(folder, constants, policy, 10, .5, output / "nominal.npz", model_path=model_path,
                initial_qpos=initial, contact_time_constant=.005, physics_timestep=.002,
                ankle_min_target=-.2, foot_clearance_boost=.08)
    assert trial["trajectory_sha256"] == "686214684b7045787b53d8c7456deea3ba8c5503e28594d001b501c0b3274121"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.geom_solref[:, 0] = .005
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    execution = TransitionExecution(model, data, reference, constants, policy)
    model_identity, data_identity = id(model), id(data)
    initial_handoff = execution.handoff()
    with np.load(trial["trajectory"]) as saved:
        for index in range(500):
            execution.crawl_step()
            np.testing.assert_array_equal(data.ctrl, saved["ctrl"][index])
            np.testing.assert_array_equal(data.qpos, saved["qpos"][index + 1])
            np.testing.assert_array_equal(data.qvel, saved["qvel"][index + 1])
    assert id(execution.model) == model_identity and id(execution.data) == data_identity
    try:
        execution.handoff()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Repeated handoff reset controller history")
    try:
        execution.transition_step(data.ctrl)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Transition commands accepted after handoff")
    fixture_model = mujoco.MjModel.from_xml_path(str(model_path))
    fixture_model.geom_solref[:, 0] = .005
    fixture = standing_pose(fixture_model, .6)
    targets = fixture.qpos[fixture_model.jnt_qposadr[fixture_model.actuator_trnid[:, 0]]].copy()
    fixture_execution = TransitionExecution(fixture_model, fixture, reference, constants, policy)
    state_before_invalid_target = physical_state(fixture_model, fixture)
    for invalid in (np.zeros(12), np.full(23, np.nan), np.full(23, 100.)):
        try:
            fixture_execution.transition_step(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid position target accepted")
        np.testing.assert_array_equal(physical_state(fixture_model, fixture), state_before_invalid_target)
    for unused in range(100):
        fixture_execution.transition_step(targets)
    assert fixture.time > 1.99 and np.linalg.norm(fixture.qvel) > 0.
    before = physical_state(fixture_model, fixture)
    standing_handoff = fixture_execution.handoff()
    np.testing.assert_array_equal(physical_state(fixture_model, fixture), before)
    bad_model = mujoco.MjModel.from_xml_path(str(model_path))
    bad_model.geom_solref[:, 0] = .005
    bad_model.actuator_biasprm[0] = 0.
    try:
        TransitionExecution(bad_model, mujoco.MjData(bad_model), reference, constants, policy)
    except ValueError:
        pass
    else:
        raise AssertionError("Torque motor incorrectly accepted as a position servo")
    result = {"sources": {name: digest(ROOT / "rl" / name) for name in
                          ("crawl_control.py", "reference_policy_baseline.py", "transition_execution.py", "check_transition_execution.py")},
              "nominal_sha256": trial["trajectory_sha256"], "nominal_10s_bitwise_equivalent": True,
              "same_model_and_data_objects": True, "repeated_handoff_rejected": True,
              "posthandoff_transition_commands_rejected": True, "torque_interface_rejected": True,
              "invalid_transition_targets_rejected_without_state_change": True,
              "nominal_handoff": initial_handoff, "standing_state_contract_only": standing_handoff,
              "limitations": "Execution-contract tests only. Standing fixture stops at handoff before any crawl step; "
                             "it is NOT a successful standing-to-crawl demonstration or safe entry test. No training."}
    path = output / "report.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"nominal_equivalent": True, "standing_fixture_state_preserved": True,
                      "standing_fixture_target_jump_rad": standing_handoff["max_target_jump_rad"],
                      "standing_fixture_torque_jump_nm": standing_handoff["max_instantaneous_generalized_torque_jump_nm"]}))
    print(path)


if __name__ == "__main__":
    main()
