import json

import mujoco
import numpy as np
import torch

from reference_policy_baseline import ROOT, materialize
from transition_execution import TransitionExecution
from verify_crawl_candidate import reconstruct


class UnsafeState(Exception):
    pass


class NearHandoffTask:
    def __init__(self):
        folder, self.constants = materialize()
        self.model_path = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
        self.bundle_path = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
        self.policy_path = folder / "policy_crawl.pt"
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.geom_solref[:, 0] = .005
        self.reference = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        self.policy = torch.jit.load(str(self.policy_path), map_location="cpu").eval()
        self.nominal = np.array(json.loads(self.bundle_path.read_text())["initial_qpos"])
        self.joints = self.model.actuator_trnid[:, 0]
        self.addresses = self.model.jnt_qposadr[self.joints]
        self.target = self.nominal[self.addresses]
        self.native_gain = self.model.actuator_gainprm.copy()
        self.native_bias = self.model.actuator_biasprm.copy()
        self.supports = [self.model.geom(name).id for name in ("left_hand_collision", "right_hand_collision", "left_foot", "right_foot")]
        self.floor = self.model.geom("floor").id
        self.phase = "transition"

    def measure(self):
        model, data = self.model, self.data
        loads, other, penetration, self_penetration = np.zeros(4), 0., 0., 0.
        for index, contact in enumerate(data.contact):
            penetration = max(penetration, -contact.dist)
            if self.floor not in (contact.geom1, contact.geom2):
                self_penetration = max(self_penetration, -contact.dist)
                continue
            geom = contact.geom2 if contact.geom1 == self.floor else contact.geom1
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            force = max(0., float(wrench[0]))
            if geom in self.supports:
                loads[self.supports.index(geom)] += force
            else:
                other += force
        gaps = [float(mujoco.mj_geomDistance(model, data, geom, self.floor, 2., None)) for geom in self.supports]
        angles = data.qpos[self.addresses]
        excess = max(0., float(np.max(model.jnt_range[self.joints, 0] - angles)),
                     float(np.max(angles - model.jnt_range[self.joints, 1])))
        limits = np.max(np.abs(model.jnt_actfrcrange[self.joints]), axis=1)
        ratio = float(np.max(np.abs(data.qfrc_actuator[model.jnt_dofadr[self.joints]]) / limits))
        row = [data.time, *data.qpos[:3], *loads, *gaps, other, ratio, excess]
        gates = {"penetration": penetration <= .005, "self_penetration": self_penetration <= .001,
                 "non_support_load": other <= 1., "torque": ratio < .95, "joint_limits": excess <= .001,
                 "height": .16 <= data.qpos[2] <= .65}
        return row, gates, [penetration, self_penetration]

    def _observe_step(self, model, data):
        row, gates, penetration = self.measure()
        self.samples.append(row)
        self.support_positions.append(data.geom_xpos[self.supports].copy())
        self.penetrations.append(penetration)
        self.positions.append(data.qpos.copy())
        self.velocities.append(data.qvel.copy())
        self.controls.append(data.ctrl.copy())
        self.phases.append(int(self.phase == "crawl"))
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            self.failure = failed
            raise UnsafeState(",".join(failed))

    def reset(self, state):
        self.model.actuator_gainprm[:] = self.native_gain
        self.model.actuator_biasprm[:] = self.native_bias
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.phase, self.tick, self.failure = "transition", 0, []
        self.previous = np.zeros(23)
        self.samples, self.support_positions, self.penetrations = [], [], []
        self.positions, self.velocities = [self.data.qpos.copy()], [self.data.qvel.copy()]
        self.controls, self.phases = [], []
        self.execution = TransitionExecution(self.model, self.data, self.reference, self.constants, self.policy, self._observe_step)
        row, gates, unused = self.measure()
        if not all(gates.values()):
            raise ValueError("Initial state fails physical screening")
        self.initial_row = row
        return self.observation()

    def observation(self):
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        row, unused, unused_penetration = self.measure()
        observation = np.r_[rotation.reshape(3, 3).T @ [0., 0., -1.],
                            self.data.qvel[:6], self.data.qpos[self.addresses] - self.target,
                            .1 * self.data.qvel[self.model.jnt_dofadr[self.joints]],
                            np.clip(row[8:12], -.01, .1), np.array(row[4:8]) / 312.,
                            self.previous, self.tick / 20., self.data.qpos[2] - self.nominal[2]].astype(np.float32)
        assert observation.shape == (88,) and np.isfinite(observation).all()
        return observation

    def finish(self):
        start = self.data.qpos[:3].copy()
        sample_start = len(self.samples)
        self.phase = "crawl"
        self.handoff = self.execution.handoff()
        self.crawl_gain, self.crawl_bias = self.model.actuator_gainprm.copy(), self.model.actuator_biasprm.copy()
        try:
            for unused in range(100):
                self.execution.crawl_step()
        except UnsafeState:
            pass
        samples = np.array(self.samples[sample_start:])
        positions = np.array(self.support_positions[sample_start:])
        events, cycles, consecutive = reconstruct(samples, positions)
        forward = float(self.data.qpos[0] - start[0])
        gates = {"physical_substeps": not self.failure, "completed_2s": len(samples) == 1000,
                 "consecutive_cycles": bool(consecutive), "forward_6cm": forward >= .06}
        success = bool(all(gates.values()))
        result = {"success": success, "failure": self.failure, "cycle_count": len(cycles),
                  "gates": gates,
                  "qualified_events": [sum(event["qualified"] for event in limb) for limb in events],
                  "forward_m": forward, "crawl_steps": len(samples),
                  "target_jump_rad": self.handoff["max_target_jump_rad"],
                  "torque_jump_nm": self.handoff["max_instantaneous_generalized_torque_jump_nm"]}
        return result

    def step(self, raw_action):
        raw_action = np.asarray(raw_action)
        if raw_action.shape != (23,) or not np.isfinite(raw_action).all():
            raise ValueError("Expected 23 finite raw actions")
        action = np.tanh(raw_action)
        target = np.clip(self.target + .12 * action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        difference = float(np.mean((action - self.previous) ** 2))
        self.previous = action
        try:
            self.execution.transition_step(target)
        except UnsafeState:
            return self.observation(), -5., True, {"success": False, "failure": self.failure, "phase": "transition"}
        self.tick += 1
        error = float(np.mean((self.data.qpos[self.addresses] - self.target) ** 2))
        reward = .02 * (np.exp(-error / .05 ** 2) - .1 * difference - .02 * np.mean(action ** 2))
        if self.tick == 20:
            result = self.finish()
            reward += 20. if result["success"] else -5.
            return self.observation(), float(reward), True, result
        return self.observation(), float(reward), False, {}

    def hand_candidate(self, rng, seed):
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = self.nominal
        mujoco.mj_forward(self.model, probe)
        feet_before = probe.geom_xpos[self.supports[2:]].copy()
        sides = ("left",) if seed % 3 == 0 else ("right",) if seed % 3 == 1 else ("left", "right")
        requested = {}
        for side in sides:
            geom = self.model.geom(f"{side}_hand_collision").id
            target = probe.geom_xpos[geom].copy()
            gap = float(rng.uniform(.01, .03))
            target[2] = self.model.geom_size[geom, 0] + gap
            joints = [self.model.joint(f"{side}_{suffix}_joint").id
                      for suffix in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")]
            columns = self.model.jnt_dofadr[joints]
            addresses = self.model.jnt_qposadr[joints]
            for unused in range(30):
                error = target - probe.geom_xpos[geom]
                if np.linalg.norm(error) <= .0005:
                    break
                jacobian = np.zeros((3, self.model.nv))
                mujoco.mj_jacGeom(self.model, probe, jacobian, None, geom)
                arm = jacobian[:, columns]
                delta = arm.T @ np.linalg.solve(arm @ arm.T + 1e-6 * np.eye(3), error)
                probe.qpos[addresses] = np.clip(probe.qpos[addresses] + np.clip(delta, -.08, .08),
                                                self.model.jnt_range[joints, 0], self.model.jnt_range[joints, 1])
                mujoco.mj_forward(self.model, probe)
            if np.linalg.norm(target - probe.geom_xpos[geom]) > .0005:
                return None
            requested[side] = gap
        np.testing.assert_array_equal(probe.qpos[:7], self.nominal[:7])
        np.testing.assert_allclose(probe.geom_xpos[self.supports[2:]], feet_before, atol=1e-12, rtol=0)
        actual = {side: float(mujoco.mj_geomDistance(self.model, probe, self.model.geom(f"{side}_hand_collision").id,
                                                  self.floor, 2., None)) for side in sides}
        if any(abs(actual[side] - gap) > .0005 for side, gap in requested.items()):
            return None
        velocity = np.zeros(self.model.nv)
        velocity[:2] = rng.uniform(-.05, .05, 2)
        return {"qpos": probe.qpos.tolist(), "qvel": velocity.tolist(), "requested_hand_gaps_m": requested,
                "actual_hand_gaps_m": actual, "mode": "+".join(sides)}

    def make_states(self, seeds, stage="near"):
        if stage not in ("near", "hands"):
            raise ValueError("Unknown initial-state curriculum stage")
        states = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            accepted = False
            for attempt in range(50):
                if stage == "near":
                    pose = self.nominal.copy()
                    pose[2] += rng.uniform(.003, .015)
                    pose[7:] += rng.uniform(-.01, .01, len(pose) - 7)
                    velocity = np.zeros(self.model.nv)
                    velocity[:3] = rng.uniform(-.1, .1, 3)
                    velocity[3:6] = rng.uniform(-.15, .15, 3)
                    state = {"qpos": pose.tolist(), "qvel": velocity.tolist()}
                else:
                    state = self.hand_candidate(rng, seed)
                    if state is None:
                        continue
                state.update(seed=int(seed), attempt=attempt, stage=stage)
                try:
                    self.reset(state)
                    accepted = True
                    break
                except ValueError:
                    continue
            if not accepted:
                raise ValueError(f"No admissible initial state for seed {seed}")
            states.append(state)
        return states

    def save_and_replay(self, path):
        replay = mujoco.MjData(self.model)
        self.model.actuator_gainprm[:] = self.native_gain
        self.model.actuator_biasprm[:] = self.native_bias
        replay.qpos[:], replay.qvel[:], replay.ctrl[:] = self.positions[0], self.velocities[0], self.target
        mujoco.mj_forward(self.model, replay)
        for index, control in enumerate(self.controls):
            if self.phases[index]:
                self.model.actuator_gainprm[:] = self.crawl_gain
                self.model.actuator_biasprm[:] = self.crawl_bias
            replay.ctrl[:] = control
            mujoco.mj_step(self.model, replay)
            mujoco.mj_forward(self.model, replay)
            np.testing.assert_allclose(replay.qpos, self.positions[index + 1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(replay.qvel, self.velocities[index + 1], atol=1e-10, rtol=0)
        np.savez_compressed(path, qpos=self.positions, qvel=self.velocities, ctrl=self.controls, phase=self.phases,
                            samples=self.samples, support_positions=self.support_positions, penetrations=self.penetrations)
