import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from unilab.base.observations import flatten_obs_dict

from direct_task import projected_gravity
from evaluate_reference import load_reference, sha256
from termination_returns import compare_returns


def audit(entry):
    assessment = json.loads(Path(entry["evaluation"]).read_text())
    checkpoint = Path(entry["checkpoint"])
    if sha256(checkpoint) != assessment["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after evaluation")
    protocol = assessment["evaluation_protocol"]
    count = len(assessment["episodes"])
    env, policy, config = load_reference(checkpoint, count, assessment["seed"], protocol)
    captured = {}
    original_update = env.update_state

    def continue_after_failure(state):
        updated = original_update(state)
        captured["failures"] = updated.terminated.copy()
        captured["height"] = env._backend.get_base_pos()[:, 2].copy()
        captured["gravity_error"] = np.sum((projected_gravity(env._backend.get_base_quat()) - [1, 0, 0]) ** 2, axis=1)
        return updated.replace(terminated=np.zeros(count, dtype=bool))

    env.update_state = continue_after_failure
    started = time.perf_counter()
    try:
        state = env.init_state()
        initial = env.get_physics_state_snapshot().copy()
        actual_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in initial]
        expected_hashes = [row["initial_state_sha256"] for row in assessment["episodes"]]
        if actual_hashes != expected_hashes:
            raise AssertionError("Counterfactual initial states differ from the fixed evaluation")
        rewards = []
        failures = []
        heights = []
        gravity_errors = []
        contributions = {name: 0.0 for name in config.reward.scales}
        gamma = float(config.algo.algorithm.gamma)
        for step in range(env.cfg.max_episode_steps):
            actions = np.asarray(policy.policy(mx.array(flatten_obs_dict(state.obs))), dtype=np.float32)
            state = env.step(actions)
            rewards.append(state.reward.copy())
            failures.append(captured["failures"])
            heights.append(captured["height"])
            gravity_errors.append(captured["gravity_error"])
            if not all(np.isfinite(values).all() for values in (state.reward, captured["height"], captured["gravity_error"], *state.obs.values())):
                raise ValueError("Nonfinite counterfactual rollout; do not interpret as valid return evidence")
            for name, scale in config.reward.scales.items():
                contributions[name] += gamma ** step * float(scale) * state.info["log"][f"reference/{name}"] * env.cfg.ctrl_dt
            if state.terminated.any():
                raise AssertionError("Counterfactual still terminates physically")
            if state.truncated.any():
                if not state.truncated.all() or step != env.cfg.max_episode_steps - 1:
                    raise AssertionError("Unexpected counterfactual timeout")
        if not state.truncated.all():
            raise AssertionError("Counterfactual did not reach the complete horizon")
        result = compare_returns(np.asarray(rewards), np.asarray(failures), gamma, env.cfg.ctrl_dt)
        reasons = Counter()
        for row, previous in zip(result["episodes"], assessment["episodes"]):
            first = row["first_failure_step"]
            expected = None if previous["survived"] else round(previous["duration_s"] / env.cfg.ctrl_dt) - 1
            if first != expected:
                raise AssertionError("First failure does not match the unmodified fixed-policy evaluation")
            if first is not None:
                episode = row["episode"]
                reason = []
                if heights[first][episode] < 0.08:
                    reason.append("low_base")
                if heights[first][episode] > 1.0:
                    reason.append("high_base")
                if gravity_errors[first][episode] > 3.5:
                    reason.append("orientation")
                if not reason:
                    raise AssertionError("Unaccounted physical termination")
                row["failure_reasons"] = reason
                reasons.update(reason)
        if not np.isclose(sum(contributions.values()), result["summary"]["mean_full_return_all_episodes"], rtol=1e-5, atol=1e-4):
            raise AssertionError("Discounted reward decomposition does not sum to measured returns")
        result.update(pd=entry["pd"], seed=entry["seed"], checkpoint=str(checkpoint), gamma=gamma,
                      initial_states_and_first_failure_match=True, failure_reasons=dict(reasons),
                      mean_full_horizon_reward_contributions=contributions,
                      diagnostic_transitions=count * env.cfg.max_episode_steps,
                      wall_seconds=time.perf_counter() - started)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="No-training fixed-policy counterfactual: measure rewards omitted by physical early termination")
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    if matrix["status"] != "completed":
        raise ValueError("Audit a completed matrix, never an in-progress experiment")
    for name in ("reference_task.py", "reference_model.py", "servo_force.py"):
        if sha256(Path(__file__).with_name(name)) != matrix["source_sha256"][name]:
            raise ValueError(f"Frozen task implementation changed: {name}")
    report = {"matrix_sha256": sha256(args.matrix), "status": "running", "runs": [],
              "training_transitions": 0, "intervention": "Suppress physical terminated flags only; keep policy, dynamics, rewards, commands, initial states and 20s timeout unchanged",
              "limitations": "Fixed deterministic policies and 0.1m/s evaluation command. Measured discounted finite-horizon rewards without critic bootstrap. A negative omitted tail demonstrates a local stopping incentive for this continuation, not causal proof of training behavior, a globally optimal policy, or a validated new controller."}
    output = args.matrix.with_name("termination_counterfactual.json")
    try:
        for entry in matrix["runs"]:
            result = audit(entry)
            report["runs"].append(result)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"pd": result["pd"], "seed": result["seed"], "summary": result["summary"],
                              "failure_reasons": result["failure_reasons"], "wall_seconds": result["wall_seconds"]}, indent=2), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(output, flush=True)


if __name__ == "__main__":
    main()
