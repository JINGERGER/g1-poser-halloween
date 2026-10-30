import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn

from artifact_utils import digest
from near_handoff_task import NearHandoffTask
from stand_transition_task import StandTransitionTask
from reference_policy_baseline import ROOT


class ActorCritic(nn.Module):
    def __init__(self, observation_size=88):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 23))
        self.critic = nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((23,), -2.3))

    def distribution(self, observation):
        return torch.distributions.Normal(self.actor(observation), self.log_std.clamp(-4., 0.).exp())


def advantages(rewards, values, dones, bootstrap):
    result = np.zeros(len(rewards), dtype=np.float32)
    estimate = 0.
    for index in reversed(range(len(rewards))):
        following = bootstrap if index == len(rewards) - 1 else values[index + 1]
        live = 1. - dones[index]
        delta = rewards[index] + .99 * following * live - values[index]
        estimate = delta + .99 * .95 * live * estimate
        result[index] = estimate
    return result


def evaluate(task, states, network, directory, label):
    results = []
    for index, state in enumerate(states):
        observation = task.reset(state)
        if network is None:
            result = task.run_baseline() if isinstance(task, StandTransitionTask) else task.finish()
        else:
            for unused in range(getattr(task, "horizon", 20)):
                with torch.no_grad():
                    action = network.actor(torch.from_numpy(observation)).numpy()
                observation, reward, done, result = task.step(action)
                if done:
                    break
            assert done
        path = directory / f"{label}_{index}.npz"
        task.save_and_replay(path)
        results.append({"state_seed": state["seed"], **result, "replay_passed": True,
                        "trajectory": str(path), "trajectory_sha256": digest(path)})
    print(json.dumps({"evaluation": label, "successes": sum(result["success"] for result in results), "cases": len(states)}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("near", "hands", "standing"), default="near")
    parser.add_argument("--seed", type=int, choices=(7, 11), default=7)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    task = StandTransitionTask() if args.stage == "standing" else NearHandoffTask()
    train_states = task.make_states(range(32), args.stage)
    heldout = task.make_states(range(1000, 1008), args.stage)
    experiment = "near_handoff_training" if args.stage == "near" else "hand_contact_training"
    if args.stage == "standing":
        experiment = "stand_transition_training"
    output = ROOT / "output/rl" / experiment / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report_path = output / "report.json"
    observation_size = len(task.reset(train_states[0]))
    network = ActorCritic(observation_size)
    initial_parameters = torch.cat([parameter.detach().flatten() for parameter in network.parameters()]).clone()
    report = {"status": "running", "seed": args.seed, "stage": args.stage, "observation_size": observation_size,
              "budget": {"updates": 8, "steps_per_update": 256, "maximum_training_steps": 2048},
              "ppo": {"epochs": 4, "minibatch": 64, "clip": .2, "gamma": .99, "gae_lambda": .95, "learning_rate": .0003,
                      "gradient_clip": .5, "value_coefficient": .5, "entropy_coefficient": .001},
              "protocol": "Near-crawl initialization only, not standing. Native position PD for 20 control steps (0.4s), "
                          "nominal joint targets +0.12*tanh(raw action), then same-state frozen crawl for 2s. "
                          "Initial pools screened for physics; disjoint train seeds 0..31 and heldout 1000..1007. "
                          "Direct handoff, untrained zero-mean adapter, and final checkpoint evaluated on identical heldout states. "
                          "No checkpoint selection or reward tuning from heldout outcomes. "
                          "Every physics substep screened; terminal success requires 2 consecutive qualified alternating cycles and 6cm crawl advance. "
                          "Failure -5; terminal success +20; tiny dense reward for joint-target proximity and action smoothness. "
                          "PPO uses raw Gaussian action likelihood; tanh is a fixed environment action transform.",
              "limitations": "One small CPU PyTorch PPO run, not a UniLab integration, full transition, 10s crawl validation or statistical success estimate.",
              "train_states": train_states, "heldout_states": heldout,
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("train_near_handoff.py", "near_handoff_task.py", "stand_transition_task.py", "transition_execution.py", "crawl_control.py", "verify_crawl_candidate.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "bundle_sha256": digest(task.bundle_path), "crawl_policy_sha256": digest(task.policy_path),
              "versions": {"torch": torch.__version__, "numpy": np.__version__}, "updates": []}
    if args.stage == "hands":
        report["initial_state_protocol"] = (
            "Local arm-Jacobian initialization only: left/right/both hands selected by seed modulo 3, "
            "requested hand gap uniform 1..3cm, other hand unchanged. Base and leg geometry unchanged. "
            "Initial world xy velocity +/-0.05m/s, other velocities zero. Same post-IK physical screening. "
            "No claim that the initial state is dynamically reachable from standing; this task begins just before hand contact.")
    if args.stage == "standing":
        report["protocol"] = (
            "True knee-0.6 standing resets, initial xy velocity +/-0.02m/s. Full 23-joint integrated target actions "
            "with integrated component at <=1rad/s, bounded by joint control ranges; no fixed-foot constraint. Six-second horizon. "
            "Known small lowering reference during 1..2s (+.2 knee, -.1 hip/ankle pitch), not a success criterion. "
            "Observe full hand gaps up to 1m and all 23 integrated target offsets (111 observations). "
            "The deterministic lowering reference adds its own target velocity. "
            "All physical gates unchanged except transition height ceiling .95m. "
            "Handoff requires all four support loads >5N for 100ms, pelvis <=.4m, base speed norm <.5, "
            "joint RMS distance to crawl pose <.35rad, followed by actual frozen crawl for 2s. "
            "Reward v2: 10*(gamma*next potential-previous potential), gamma=.99; terminal potential ZERO. "
            "Potential=4*height drop+2*mean hand-gap reduction; action cost .001; "
            "physical failure -5, timeout -2, terminal crawl success +50. No alive reward. "
            "Direct column is standing hold; untrained column is known lowering seed only. "
            "Final checkpoint only, fixed 2048-step pilot; no reward tuning from heldout evaluation.")
        report["limitations"] = "Full-start task pilot with very small training budget, not a feasibility verdict. No full transition demonstrated unless terminal success passes. Readiness thresholds are an initial fixed contract, not a proven necessary region."

    def write():
        report["elapsed_s"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write()
    try:
        report["direct"] = evaluate(task, heldout, None, output, "direct")
        report["untrained"] = evaluate(task, heldout, network, output, "untrained")
        write()
        if args.stage == "hands" and all(item["success"] for item in report["direct"] + report["untrained"]):
            report.update(status="completed", training_skipped=True,
                          reason="Both no-learning baselines passed every screening case; no measured room for a success-rate gain on this set.",
                          heldout_success_counts={name: sum(item["success"] for item in report[name]) for name in ("direct", "untrained")})
            return
        optimizer = torch.optim.Adam(network.parameters(), lr=.0003)
        observation = task.reset(train_states[int(rng.integers(len(train_states)))])
        for iteration in range(8):
            observations, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
            outcomes = []
            episode_details = []
            for unused in range(256):
                tensor = torch.from_numpy(observation)
                with torch.no_grad():
                    distribution = network.distribution(tensor)
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action).sum().item()
                    value = network.critic(tensor).item()
                observations.append(observation.copy())
                actions.append(action.numpy())
                log_probs.append(log_prob)
                values.append(value)
                observation, reward, done, info = task.step(action.numpy())
                rewards.append(reward)
                dones.append(float(done))
                if done:
                    outcomes.append(info["success"])
                    episode_details.append(info)
                    observation = task.reset(train_states[int(rng.integers(len(train_states)))])
            with torch.no_grad():
                bootstrap = network.critic(torch.from_numpy(observation)).item()
            advantage = advantages(rewards, values, dones, bootstrap)
            returns = torch.from_numpy(advantage + np.array(values, dtype=np.float32))
            advantage = torch.from_numpy((advantage - advantage.mean()) / (advantage.std() + 1e-8))
            obs_tensor, act_tensor = torch.from_numpy(np.array(observations)), torch.from_numpy(np.array(actions))
            old_log = torch.tensor(log_probs, dtype=torch.float32)
            losses = []
            for unused_epoch in range(4):
                permutation = rng.permutation(256)
                for start in range(0, 256, 64):
                    batch = permutation[start:start + 64]
                    distribution = network.distribution(obs_tensor[batch])
                    log_prob = distribution.log_prob(act_tensor[batch]).sum(-1)
                    ratio = (log_prob - old_log[batch]).exp()
                    actor_loss = -torch.minimum(ratio * advantage[batch], ratio.clamp(.8, 1.2) * advantage[batch]).mean()
                    value_loss = (network.critic(obs_tensor[batch]).squeeze(-1) - returns[batch]).square().mean()
                    loss = actor_loss + .5 * value_loss - .001 * distribution.entropy().sum(-1).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPO loss")
                    optimizer.zero_grad()
                    loss.backward()
                    gradient = nn.utils.clip_grad_norm_(network.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for parameter in network.parameters()):
                        raise FloatingPointError("Nonfinite PPO parameters")
                    losses.append(float(loss.detach()))
            row = {"iteration": iteration + 1, "training_steps": (iteration + 1) * 256,
                   "episodes_completed": len(outcomes), "successes": sum(outcomes), "mean_reward": float(np.mean(rewards)),
                   "mean_loss": float(np.mean(losses)), "last_unclipped_gradient_norm": float(gradient)}
            if args.stage == "standing":
                row["episode_details"] = episode_details
            report["updates"].append(row)
            print(json.dumps(row), flush=True)
            write()
        checkpoint = output / "final.pt"
        torch.save(network.state_dict(), checkpoint)
        loaded = ActorCritic(observation_size)
        loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        report["trained"] = evaluate(task, heldout, loaded, output, "trained")
        report["parameter_l2_change"] = float(torch.linalg.vector_norm(torch.cat([parameter.detach().flatten() for parameter in network.parameters()]) - initial_parameters))
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_sha256"] = digest(checkpoint)
        report["heldout_success_counts"] = {name: sum(item["success"] for item in report[name]) for name in ("direct", "untrained", "trained")}
        report["improves_direct_on_this_set"] = report["heldout_success_counts"]["trained"] > report["heldout_success_counts"]["direct"]
        report["improves_untrained_on_this_set"] = report["heldout_success_counts"]["trained"] > report["heldout_success_counts"]["untrained"]
        report["added_learning_benefit_on_this_set"] = (
            report["improves_direct_on_this_set"] and report["improves_untrained_on_this_set"])
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write()
        print(report_path)


if __name__ == "__main__":
    main()
