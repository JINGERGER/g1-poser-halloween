import argparse
import json
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from torch import nn

from artifact_utils import digest
from closed_loop_transition_task import ClosedLoopTransitionTask
from recovery_transition_task import RecoveryTransitionTask
from reference_policy_baseline import ROOT
from train_near_handoff import ActorCritic, advantages, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=("crawl", "none", "event"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=64)
    args = parser.parse_args()
    if args.updates <= 0:
        raise ValueError("Updates must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    task = RecoveryTransitionTask() if args.prior == "event" else ClosedLoopTransitionTask(args.prior == "crawl")
    train_states = task.make_states(range(32))
    heldout = task.make_states(range(1000, 1008))
    observation_size = len(task.reset(train_states[0]))
    network = ActorCritic(observation_size)
    with torch.no_grad():
        network.log_std.fill_(-.7)
    output = ROOT / "output/rl/closed_loop_transition" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "prior": args.prior, "seed": args.seed,
              "budget": {"updates": args.updates, "steps_per_update": 2048, "maximum_training_steps": args.updates * 2048},
              "protocol": "True standing starts, training xy speed +/-0.02m/s, disjoint seeds for final heldout. "
              "Reference PD x1.4, wrists native. Actor residual 1.5rad*tanh filtered .5 each 60ms, "
              "raw crawl policy refreshed every20ms when enabled; none prior uses standing joint center. "
              "Per2ms position target projection to 50% instantaneous predicted actuator limits; "
              "all actual-state gates unchanged. Guard is not a guarantee of post-step safety; fail on any gate. "
              "161 observations expose residual filter, raw policy history, integrated target and event state. "
              "Event state tracks first-hand contact and bilateral100ms; readiness then continuous frozen crawl2s. "
              "PPO raw Gaussian likelihood, initial std exp(-.7), gamma .99, lambda .95, Adam3e-4, clip .2, "
              "4epochs, minibatch256, gradient .5. No observation normalization. Final checkpoint only.",
              "reward": "2 height/hand-gap potential terms as existing task with gamma .99 and ZERO terminal potential; "
              "+50 success/-5 failure or timeout. Each20ms: .02*exp(-positive mean hand gap/.2)*exp(-base6speed/2) "
              "+.05 for both hands loaded all last10 substeps times exp(-base6speed/2). Residual magnitude cost .001. "
              "Contact shaping is not success and may produce local optima; final acceptance remains unchanged.",
              "limitations": "CPU MuJoCo/PyTorch experiment, not UniLab integration or hardware deployment. "
              "One seed per prior comparison is not statistical evidence. No curriculum reset stitching.",
              "observation_size": observation_size, "train_states": train_states, "heldout_states": heldout,
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("train_closed_loop_transition.py", "closed_loop_transition_task.py", "recovery_transition_task.py", "position_torque_guard.py",
                           "stand_transition_task.py", "near_handoff_task.py", "train_near_handoff.py", "transition_execution.py", "crawl_control.py")},
              "assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                         for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()},
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path), "updates": []}
    report_path = output / "report.json"
    if args.prior == "event":
        report["protocol"] = (
            "True standing starts; native standing PD, actual actuator limits unchanged. "
            "40ms actor decisions with .65 previous/.35 new filtered targets, 2ms torque guard at .8 predicted force. "
            "Soft bilateral exploration: common latent plus .2 independent difference, invertible before tanh. "
            "Full position range around gravity-conditioned standing/crawl posture center, no fixed timed trajectory. "
            "139 observations expose filter, targets, stage, bilateral contact duration and prefix violation flag. "
            "Training only: retain failure contacts for recovery learning; stop on >1cm penetration, height outside .08..1.1, "
            "nonfinite state, or6s timeout. Guard infeasibility falls back to bounded requested target with penalty; "
            "real model force saturation remains. Every original gate violation is recorded and invalidates strict success. "
            "At readiness, tail is evaluated strictly; recovery after prefix violation is NOT full success. "
            "All untrained/final heldout evaluations use original strict gates and immediate failure, never recovery mode. "
            "PPO settings unchanged:2048 steps/update,4epochs, minibatch256, gamma .99, lambda .95, Adam3e-4, clip .2, grad .5. "
            "Final checkpoint only. No changes to collision geometry, inertia, friction or force limits.")
        report["reward"] = (
            "Each20ms: .02*(positive gravity-x + .5*exp(-((height-.25)/.2)^2) + .5*exp(-positive mean hand gap/.12) "
            "+ bilateral20ms*exp(-base6speed/2)); subtract .02*(2*min(non-support load/312,2)+20*penetration+max(torque_ratio-.9,0)). "
            "Subtract .01 per strict-invalid substep, .02 per guard fallback, .001 mean filtered action squared. "
            "Strict full success+50; recovery-only tail success+5; catastrophic/timeout/tail failure-5. "
            "Shaping/recovery reward is not certification; strict heldout determines acceptance.")

    def write():
        report["elapsed_s"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write()
    try:
        report["untrained"] = evaluate(task, heldout, network, output, "untrained")
        write()
        if args.prior == "event":
            task.training_mode = True
        optimizer = torch.optim.Adam(network.parameters(), lr=.0003)
        observation = task.reset(train_states[int(rng.integers(len(train_states)))])
        for iteration in range(args.updates):
            observations, actions, log_probs, values, rewards, dones, outcomes = [], [], [], [], [], [], []
            for unused in range(2048):
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
                    outcomes.append(info)
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
                permutation = rng.permutation(2048)
                for start in range(0, 2048, 256):
                    batch = permutation[start:start + 256]
                    distribution = network.distribution(obs_tensor[batch])
                    ratio = (distribution.log_prob(act_tensor[batch]).sum(-1) - old_log[batch]).exp()
                    actor_loss = -torch.minimum(ratio * advantage[batch], ratio.clamp(.8, 1.2) * advantage[batch]).mean()
                    value_loss = (network.critic(obs_tensor[batch]).squeeze(-1) - returns[batch]).square().mean()
                    loss = actor_loss + .5 * value_loss - .001 * distribution.entropy().sum(-1).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPO loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            summary = {"iteration": iteration + 1, "training_steps": (iteration + 1) * 2048,
                       "episodes": len(outcomes), "successes": sum(item["success"] for item in outcomes),
                       "recovery_only": sum(item.get("recovery_only", False) for item in outcomes),
                       "any_hand_loaded": sum(item["any_hand_loaded"] for item in outcomes),
                       "max_both_hands_s": max([0.] + [item["both_hands_loaded_s"] for item in outcomes]),
                       "failure_counts": dict(Counter(reason for item in outcomes for reason in item["failure"])),
                       "mean_reward": float(np.mean(rewards)), "mean_loss": float(np.mean(losses))}
            report["updates"].append(summary)
            checkpoint = output / "latest.pt"
            torch.save({"network": network.state_dict(), "observation_size": observation_size, "prior": args.prior,
                        "training_steps": summary["training_steps"], "optimizer": optimizer.state_dict()}, checkpoint)
            print(json.dumps(summary), flush=True)
            write()
        loaded = ActorCritic(observation_size)
        loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["network"])
        if args.prior == "event":
            task.training_mode = False
        report["trained"] = evaluate(task, heldout, loaded, output, "trained")
        report["checkpoint"] = str(checkpoint.relative_to(ROOT))
        report["checkpoint_sha256"] = digest(checkpoint)
        report["success_counts"] = {name: sum(item["success"] for item in report[name]) for name in ("untrained", "trained")}
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write()
        print(report_path)


if __name__ == "__main__":
    main()
