from pathlib import Path
import json

from omegaconf import OmegaConf
from unilab.algos.mlx.ppo import MLPActorCritic
from unilab.base import registry

import crawl_task
import direct_task
import single_foot
import anchored_foot


def load_policy(checkpoint: Path, num_envs: int, seed: int):
    payload = json.loads(checkpoint.with_name("run_config.json").read_text())
    config = OmegaConf.create(payload["config"])
    overrides = OmegaConf.to_container(config.env, resolve=True)
    overrides["reward_config"] = OmegaConf.to_container(config.reward, resolve=True)
    overrides["seed"] = seed
    overrides["reset_joint_noise"] = 0.02
    task = config.training.task_name
    if task not in ("G1CrawlFlat23", "G1CrawlDirect23V2", "G1CrawlBalanced23V3", "G1SingleFoot23V4", "G1AnchoredFoot23V5"):
        raise ValueError(f"Unsupported policy task: {task}")
    env = registry.make(task, num_envs=num_envs, sim_backend="mujoco", env_cfg_override=overrides)
    try:
        model = MLPActorCritic(
            obs_dim=env.obs_groups_spec["obs"], action_dim=23,
            actor_hidden_dims=config.algo.policy.actor_hidden_dims,
            critic_hidden_dims=config.algo.policy.critic_hidden_dims,
            activation=config.algo.policy.activation,
            noise_std_type=config.algo.policy.get("noise_std_type", "scalar"),
            obs_normalization=bool(config.algo.empirical_normalization),
        )
        model.load_weights(str(checkpoint), strict=True)
        model.eval()
        return env, model
    except BaseException:
        env.close()
        raise
