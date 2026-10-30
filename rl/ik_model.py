import math

import mlx.core as mx
from unilab.algos.mlx.ppo.model import MLPActorCritic

from reference_model import finite_elu


def build_ik_model(cfg, obs_dim, action_dim, dtype=None):
    if obs_dim != 178 or action_dim != 23 or cfg.empirical_normalization:
        raise ValueError("IK model requires 178 observations and 23 actions")
    model = MLPActorCritic(obs_dim=178, action_dim=23, actor_hidden_dims=cfg.policy.actor_hidden_dims,
                          critic_hidden_dims=cfg.policy.critic_hidden_dims, activation="elu",
                          init_log_std=math.log(cfg.policy.init_noise_std), noise_std_type="scalar",
                          dtype=mx.float32 if dtype is None else dtype)
    model.actor.activation = finite_elu
    model.critic.activation = finite_elu
    return model
