import math

import mlx.core as mx
import mlx.nn as nn
from unilab.algos.mlx.common import MLP
from unilab.algos.mlx.ppo.model import MLPActorCritic


def finite_elu(values):
    return mx.where(values >= 0, values, mx.expm1(mx.minimum(values, 0)))


class ObservationSlice(nn.Module):
    def __init__(self, network, start, stop):
        super().__init__()
        self.network = network
        self.start = start
        self.stop = stop

    def __call__(self, observation):
        if observation.ndim != 2 or observation.shape[1] != 156:
            raise ValueError("Expected packed actor75 + critic81 observations")
        return self.network(observation[:, self.start:self.stop])


def build_reference_model(cfg, obs_dim, action_dim, dtype=None):
    if obs_dim != 156 or action_dim != 23 or cfg.empirical_normalization:
        raise ValueError("Reference model requires 75/81 inputs, 23 actions, no shared normalization")
    policy = cfg.policy
    model = MLPActorCritic(
        obs_dim=75, action_dim=23,
        actor_hidden_dims=policy.actor_hidden_dims, critic_hidden_dims=policy.critic_hidden_dims,
        activation=policy.activation, init_log_std=math.log(policy.init_noise_std),
        noise_std_type="scalar", dtype=mx.float32 if dtype is None else dtype,
    )
    critic = MLP(81, 1, policy.critic_hidden_dims, activation=policy.activation)
    critic.init_orthogonal(hidden_gain=math.sqrt(2.0), output_gain=1.0)
    if policy.activation == "elu":
        model.actor.activation = finite_elu
        critic.activation = finite_elu
    model.actor = ObservationSlice(model.actor, 0, 75)
    model.critic = ObservationSlice(critic, 75, 156)
    return model
