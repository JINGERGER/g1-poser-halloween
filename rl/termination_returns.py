import numpy as np


def compare_returns(rewards, failures, gamma, dt):
    rewards = np.asarray(rewards, dtype=float)
    failures = np.asarray(failures, dtype=bool)
    if rewards.ndim != 2 or not rewards.size or rewards.shape != failures.shape:
        raise ValueError("Expected matching nonempty [steps, episodes] arrays")
    if not np.isfinite(rewards).all() or not 0 < gamma <= 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid rewards, discount or timestep")
    discounted = rewards * np.power(gamma, np.arange(len(rewards)))[:, None]
    full_return = discounted.sum(axis=0)
    results = []
    for episode in range(rewards.shape[1]):
        indices = np.flatnonzero(failures[:, episode])
        first = int(indices[0]) if len(indices) else None
        prefix = float(discounted[:first + 1, episode].sum()) if first is not None else None
        tail = float(discounted[first + 1:, episode].sum()) if first is not None else None
        results.append({"episode": episode, "first_failure_step": first,
                        "first_failure_s": (first + 1) * dt if first is not None else None,
                        "full_discounted_return": float(full_return[episode]), "stopped_discounted_return": prefix,
                        "discarded_tail_return": tail, "stopping_benefit": -tail if tail is not None else None})
    failed = [row for row in results if row["first_failure_step"] is not None]
    return {"episodes": results, "summary": {
        "episode_count": len(results), "failure_count": len(failed),
        "mean_first_failure_s": float(np.mean([row["first_failure_s"] for row in failed])) if failed else None,
        "mean_stopped_return": float(np.mean([row["stopped_discounted_return"] for row in failed])) if failed else None,
        "mean_full_return_failed_episodes": float(np.mean([row["full_discounted_return"] for row in failed])) if failed else None,
        "mean_full_return_all_episodes": float(full_return.mean()),
        "mean_stopping_benefit": float(np.mean([row["stopping_benefit"] for row in failed])) if failed else None,
        "fraction_failed_episodes_with_stopping_benefit": float(np.mean([row["stopping_benefit"] > 1e-8 for row in failed])) if failed else None,
    }}
