import numpy as np

from short_step_assets import prepare
from short_step_task import ShortStepCfg, ShortStepEnv
from unload_objective_task import UnloadObjectiveCfg, UnloadObjectiveEnv, objective_weights


def main():
    bundle = str(prepare())
    original = ShortStepEnv(ShortStepCfg(bundle=bundle), 2)
    objective = UnloadObjectiveEnv(UnloadObjectiveCfg(bundle=bundle), 2)
    try:
        original.init_state()
        objective.init_state()
        rng = np.random.default_rng(1)
        terminals = 0
        for index in range(240):
            actions = np.zeros((2, 23)) if index < 200 else rng.normal(0, .15, (2, 23))
            old = original.step(actions)
            new = objective.step(actions)
            np.testing.assert_array_equal(old.obs["obs"], new.obs["obs"])
            np.testing.assert_array_equal(old.terminated, new.terminated)
            np.testing.assert_array_equal(old.truncated, new.truncated)
            for env_index, (old_row, new_row) in enumerate(zip(old.info["evaluation"], new.info["evaluation"])):
                for key, value in old_row.items():
                    if key != "reward_terms":
                        assert new_row[key] == value
                expected_terms = {**old_row["reward_terms"], "unload": float(np.clip(1 - old_row["loads_n"][0] / 100, 0, 1))}
                assert new_row["reward_terms"] == expected_terms
                expected_delta = .02 * (sum(objective.objective_weights[key] * value for key, value in expected_terms.items())
                                       - sum(original.protocol["reward"][key] * value for key, value in old_row["reward_terms"].items()))
                np.testing.assert_allclose(new.reward[env_index], old.reward[env_index] + expected_delta, atol=1e-6, rtol=0)
                np.testing.assert_array_equal(original.datas[env_index].qpos, objective.datas[env_index].qpos)
            if np.any(old.terminated):
                terminals += int(np.sum(old.terminated))
                np.testing.assert_array_equal(old.final_observation["obs"], new.final_observation["obs"])
        assert terminals > 0
        assert objective_weights(original.protocol["reward"])["failure"] == -10.
        assert objective_weights(original.protocol["reward"])["event"] == original.protocol["reward"]["event"]
        assert objective_weights(original.protocol["reward"])["success"] == original.protocol["reward"]["success"]
        try:
            UnloadObjectiveEnv(UnloadObjectiveCfg(bundle=bundle, reward_config={"scales": {}}))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid reward config accepted")
    finally:
        original.close()
        objective.close()
    print("PASS: reward-only difference across 480 transitions, identical physics/events/termination/autoreset/final observations, unchanged failure/event/success bonuses and reward config validation")


if __name__ == "__main__":
    main()
