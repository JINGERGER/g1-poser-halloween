import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf

from prepare import ROOT
from reference_model import build_reference_model, finite_elu
from train_reference import load_runner


def main():
    from unilab.algos.mlx.common.activations import get_activation

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    args = parser.parse_args()
    samples = mx.array([-1000, -2, -0.001, 0, 0.001, 2, 1000], dtype=mx.float32)
    old_gradient = np.array(mx.grad(lambda values: mx.sum(get_activation("elu")(values)))(samples))
    new_gradient = np.array(mx.grad(lambda values: mx.sum(finite_elu(values)))(samples))
    assert np.isfinite(new_gradient).all()
    expected = np.exp(np.minimum(np.array(samples), 0))
    assert np.allclose(new_gradient, expected, atol=1e-6)
    assert np.allclose(np.array(finite_elu(samples)), np.where(np.array(samples) >= 0, np.array(samples), np.expm1(np.minimum(np.array(samples), 0))), atol=1e-6)
    cfg = OmegaConf.load(Path(__file__).with_name("reference.yaml")).algo
    mx.random.seed(123)
    model = build_reference_model(cfg, 156, 23)
    observation = mx.random.normal((4, 156))
    actor_changed = mx.concatenate((observation[:, :75] + 10, observation[:, 75:]), axis=1)
    critic_changed = mx.concatenate((observation[:, :75], observation[:, 75:] + 10), axis=1)
    assert np.array_equal(np.array(model.policy(observation)), np.array(model.policy(critic_changed)))
    assert np.array_equal(np.array(model.value(observation)), np.array(model.value(actor_changed)))
    assert not np.allclose(np.array(model.value(observation)), np.array(model.value(critic_changed)))
    assert not np.allclose(np.array(model.policy(observation)), np.array(model.policy(actor_changed)))
    runner = load_runner(Path.home() / "UniLab")
    state = SimpleNamespace(truncated=np.array([True, False, False, True]),
                            final_observation={"obs": np.array(observation[:, :75]), "critic": np.array(observation[:, 75:])}, info={})
    assert np.allclose(np.array(runner.get_time_limit_bootstrap_values(state, model)), np.array(model.value(observation)))
    with tempfile.TemporaryDirectory(dir=ROOT / "output/rl") as directory:
        path = str(Path(directory) / "roundtrip.safetensors")
        model.save_weights(path)
        restored = build_reference_model(cfg, 156, 23)
        restored.load_weights(path, strict=True)
        assert np.array_equal(np.array(model.policy(observation)), np.array(restored.policy(observation)))
        assert np.array_equal(np.array(model.value(observation)), np.array(restored.value(observation)))
    report = {"actor_input": 75, "critic_input": 81, "actor_critic_isolation": "passed", "timeout_bootstrap": "passed", "checkpoint_roundtrip": "passed",
              "upstream_elu_nonfinite_gradient_reproduced": bool(not np.isfinite(old_gradient).all()), "local_elu_gradient": "finite and matches analytic derivative"}
    report["trained_checkpoints"] = []
    for path in args.checkpoint:
        saved = json.loads(path.with_name("run_config.json").read_text())["config"]
        restored = build_reference_model(OmegaConf.create(saved["algo"]), 156, 23)
        restored.load_weights(str(path), strict=True)
        assert np.isfinite(np.array(restored.policy(observation))).all()
        assert np.isfinite(np.array(restored.value(observation))).all()
        first_weights = mx.load(str(path.with_name("model_0.safetensors")))
        final_weights = mx.load(str(path))
        changed = {branch: any(not np.array_equal(np.array(first_weights[name]), np.array(value))
                               for name, value in final_weights.items() if name.startswith(branch + "."))
                   for branch in ("actor", "critic")}
        assert all(changed.values())
        report["trained_checkpoints"].append({"path": str(path), "changed_since_iteration_0": changed, "strict_load_and_finite_inference": "passed"})
    (ROOT / "output/rl/reference_model_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
