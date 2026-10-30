import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import fork_prefix
from transition_v1 import DEFAULT, LOWER, PARAMETERS, UPPER, StandingTransitionV1


def evaluate(task, branch, state, parameters, save=None):
    task.reset(state)
    controller = StandingTransitionV1(task, parameters)
    guard = PositionTorqueGuard(task.model, fraction=.8)
    best_score, probes = -100., []
    success = False
    best_state = None
    for step in range(200):
        try:
            guard.advance(task.execution, controller.target(task))
        except (UnsafeState, InfeasiblePositionTarget) as error:
            if not task.failure:
                task.failure = [type(error).__name__]
            break
        row, unused, unused_penetrations = task.measure()
        rotation = task.data.xmat[task.model.body("pelvis").id].reshape(3, 3)
        pitch = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        joint_error = float(np.sqrt(np.mean((task.data.qpos[task.addresses] - task.crawl_target) ** 2)))
        cost = (2. * abs(pitch - 1.5) + 4. * abs(task.data.qpos[2] - .28)
                + 2. * joint_error + 2. * max(0., np.mean(row[8:10]))
                + .4 * np.linalg.norm(task.data.qvel[:3]) + .15 * np.linalg.norm(task.data.qvel[3:6])
                + 3. * max(0., -task.data.qvel[0]) + 2. * abs(task.data.qpos[1]))
        score = -float(cost)
        if score > best_score:
            best_score = score
            best_state = {"time_s": float(task.data.time), "pitch_rad": pitch, "height_m": float(task.data.qpos[2]),
                          "joint_rms_rad": joint_error, "hand_gap_m": float(np.mean(row[8:10])),
                          "base_velocity": task.data.qvel[:6].tolist()}
        if step % 5 == 4 and pitch > .65 and task.data.qpos[2] < .55:
            fork_prefix(task, branch)
            result = branch.finish()
            probes.append({"time_s": float(task.data.time), **result})
            if result["success"]:
                success = True
                best_score = 100. + result["forward_m"]
                if save is not None:
                    branch.save_and_replay(save)
                break
    result = {"success": success, "score": best_score, "best_state": best_state,
              "failure": task.failure, "fold_start_s": controller.fold_start, "probes": probes,
              "parameters": np.asarray(parameters).tolist()}
    if save is not None:
        if not success:
            task.save_and_replay(save)
        result.update(trajectory=str(save.relative_to(ROOT)), trajectory_sha256=digest(save), replay_passed=True)
    return result


def main():
    torch.set_num_threads(1)
    task, branch = StandTransitionTask(), StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    output = ROOT / "output/rl/transition_v1_search" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    report = {"status": "running", "budget": {"seeds": [7, 11, 19], "generations": 8, "population": 24, "search_rollouts": 576},
              "parameters": PARAMETERS, "lower": LOWER.tolist(), "upper": UPPER.tolist(), "initial_state": state,
              "assets": assets, "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          (Path(__file__).name, "transition_v1.py", "standing_capture_routes.py", "stand_transition_task.py",
                           "near_handoff_task.py", "position_torque_guard.py", "transition_execution.py", "crawl_control.py")},
              "protocol": "True standing, native PD, forward-pitch event then fold, physical gates unchanged. "
              "CEM three independent restarts, six elites, 50% update, minimum std5% of range, four uniform global samples per generation. "
              "Ranking combines forward orientation, height, joint pose, hand gap and linear/angular velocities; "
              "success only actual2s frozen crawl on copied full prefix. Offline branch probes every100ms "
              "when pitch>.65,height<.55 are a heuristic search filter, NOT certified capture region or deployable switching. "
              "Failed search rollouts not saved; default and selected per restart exactly replayed. No learned policy claimed.",
              "runs": []}
    path = output / "report.json"
    started = time.monotonic()
    report["default"] = evaluate(task, branch, state, DEFAULT, output / "default.npz")
    try:
        for seed in report["budget"]["seeds"]:
            rng = np.random.default_rng(seed)
            mean, std = DEFAULT.copy(), (UPPER - LOWER) * .22
            run = {"seed": seed, "generations": [], "candidates": []}
            report["runs"].append(run)
            for generation in range(8):
                population = np.clip(rng.normal(mean, std, (24, len(DEFAULT))), LOWER, UPPER)
                population[0] = mean
                population[-4:] = rng.uniform(LOWER, UPPER, (4, len(DEFAULT)))
                results = [evaluate(task, branch, state, parameters) for parameters in population]
                run["candidates"].extend(results)
                elite = sorted(results, key=lambda item: item["score"], reverse=True)[:6]
                values = np.array([item["parameters"] for item in elite])
                mean = .5 * mean + .5 * values.mean(axis=0)
                std = np.maximum(.5 * std + .5 * values.std(axis=0), .05 * (UPPER - LOWER))
                summary = {"generation": generation, "best_score": elite[0]["score"],
                           "successes": sum(item["success"] for item in results), "best_state": elite[0]["best_state"]}
                run["generations"].append(summary)
                report["elapsed_s"] = time.monotonic() - started
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(seed, summary, flush=True)
            selected = max(run["candidates"], key=lambda item: item["score"])
            run["selected"] = evaluate(task, branch, state, selected["parameters"], output / f"selected_{seed}.npz")
            np.testing.assert_allclose(run["selected"]["score"], selected["score"], atol=1e-10, rtol=0)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
