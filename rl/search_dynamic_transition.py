import json
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch

from artifact_utils import digest
from compare_full_transition_routes import joint_target
from reference_policy_baseline import ROOT
from stand_transition_task import StandTransitionTask, UnsafeState


def command_at(parameters, initial, goal, lower, upper, elapsed):
    parameters = np.asarray(parameters)
    if parameters.shape != (70,) or not np.isfinite(parameters).all():
        raise ValueError("Expected 69 knot coordinates and one duration")
    duration = float(np.clip(parameters[-1], .6, 3.))
    knots = np.vstack([initial, parameters[:-1].reshape(3, 23), goal])
    knots = np.clip(knots, lower, upper)
    coordinate = float(np.clip(elapsed / duration, 0., 1.)) * 4.
    index = min(int(coordinate), 3)
    fraction = coordinate - index
    fraction = fraction * fraction * (3. - 2. * fraction)
    return knots[index] + fraction * (knots[index + 1] - knots[index])


def longest_interval(loaded):
    consecutive, longest = 0, 0
    for value in loaded:
        consecutive = consecutive + 1 if value else 0
        longest = max(longest, consecutive)
    return max(0, longest - 1) * .002


def summarize_prefix(samples, failed_last):
    samples = np.asarray(samples)
    valid = samples[:-1] if failed_last else samples
    if not len(valid):
        return {"both_hands_s": 0., "either_hand_s": 0., "min_hand_gap_m": None, "gap_progress": 0.}
    hands = valid[:, 4:6] > 5.
    gap = float(valid[:, 8:10].mean(axis=1).min())
    return {"both_hands_s": longest_interval(hands.all(axis=1)),
            "either_hand_s": longest_interval(hands.any(axis=1)), "min_hand_gap_m": gap,
            "gap_progress": float(np.clip(1. - gap / .78, 0., 1.))}


def ranking(result):
    return (int(result["success"]), int(result["both_hands_s"] >= .1),
            min(result["both_hands_s"], .1), min(result["either_hand_s"], .1), result["gap_progress"])


def rollout(task, state, parameters, path):
    task.reset(state)
    result = None
    transition_failed = False
    for step in range(round((float(np.clip(parameters[-1], .6, 3.)) + 1.) / .02)):
        target = command_at(parameters, task.target, task.crawl_target,
                            task.model.actuator_ctrlrange[:, 0], task.model.actuator_ctrlrange[:, 1], (step + 1) * .02)
        try:
            task.execution.transition_step(target)
        except UnsafeState:
            transition_failed = True
            break
        if task.ready():
            result = task.finish()
            break
    mask = np.array(task.phases) == 0
    samples = np.array(task.samples)[mask]
    prefix = summarize_prefix(samples, transition_failed)
    task.save_and_replay(path)
    return {"success": bool(result and result["success"]), "handoff": result,
            "failure": task.failure or ([] if result and result["success"] else ["no_successful_handoff"]),
            "transition_time_s": float(samples[-1, 0]), **prefix,
            "peak_torque_ratio": float(samples[:, 13].max()),
            "peak_penetration_m": np.max(task.penetrations, axis=0).tolist(),
            "trajectory": str(path.relative_to(ROOT)), "trajectory_sha256": digest(path), "replay_passed": True}


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    task = StandTransitionTask()
    source = ROOT / "output/rl/full_transition_routes/2026-10-23_15-10-52-892109/report.json"
    reference = json.loads(source.read_text())
    if reference["bundle_sha256"] != digest(task.bundle_path) or reference["policy_sha256"] != digest(task.policy_path):
        raise ValueError("Reference assets differ from current task")
    assets = {str(path.relative_to(task.model_path.parent)): digest(path)
              for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}
    if reference["assets"] != assets:
        raise ValueError("Reference model differs from current task")
    nodes = next(case["nodes"] for case in reference["cases"] if case["route"] == "lean_first")
    state = {"qpos": nodes[0]["qpos"], "qvel": np.zeros(task.model.nv).tolist()}
    np.testing.assert_allclose(np.array(state["qpos"])[task.addresses], task.target, atol=1e-12)
    output = ROOT / "output/rl/dynamic_transition_search" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "budget": {"families": 2, "seeds": [7, 11], "generations": 4,
              "population": 12, "search_rollouts": 192, "heldout_rollouts": 16},
              "protocol": "True standing, zero initial velocity for search. Three independent 23-joint position knots "
              "plus duration .6..3s, smooth interpolation, exact standing and crawl joint endpoints. "
              "Native PD, original actuator force limits and physical gates checked every 2ms. "
              "Up to 1s settling, then existing readiness and actual 2s crawl. No base override or external assistance. "
              "Reference-centered sigma .35rad vs endpoint-linear broad sigma .9rad. "
              "Per family/seed: four generations of 12, top four update diagonal mean/std with .5 smoothing; "
              "std floor .15rad/.2s, ceiling 1rad/.7s. Two fresh prior samples each generation, one mean candidate. "
              "Final global-best search candidate per family/seed tested once on four disjoint velocity perturbations. "
              "Every rollout control-replayed and saved. No budget extension or retuning from heldout results.",
              "ranking": "Lexicographic: full success, bilateral hand load >5N sustained >=100ms, "
              "bilateral duration capped .1s, either-hand duration capped .1s, minimum valid-prefix mean hand gap. "
              "First failing physics sample excluded from support/progress ranking. Gap is only an exploratory "
              "tie-breaker, not success or evidence of feasibility; a later-failed trajectory can still rank on its valid prefix.",
              "limitations": "Open-loop dynamic search, not RL or a robust feedback controller. 70-dimensional limited-budget "
              "search cannot prove impossibility; two proposal widths are different search families, not isolated causal ablations. "
              "No score for mere survival or base-height reduction. No hard left/right symmetry.",
              "sources": {name: digest(ROOT / "rl" / name) for name in
                          ("search_dynamic_transition.py", "stand_transition_task.py", "near_handoff_task.py", "transition_execution.py")},
              "reference_report": str(source.relative_to(ROOT)), "reference_sha256": digest(source), "assets": assets,
              "bundle_sha256": digest(task.bundle_path), "policy_sha256": digest(task.policy_path),
              "search_initial_state": state, "runs": []}
    heldout = []
    for index, (axis, speed) in enumerate(((0, .03), (0, -.03), (1, .03), (1, -.03))):
        velocity = np.zeros(task.model.nv)
        velocity[axis] = speed
        heldout.append({"id": index, "qpos": state["qpos"], "qvel": velocity.tolist()})
    report["heldout_states"] = heldout
    path = output / "report.json"

    def write():
        report["elapsed_s"] = time.monotonic() - started
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write()
    try:
        for family in ("reference", "broad"):
            for seed in (7, 11):
                rng = np.random.default_rng(seed)
                if family == "reference":
                    center = np.concatenate([joint_target(nodes, task.addresses, phase) for phase in (.25, .5, .75)] + [[2.]])
                    initial_std = np.r_[np.full(69, .35), .5]
                else:
                    center = np.concatenate([task.target + phase * (task.crawl_target - task.target)
                                             for phase in (.25, .5, .75)] + [[1.5]])
                    initial_std = np.r_[np.full(69, .9), .7]
                mean, std = center.copy(), initial_std.copy()
                run = {"family": family, "seed": seed, "generations": [], "candidates": []}
                report["runs"].append(run)
                for generation in range(4):
                    population = rng.normal(mean, std, (12, 70))
                    population[0] = mean
                    population[-2:] = rng.normal(center, initial_std, (2, 70))
                    population[:, :-1] = np.clip(population[:, :-1],
                                                np.tile(task.model.actuator_ctrlrange[:, 0], 3),
                                                np.tile(task.model.actuator_ctrlrange[:, 1], 3))
                    population[:, -1] = np.clip(population[:, -1], .6, 3.)
                    candidates = []
                    for index, parameters in enumerate(population):
                        filename = output / f"{family}_{seed}_{generation}_{index}.npz"
                        result = rollout(task, state, parameters, filename)
                        candidate = {"generation": generation, "index": index, "parameters": parameters.tolist(), **result}
                        candidates.append(candidate)
                        run["candidates"].append(candidate)
                    elite = sorted(candidates, key=ranking, reverse=True)[:4]
                    elite_parameters = np.array([candidate["parameters"] for candidate in elite])
                    mean = .5 * mean + .5 * elite_parameters.mean(axis=0)
                    std = np.clip(.5 * std + .5 * elite_parameters.std(axis=0), np.r_[np.full(69, .15), .2],
                                  np.r_[np.ones(69), .7])
                    summary = {"generation": generation, "successes": sum(item["success"] for item in candidates),
                               "bilateral_100ms": sum(item["both_hands_s"] >= .1 for item in candidates),
                               "best_rank": ranking(elite[0]),
                               "failures": dict(Counter(reason for item in candidates for reason in item["failure"]))}
                    run["generations"].append(summary)
                    print(json.dumps({"family": family, "seed": seed, **summary}), flush=True)
                    write()
                best = max(run["candidates"], key=ranking)
                run["selected"] = {"generation": best["generation"], "index": best["index"], "rank": ranking(best)}
        for run in report["runs"]:
            best = max(run["candidates"], key=ranking)
            run["heldout"] = []
            for test in heldout:
                filename = output / f"heldout_{run['family']}_{run['seed']}_{test['id']}.npz"
                run["heldout"].append({"id": test["id"], **rollout(task, test, np.array(best["parameters"]), filename)})
                write()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write()
        print(path)


if __name__ == "__main__":
    main()
