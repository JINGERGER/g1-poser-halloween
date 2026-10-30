import json
from datetime import datetime
from time import perf_counter

import numpy as np
import torch

from artifact_utils import digest
from position_torque_guard import InfeasiblePositionTarget, PositionTorqueGuard
from reference_policy_baseline import ROOT
from search_dynamic_transition import longest_interval
from stand_transition_task import StandTransitionTask, UnsafeState
from standing_capture_routes import smooth


SUFFIXES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
MIRRORED = {"hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw", "wrist_roll"}


def mapping(task):
    names = [task.model.joint(int(joint)).name for joint in task.joints]
    indices = np.array([[names.index(f"{side}_{suffix}_joint") for suffix in SUFFIXES] for side in ("left", "right")])
    signs = np.array([[-1. if side == 1 and suffix in MIRRORED else 1. for suffix in SUFFIXES] for side in (0, 1)])
    return indices, signs


def target_at(task, parameters, elapsed):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (23,) or not np.isfinite(parameters).all() or parameters[-1] <= 0.:
        raise ValueError("Expected 22 finite knot coordinates and a positive duration")
    if not np.isfinite(elapsed) or elapsed < 0.:
        raise ValueError("Expected finite nonnegative elapsed time")
    indices, signs = mapping(task)
    knots = np.repeat(task.target[None, :], 4, axis=0)
    knots[-1] = task.crawl_target
    for index, values in enumerate(np.asarray(parameters[:-1]).reshape(2, 11), start=1):
        knots[index, indices] = signs * values
    coordinate = np.clip(elapsed / parameters[-1], 0., 1.) * 3.
    lower = min(int(coordinate), 2)
    blend = smooth(coordinate - lower)
    return (1. - blend) * knots[lower] + blend * knots[lower + 1]


def rollout(task, state, parameters, path=None, handoff_delay=None):
    task.reset(state)
    guard = PositionTorqueGuard(task.model)
    failure, handoff = [], None
    try:
        for unused in range(25):
            guard.advance(task.execution, task.target)
        for step in range(round((parameters[-1] + 1.) / .02)):
            guard.advance(task.execution, target_at(task, parameters, (step + 1) * .02))
            ready = task.ready()
            if handoff_delay is not None:
                count = max(1, round(handoff_delay / .002))
                window = np.asarray(task.samples[-count:])
                ready = len(window) == count and bool(np.all(window[:, 4:6] > 5.))
            if ready:
                handoff = task.finish()
                break
    except (UnsafeState, InfeasiblePositionTarget) as error:
        failure = task.failure or [str(error)]
    rows = np.asarray(task.samples)
    valid = rows[:-1] if task.failure else rows
    valid = valid[valid[:, 0] > .5]
    result = {"success": bool(handoff and handoff["success"] and not task.failure), "handoff": handoff,
              "failure": failure or task.failure, "elapsed_s": float(task.data.time),
              "both_hands_s": longest_interval(np.all(valid[:, 4:6] > 5., axis=1)) if len(valid) else 0.,
              "best_worst_hand_gap_m": float(np.min(np.max(valid[:, 8:10], axis=1))) if len(valid) else 2.,
              "min_base_height_m": float(valid[:, 3].min()) if len(valid) else None,
              "peak_penetration_m": np.max(task.penetrations, axis=0).tolist(),
              "peak_torque_ratio": float(rows[:, 13].max()), "parameters": parameters.tolist()}
    if path is not None:
        task.save_and_replay(path)
        result.update(trajectory=str(path.relative_to(ROOT)), trajectory_sha256=digest(path), replay_passed=True)
    return result


def ranking(result):
    return (int(result["success"]), int(result["both_hands_s"] >= .1), min(.1, result["both_hands_s"]),
            -result["best_worst_hand_gap_m"])


def main():
    torch.set_num_threads(1)
    task = StandTransitionTask()
    state = task.make_states([0])[0]
    state["qvel"] = np.zeros(task.model.nv).tolist()
    indices, signs = mapping(task)
    ranges = task.model.jnt_range[task.joints]
    lower, upper = [], []
    for column in range(len(SUFFIXES)):
        bounds = [sorted(signs[side, column] * ranges[indices[side, column]]) for side in (0, 1)]
        lower.append(max(bound[0] for bound in bounds) + .03)
        upper.append(min(bound[1] for bound in bounds) - .03)
    lower, upper = np.r_[np.tile(lower, 2), .4], np.r_[np.tile(upper, 2), 2.]
    symmetric_start = np.mean(task.target[indices] * signs, axis=0)
    symmetric_goal = np.mean(task.crawl_target[indices] * signs, axis=0)
    output = ROOT / "output/rl/coordinated_contact_search" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    sources = ("search_coordinated_contact.py", "stand_transition_task.py", "near_handoff_task.py",
               "position_torque_guard.py", "transition_execution.py", "search_dynamic_transition.py")
    report = {"status": "running", "full_transition_validated": False, "families": [],
              "protocol": "Two mirrored 11-joint knots plus duration: 23D bounded shooting search. True standing and .5s native settling in every rollout; no asymmetric stepping prefix. 2 families x 8 generations x 16 candidates, with 2 prior restarts and mean retention each generation. No static equilibrium gate; original 2ms physical gates and actual frozen-policy handoff unchanged. Valid-prefix gap is a search score, never success.",
              "limitations": "A structural symmetry prior, not motion capture or Olson policy reproduction. Deterministic nominal search only; no robustness or hardware claim.",
              "sources": {name: digest(ROOT / "rl" / name) for name in sources},
              "source_snapshots": {name: (ROOT / "rl" / name).read_text() for name in sources},
              "policy_sha256": digest(task.policy_path), "bundle_sha256": digest(task.bundle_path),
              "model_assets": {str(path.relative_to(task.model_path.parent)): digest(path)
                               for path in sorted(task.model_path.parent.rglob("*")) if path.is_file()}}
    started = perf_counter()
    for family_index, family in enumerate(("splay_leads", "fold_leads")):
        fractions = np.full((2, 11), .35)
        fractions[1] = .8
        fractions[0, [1, 2]] = .65 if family == "splay_leads" else .25
        fractions[0, [0, 3]] = .25 if family == "splay_leads" else .65
        fractions[0, 6:] = .15
        center = np.r_[(symmetric_start + fractions * (symmetric_goal - symmetric_start)).ravel(), .9]
        center = np.clip(center, lower, upper)
        std = np.r_[np.full(22, .35), .3]
        initial_std, mean = std.copy(), center.copy()
        rng = np.random.default_rng(310 + family_index)
        entry = {"family": family, "seed": 310 + family_index, "generations": []}
        report["families"].append(entry)
        best = None
        for generation in range(8):
            population = rng.normal(mean, std, (16, 23))
            population[0] = mean
            population[-2:] = rng.normal(center, initial_std, (2, 23))
            population = np.clip(population, lower, upper)
            results = [rollout(task, state, parameters) for parameters in population]
            ordered = sorted(range(len(results)), key=lambda index: ranking(results[index]), reverse=True)
            selected = results[ordered[0]]
            if best is None or ranking(selected) > ranking(best):
                best = selected
            elite = population[ordered[:4]]
            mean = .5 * mean + .5 * elite.mean(axis=0)
            std = np.maximum(.5 * std + .5 * elite.std(axis=0), np.r_[np.full(22, .12), .15])
            entry["generations"].append({"index": generation, "candidates": results, "best_score": ranking(best)})
            print(family, generation, "gap", best["best_worst_hand_gap_m"], "hands_s", best["both_hands_s"],
                  "failure", best["failure"], "wall_s", perf_counter() - started, flush=True)
            report["wall_time_s"] = perf_counter() - started
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if best["success"]:
                break
        verified = rollout(task, state, np.array(best["parameters"]), output / f"{family}.npz")
        assert ranking(verified) == ranking(best)
        entry["verified_best"] = verified
        report["full_transition_validated"] |= verified["success"]
        if verified["success"]:
            break
    report["status"] = "completed"
    report["wall_time_s"] = perf_counter() - started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
