from pathlib import Path
import argparse
import json

import mlx.core as mx
import numpy as np
from policy import load_policy
from direct_task import BalancedEnv
from single_foot import SingleFootEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate first episodes on held-out crawl resets without selecting successful clips")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    env, model = load_policy(args.checkpoint, args.episodes, args.seed)
    alive = np.ones(args.episodes, dtype=bool)
    successes = np.zeros(args.episodes, dtype=bool)
    lengths = np.zeros(args.episodes, dtype=int)
    totals = {name: np.zeros(args.episodes) for name in ("speed_error", "support_fraction", "slip", "clearance_error", "peak_swing_contact", "peak_swing_count", "peak_swing_clearance")}
    progress = np.zeros(args.episodes)
    maximum_drift = np.zeros(args.episodes)
    maximum_anchor_drift = np.zeros(args.episodes)
    limb_totals = {name: np.zeros((args.episodes, 4)) for name in ("limb_peak_count", "limb_peak_contact", "limb_peak_site_rise")}
    contact_totals = np.zeros((args.episodes, 4))
    geometry_totals = {name: np.zeros((args.episodes, 4)) for name in ("limb_peak_geom_gap", "limb_peak_clear", "limb_landing_count", "limb_landing_contact")} if isinstance(env, BalancedEnv) else {}
    cycle_totals = {name: np.zeros((args.episodes, 4)) for name in ("peak", "clear", "landing", "contact", "completed", "passed")}
    previous_landing = np.zeros((args.episodes, 4), dtype=bool)
    try:
        state = env.init_state()
        for step in range(env.cfg.max_episode_steps):
            action = np.asarray(model.policy(mx.array(state.obs["obs"])))
            state = env.step(action)
            lengths[alive] += 1
            for name in totals:
                totals[name][alive] += state.info["evaluation"][name][alive]
            for name in limb_totals:
                limb_totals[name][alive] += state.info["evaluation"][name][alive]
            contact_totals[alive] += state.info["evaluation"]["limb_contact"][alive]
            for name in geometry_totals:
                geometry_totals[name][alive] += state.info["evaluation"][name][alive]
            if geometry_totals:
                metrics = state.info["evaluation"]
                landing = metrics["limb_landing_count"] > 0
                for target, source in (("peak", "limb_peak_count"), ("clear", "limb_peak_clear"), ("landing", "limb_landing_count"), ("contact", "limb_landing_contact")):
                    cycle_totals[target][alive] += metrics[source][alive]
                finished = previous_landing & ~landing & alive[:, None]
                minimum_samples = max(1, int(env.cfg.phase_period * 0.1 / env.cfg.ctrl_dt) - 1)
                complete = finished & (cycle_totals["peak"] >= minimum_samples) & (cycle_totals["landing"] >= minimum_samples)
                passed = complete & (cycle_totals["clear"] / np.maximum(cycle_totals["peak"], 1) >= 0.7) & (cycle_totals["contact"] / np.maximum(cycle_totals["landing"], 1) >= 0.9)
                cycle_totals["completed"] += complete
                cycle_totals["passed"] += passed
                for name in ("peak", "clear", "landing", "contact"):
                    cycle_totals[name][finished] = 0
                previous_landing = landing
            progress[alive] = state.info["evaluation"]["progress"][alive]
            maximum_drift[alive] = np.maximum(maximum_drift[alive], state.info["evaluation"]["horizontal_drift"][alive])
            if "stance_anchor_drift" in state.info["evaluation"]:
                maximum_anchor_drift[alive] = np.maximum(maximum_anchor_drift[alive], state.info["evaluation"]["stance_anchor_drift"][alive])
            successes |= alive & state.truncated & ~state.terminated
            alive &= ~(state.terminated | state.truncated)
            if not alive.any():
                break
        peak_contact = float(totals["peak_swing_contact"].sum() / max(totals["peak_swing_count"].sum(), 1))
        peak_clearance = float(totals["peak_swing_clearance"].sum() / max(totals["peak_swing_count"].sum(), 1))
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "task": type(env).__name__,
            "target_lift_m": env.cfg.lift_height,
            "seed": args.seed,
            "episodes": args.episodes,
            "survival_rate_20s": float(successes.mean()),
            "mean_duration_seconds": float(lengths.mean() * env.cfg.ctrl_dt),
            "mean_progress_m": float(progress.mean()),
            "p95_max_horizontal_drift_m": float(np.quantile(maximum_drift, 0.95)),
            "p95_max_stance_anchor_drift_m": float(np.quantile(maximum_anchor_drift, 0.95)) if "stance_anchor_drift" in state.info["evaluation"] else None,
            "mean_speed_mae": float(np.mean(totals["speed_error"] / lengths)),
            "mean_contact_fraction": float(np.mean(totals["support_fraction"] / lengths)),
            "mean_contact_slip_squared": float(np.mean(totals["slip"] / lengths)),
            "mean_clearance_error_m": float(np.mean(totals["clearance_error"] / lengths)),
            "peak_swing_contact_fraction": peak_contact,
            "peak_swing_clearance_m": peak_clearance,
            "stage": "single-foot" if isinstance(env, SingleFootEnv) else "support" if env.cfg.speed_max == 0 else "crawl",
            "flat_gate_passed": bool(successes.mean() >= 0.95 and np.mean(totals["speed_error"] / lengths) < 0.05 and env.cfg.speed_min > 0 and peak_contact < 0.3 and peak_clearance > 0.04),
            "low_lift_screen_passed": bool(successes.mean() >= 0.95 and env.cfg.speed_min > 0 and peak_contact < 0.3 and peak_clearance >= 0.015 and env.cfg.lift_height == 0.02),
            "per_limb_peak_swing": {
                name: {
                    "all_step_contact_fraction": float(np.mean(contact_totals[:, index] / lengths)),
                    "sample_count": int(limb_totals["limb_peak_count"][:, index].sum()),
                    "contact_fraction": float(limb_totals["limb_peak_contact"][:, index].sum() / max(limb_totals["limb_peak_count"][:, index].sum(), 1)),
                    "site_rise_m": float(limb_totals["limb_peak_site_rise"][:, index].sum() / max(limb_totals["limb_peak_count"][:, index].sum(), 1)),
                } for index, name in enumerate(("left_palm", "right_palm", "left_foot", "right_foot"))
            },
            "note": "Numerical gate only; video review and multiple training seeds are still required.",
        }
        report["balanced_lift_gate_passed"] = None
        if geometry_totals:
            peak_counts = limb_totals["limb_peak_count"].sum(axis=0)
            landing_counts = geometry_totals["limb_landing_count"].sum(axis=0)
            clear_fraction = geometry_totals["limb_peak_clear"].sum(axis=0) / np.maximum(peak_counts, 1)
            landing_fraction = geometry_totals["limb_landing_contact"].sum(axis=0) / np.maximum(landing_counts, 1)
            completed = cycle_totals["completed"].sum(axis=0)
            cycle_fraction = cycle_totals["passed"].sum(axis=0) / np.maximum(completed, 1)
            for index, item in enumerate(report["per_limb_peak_swing"].values()):
                item.update({
                    "mean_geom_gap_m": float(geometry_totals["limb_peak_geom_gap"][:, index].sum() / max(peak_counts[index], 1)),
                    "clear_1_5cm_fraction": float(clear_fraction[index]),
                    "landing_sample_count": int(landing_counts[index]),
                    "landing_contact_fraction": float(landing_fraction[index]),
                    "complete_swing_landing_windows": int(completed[index]),
                    "swing_landing_success_fraction": float(cycle_fraction[index]),
                })
                if peak_counts[index] == 0:
                    for key in ("contact_fraction", "site_rise_m", "mean_geom_gap_m", "clear_1_5cm_fraction"):
                        item[key] = None
                if landing_counts[index] == 0:
                    item["landing_contact_fraction"] = None
                if completed[index] == 0:
                    item["swing_landing_success_fraction"] = None
            report["balanced_lift_gate_passed"] = bool(successes.mean() >= 0.95 and np.all(completed > 0) and np.all(clear_fraction >= 0.7) and np.all(landing_fraction >= 0.9) and np.all(cycle_fraction >= 0.9))
            report["note"] += " Balanced lift gate checks each limb's 1.5 cm geometry clearance and paired swing/landing windows, excluding incomplete windows; it does not certify forward locomotion or stability throughout the whole cycle."
            if isinstance(env, SingleFootEnv):
                selected = env.cfg.target_foot
                stance = [index for index in range(4) if index != selected]
                stance_contact = np.mean(contact_totals[:, stance] / lengths[:, None], axis=0)
                report["target_foot"] = ("left_foot", "right_foot")[selected - 2]
                report["balanced_lift_gate_passed"] = None
                report["single_foot_lift_gate_passed"] = bool(successes.mean() >= 0.95 and completed[selected] > 0 and clear_fraction[selected] >= 0.7 and landing_fraction[selected] >= 0.9 and cycle_fraction[selected] >= 0.9 and np.all(stance_contact >= 0.95) and np.quantile(maximum_drift, 0.95) <= 0.1)
                report["note"] += " Only the selected foot is scheduled to swing; other limbs have null swing metrics and must maintain >=95% contact individually. The p95 of episode maximum horizontal drift must be <=10 cm. This does not certify the opposite foot or alternating crawling."
        output = args.checkpoint.with_name(f"{args.checkpoint.stem}_evaluation_seed{args.seed}.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
