from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from component_unload_ramps import guard
from forearm_alignment import load_meshes
from load_transfer import physical_metrics
from short_step_task import ShortStepCfg, ShortStepEnv
from target_load_ablation import digest


ROOT = Path(__file__).resolve().parents[1]


def aligned_cad(model_path):
    manifest_path = model_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    evidence = {str(manifest_path): digest(manifest_path)}
    current = manifest
    for key in ("source_report", "source_report", "source_report", "alignment_report", "source_report"):
        path = Path(current[key])
        assert digest(path) == current[key + "_sha256"]
        evidence[str(path)] = digest(path)
        if key == "alignment_report":
            alignment = json.loads(path.read_text())
        current = json.loads(path.read_text())
    part = next(part for part in current["parts"] if Path(part["file"]).name == "forearm.3mf")
    visual = current["current_visual_forearm"]
    for item in (part, visual):
        assert digest(item["file"]) == item["sha256"]
        evidence[item["file"]] = item["sha256"]
    vertices, unused_triangles, unused_visual = load_meshes(Path(part["file"]), Path(visual["file"]))
    transform = alignment["selected"]
    return vertices @ np.array(transform["rotation_column_vector"]).T + transform["translation_m"], evidence


def replay(bundle, trial, cad):
    assert digest(trial["trajectory"]) == trial["trajectory_sha256"]
    env = ShortStepEnv(ShortStepCfg(bundle=str(bundle), reference_gain=trial["reference_gain"]))
    env.init_state()
    model, data = env.model, env.datas[0]
    shell, hand, floor = (model.geom(name).id for name in ("left_forearm", "left_hand_collision", "floor"))
    body = int(model.geom_bodyid[shell])
    assert model.geom_bodyid[hand] == body
    assert model.geom_margin[shell] == 0 and model.geom_gap[shell] == 0
    np.testing.assert_allclose(data.geom_xpos[floor], 0., atol=1e-12)
    np.testing.assert_allclose(data.geom_xmat[floor].reshape(3, 3), np.eye(3), atol=1e-12)
    mesh = model.geom_dataid[shell]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    rows = []
    elapsed = 0.

    def measure():
        rotation = data.geom_xmat[shell].reshape(3, 3)
        shell_gap = float(np.min(vertices @ rotation[2]) + data.geom_xpos[shell, 2])
        body_rotation = data.xmat[body].reshape(3, 3)
        heights = cad @ body_rotation[2] + data.xpos[body, 2]
        cad_gap = float(np.min(heights))
        assert abs(shell_gap - cad_gap) < 1e-6
        hand_gap = float(data.geom_xpos[hand, 2] - model.geom_size[hand, 0])
        contacts = []
        for index, contact in enumerate(data.contact):
            if {int(contact.geom1), int(contact.geom2)} != {shell, floor}:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            contacts.append({"dist_m": float(contact.dist), "normal_force_n": max(0., float(wrench[0])), "position_m": contact.pos.tolist()})
        metric = physical_metrics(model, data)
        force = sum(contact["normal_force_n"] for contact in contacts)
        np.testing.assert_allclose(force, metric["other_loads_n"].get("left_forearm", 0.), atol=1e-10)
        return {"elapsed_s": elapsed, "shell_gap_m": shell_gap, "cad_gap_m": cad_gap,
                "hand_gap_m": hand_gap, "shell_above_ball_bottom_m": shell_gap - hand_gap,
                "base_z_m": float(data.qpos[2]), "wrist_axis_elevation_deg": float(np.degrees(np.arcsin(np.clip(body_rotation[2, 0], -1, 1)))),
                "lowest_cad_vertex_local_m": cad[int(np.argmin(heights))].tolist(),
                "left_load_n": metric["loads_n"][0], "forearm_load_n": force, "contacts": contacts}, metric

    try:
        initial, unused = measure()
        rows.append(initial)
        failure = None
        with np.load(trial["trajectory"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(data.qpos, saved["qpos"][0])
            for index, action in enumerate(saved["actions"]):
                frame = min(index + 1, len(env.offsets) - 1)
                target = env.initial_ctrl + env.cfg.reference_gain * env.offsets[frame] + env.protocol["residual_scale_rad"] * np.clip(action.astype(float), -1, 1)
                target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                data.ctrl[:] += np.clip(target - data.ctrl, -env.protocol["max_ctrl_delta_rad"], env.protocol["max_ctrl_delta_rad"])
                for unused in range(env.cfg.sim_substeps):
                    mujoco.mj_step(model, data)
                    peak = max((-contact.dist for contact in data.contact), default=0.)
                    mujoco.mj_forward(model, data)
                    elapsed += env.cfg.sim_dt
                    row, metric = measure()
                    rows.append(row)
                    failure = guard(model, data, metric, max(peak, metric["penetration"]["depth_m"]))
                    if any(floor not in (contact.geom1, contact.geom2) and contact.dist < -.001 for contact in data.contact):
                        failure = "self_collision"
                    if failure:
                        break
                np.testing.assert_allclose(data.qpos, saved["qpos"][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(data.qvel, saved["qvel"][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(elapsed, trial["trace"][index]["elapsed_s"], atol=1e-10, rtol=0)
                if failure:
                    assert index == len(saved["actions"]) - 1
            assert failure == trial["failure"]
        joints = model.actuator_trnid[:, 0]
        differences = data.qpos[model.jnt_qposadr[joints]] - env.initial_qpos[model.jnt_qposadr[joints]]
        changes = [{"joint": model.joint(int(joints[index])).name, "delta_rad": float(differences[index])}
                   for index in np.argsort(-abs(differences))[:6]]
        return {"failure": failure, "elapsed_s": elapsed, "native_replay_passed": True,
                "initial": initial, "final": rows[-1], "first_shell_contact": next((row for row in rows if row["contacts"]), None),
                "first_forearm_over_1n": next((row for row in rows if row["forearm_load_n"] > 1), None),
                "max_cad_hull_gap_difference_m": max(abs(row["shell_gap_m"] - row["cad_gap_m"]) for row in rows),
                "largest_joint_changes": changes, "trace": rows}
    finally:
        env.close()


def main():
    curve_path = ROOT / "output/rl/unload_curve/2026-10-19_21-59-26-517322/report.json"
    curve = json.loads(curve_path.read_text())
    assert curve["status"] == "completed"
    output = ROOT / "output/rl/forearm_contact_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output.mkdir(parents=True)
    report = {"status": "running", "curve_report": str(curve_path), "curve_sha256": digest(curve_path),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in (Path(__file__).name, "forearm_alignment.py", "short_step_task.py", "component_unload_ramps.py", "load_transfer.py")},
              "limitations": "Native replay only, no policy updates or new actions. CAD comparison uses previously selected alignment, not independently measured hardware calibration. Convex-hull versus CAD plane extent agreement does not validate arbitrary non-planar contacts or printed compliance. No counterfactual force claims.", "runs": []}
    path = output / "report.json"
    try:
        for run in curve["runs"]:
            for checkpoint in run["evaluations"]:
                evidence = checkpoint["report"]
                assert digest(evidence["path"]) == evidence["sha256"]
                evaluation = json.loads(Path(evidence["path"]).read_text())
                bundle_path = Path(evaluation["bundle"])
                assert digest(bundle_path) == evaluation["bundle_sha256"]
                bundle = json.loads(bundle_path.read_text())
                cad, provenance = aligned_cad(Path(bundle["model"]))
                report["cad_provenance"] = provenance
                report["bundle"] = str(bundle_path)
                report["bundle_sha256"] = digest(bundle_path)
                cases = [trial for trial in evaluation["runs"] if trial["label"] == "ppo_deterministic"]
                if run["seed"] == 1 and checkpoint["updates"] == 64:
                    cases += [trial for trial in evaluation["runs"] if trial["label"] in ("hold", "ik_zero_residual")]
                for trial in cases:
                    result = replay(bundle_path, trial, cad)
                    report["runs"].append({"seed": run["seed"], "updates": checkpoint["updates"], "label": trial["label"],
                                           "evaluation": evidence, "trajectory_sha256": trial["trajectory_sha256"], **result})
                    print(json.dumps({"seed": run["seed"], "updates": checkpoint["updates"], "label": trial["label"], "final": result["final"]}), flush=True)
        assert len(report["runs"]) == 8
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
