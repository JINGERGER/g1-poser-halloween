import argparse
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from load_transfer import physical_metrics
from prepare import ROOT, SUPPORTS
from servo_force import position_servo_force, validate_position_servos
from target_load_ablation import digest


PROFILES = (("nominal", 1., 1, 1.), ("half_timestep", .5, 1, 1.),
            ("double_iterations", 1., 2, 1.), ("contact_time_0.8", 1., 1, .8), ("contact_time_1.2", 1., 1, 1.2))


def configure(path, profile):
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep *= profile[1]
    model.opt.iterations *= profile[2]
    assert np.all(model.geom_solref[:, 0] > 0)
    model.geom_solref[:, 0] *= profile[3]
    return model


def inventory(model, qpos):
    validate_position_servos(model)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    lower = position_servo_force(model, model.actuator_ctrlrange[:, 0], data.qpos[7:], np.zeros(model.nu))
    upper = position_servo_force(model, model.actuator_ctrlrange[:, 1], data.qpos[7:], np.zeros(model.nu))
    actuators = []
    for index, joint in enumerate(model.actuator_trnid[:, 0]):
        actuators.append({"joint": model.joint(int(joint)).name, "joint_range_rad": model.jnt_range[joint].tolist(),
                          "ctrl_range_rad": model.actuator_ctrlrange[index].tolist(), "gear": model.actuator_gear[index].tolist(),
                          "kp": float(model.actuator_gainprm[index, 0]), "kv": float(-model.actuator_biasprm[index, 2]),
                          "dyntype": int(model.actuator_dyntype[index]), "gaintype": int(model.actuator_gaintype[index]),
                          "biastype": int(model.actuator_biastype[index]),
                          "joint_force_range_nm": model.jnt_actfrcrange[joint].tolist(), "armature": float(model.dof_armature[index + 6]),
                          "zero_velocity_torque_range_at_pose_nm": [float(lower[index]), float(upper[index])]})
    bodies = []
    for index in range(1, model.nbody):
        inertia = model.body_inertia[index]
        bodies.append({"body": model.body(index).name, "mass_kg": float(model.body_mass[index]), "principal_inertia": inertia.tolist(),
                       "inertial_position_m": model.body_ipos[index].tolist(),
                       "inertia_sanity": bool(model.body_mass[index] >= 0 and np.all(inertia >= 0) and 2 * max(inertia) <= sum(inertia) + 1e-10)})
    geoms = []
    for name in (*SUPPORTS, "floor"):
        geom = model.geom(name).id
        geoms.append({"name": name, "type": int(model.geom_type[geom]), "size": model.geom_size[geom].tolist(),
                      "condim": int(model.geom_condim[geom]), "friction": model.geom_friction[geom].tolist(),
                      "solref": model.geom_solref[geom].tolist(), "solimp": model.geom_solimp[geom].tolist(),
                      "priority": int(model.geom_priority[geom]), "margin": float(model.geom_margin[geom])})
    return {"mass_kg": float(sum(model.body_mass)), "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "timestep": model.opt.timestep, "iterations": model.opt.iterations, "integrator": int(model.opt.integrator),
            "gravity": model.opt.gravity.tolist(), "actuators": actuators, "bodies": bodies, "support_geoms": geoms,
            "limitations": "XML parameters, not measured hardware. Static torque intervals at one pose do not certify coupled body-wrench authority, torque-speed limits, thermal limits or sustained power."}


def model_diff(reference, current):
    differences = []
    groups = (("body", reference.nbody, ("mass", "inertia", "ipos", "iquat", "pos", "quat")),
              ("joint", reference.njnt, ("range", "axis", "pos", "actfrcrange")),
              ("geom", reference.ngeom, ("type", "size", "pos", "quat", "contype", "conaffinity", "condim", "friction", "solref", "solimp")),
              ("actuator", reference.nu, ("gainprm", "biasprm", "ctrlrange", "gear")))
    for kind, count, fields in groups:
        for index in range(count):
            original = getattr(reference, kind)(index)
            if not original.name:
                continue
            try:
                updated = getattr(current, kind)(original.name)
            except KeyError:
                differences.append({"kind": kind, "name": original.name, "missing_in_snapshot": True})
                continue
            for field in fields:
                prefix = "jnt" if kind == "joint" else kind
                before = np.asarray(getattr(reference, f"{prefix}_{field}")[index])
                after = np.asarray(getattr(current, f"{prefix}_{field}")[updated.id])
                if before.shape != after.shape or not np.allclose(before, after, atol=1e-12, rtol=0):
                    differences.append({"kind": kind, "name": original.name, "field": field, "source": before.tolist(), "snapshot": after.tolist()})
    return differences


def effective_contacts(model, data):
    return [{"pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name], "dim": int(contact.dim),
             "solref": contact.solref.tolist(), "solimp": contact.solimp.tolist(), "friction": contact.friction.tolist()}
            for contact in data.contact]


def replay_profile(model, saved, output):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    data.ctrl[:] = saved["initial_ctrl"]
    mujoco.mj_forward(model, data)
    contacts = effective_contacts(model, data)
    peak = physical_metrics(model, data)["penetration"]["depth_m"]
    positions, velocities, controls, rows = [data.qpos.copy()], [data.qvel.copy()], [], []
    substeps = round(float(saved["dt"]) / model.opt.timestep)
    assert abs(substeps * model.opt.timestep - float(saved["dt"])) < 1e-12
    for control in saved["ctrl"]:
        data.ctrl[:] = control
        controls.append(control.copy())
        for substep in range(substeps):
            mujoco.mj_step(model, data)
            peak = max(peak, max((-contact.dist for contact in data.contact), default=0))
        mujoco.mj_forward(model, data)
        row = physical_metrics(model, data)
        row["elapsed_s"] = len(controls) * float(saved["dt"])
        rows.append(row)
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        peak = max(peak, row["penetration"]["depth_m"])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or peak > .02 or row["base_height_m"] < .1:
            break
    slips = [row["loaded_slip_mps"] for row in rows if row["loaded_slip_mps"] is not None]
    passed = bool(len(rows) == len(saved["ctrl"]) and peak <= .005 and slips and np.mean(slips) <= .02
                  and all(row["other_total_n"] <= 1 and .16 <= row["base_height_m"] <= .65 for row in rows)
                  and np.mean([max(row["torque_ratios"]) >= .95 for row in rows]) <= .1)
    np.savez_compressed(output, qpos=positions, qvel=velocities, ctrl=controls, initial_physics_state=saved["initial_physics_state"], initial_ctrl=saved["initial_ctrl"], dt=saved["dt"])
    return {"duration_s": len(rows) * float(saved["dt"]), "historical_engineering_gate_passed": passed,
            "max_penetration_m": peak, "final_loads_n": rows[-1]["loads_n"], "final_com_xy_m": rows[-1]["com_xy"],
            "mean_loaded_slip_mps": float(np.mean(slips)) if slips else None, "initial_effective_contacts": contacts,
            "trajectory": str(output), "trace": rows}


def mesh_hashes(path):
    root = ET.parse(path).getroot()
    directory = Path(root.find("compiler").get("meshdir", "."))
    if not directory.is_absolute():
        directory = path.parent / directory
    return {str((directory / mesh.get("file")).resolve()): digest(directory / mesh.get("file"))
            for mesh in root.findall("./asset/mesh") if mesh.get("file")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("equilibrium_report", type=Path)
    args = parser.parse_args()
    equilibrium = json.loads(args.equilibrium_report.read_text())
    assert equilibrium["status"] == "completed" and equilibrium["solution"]["passed"]
    for name, expected in equilibrium["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    chain = equilibrium
    for key in ("audit_report", "source_report", "baseline_report", "geometry_report", "source_report"):
        assert digest(chain[key]) == chain[key + "_sha256"]
        chain = json.loads(Path(chain[key]).read_text())
    path = Path(chain["model"])
    assert digest(path) == chain["model_sha256"] and mujoco.__version__ == chain["mujoco"]
    source_xml = ROOT / "g1_description/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    reference = mujoco.MjModel.from_xml_path(str(source_xml))
    destination = ROOT / "output/rl/model_validity_audit" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "training_updates": 0, "runs": [], "mujoco": mujoco.__version__,
              "equilibrium_report": str(args.equilibrium_report.resolve()), "equilibrium_report_sha256": digest(args.equilibrium_report),
              "model": str(path), "model_sha256": digest(path), "source_xml": str(source_xml), "source_xml_sha256": digest(source_xml),
              "mesh_sha256": {**mesh_hashes(path), **mesh_hashes(source_xml)},
              "manifest": json.loads(path.with_name("manifest.json").read_text()), "manifest_sha256": digest(path.with_name("manifest.json")),
              "source_sha256": {**equilibrium["source_sha256"], Path(__file__).name: digest(__file__)},
              "inventory": inventory(model, equilibrium["solution"]["qpos"]), "source_snapshot_differences": model_diff(reference, model),
              "collision_exclusions": {"source": [dict(item.attrib) for item in ET.parse(source_xml).getroot().findall("./contact/exclude")],
                                       "snapshot": [dict(item.attrib) for item in ET.parse(path).getroot().findall("./contact/exclude")]},
              "protocol": {"profiles": PROFILES, "trajectories": ["hypothetical_hold", "real_transfer"], "reoptimization": False,
                           "emergency_penetration_m": .02, "historical_gate_m": .005},
              "limitations": "Sensitivity audit only; no hardware calibration, controller tuning or threshold relaxation. Identical state and controls, no re-equilibration after changing contact time constant. Diagnostic continuation beyond historical gate is not approved safe execution. Named compiled-object diff excludes unnamed geoms and mesh vertex comparisons; meshes are hashed."}
    try:
        for profile in PROFILES:
            for branch in report["protocol"]["trajectories"]:
                original = next(run for run in equilibrium["runs"] if run["branch"] == branch)
                with np.load(original["trajectory"], allow_pickle=False) as saved:
                    configured = configure(path, profile)
                    result = replay_profile(configured, saved, destination / f"{profile[0]}_{branch}.npz")
                    if profile[0] == "nominal":
                        with np.load(result["trajectory"], allow_pickle=False) as actual:
                            for key in ("qpos", "qvel", "ctrl"):
                                np.testing.assert_allclose(actual[key], saved[key], atol=1e-9, rtol=0)
                    result.update(profile=profile[0], branch=branch, source_trajectory=original["trajectory"], source_trajectory_sha256=digest(original["trajectory"]))
                    report["runs"].append(result)
                    print(json.dumps({key: value for key, value in result.items() if key not in ("trace", "initial_effective_contacts")}), flush=True)
                    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
