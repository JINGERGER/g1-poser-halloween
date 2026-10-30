import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import mujoco

from model_validity_audit import PROFILES, configure, mesh_hashes, replay_profile
from prepare import ROOT
from target_load_ablation import digest


def hold_gate(result, initial_com):
    with np.load(result["trajectory"], allow_pickle=False) as trajectory:
        velocities = np.max(np.abs(trajectory["qvel"][1:]), axis=1)
    rows = result["trace"]
    errors = [float(np.linalg.norm(np.array(row["com_xy"]) - initial_com)) for row in rows]
    passed = bool(result["historical_engineering_gate_passed"] and len(rows) == 150
                  and all(errors[index] <= .005 and velocities[index] <= .05
                          and all(load > 5 for load in rows[index]["loads_n"])
                          and all(rows[index]["floor_contacts"]) for index in range(max(0, len(rows) - 50), len(rows))))
    return {"hold_gate_passed": passed, "final_com_error_m": errors[-1],
            "final_velocity_max": float(velocities[-1]), "max_com_error_m": max(errors)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.initial_report.read_text())
    assert source["status"] == "completed"
    for name, expected in source["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert digest(source["model"]) == source["model_sha256"]
    assert digest(source["source_report"]) == source["source_report_sha256"]
    candidate = json.loads(Path(source["source_report"]).read_text())
    path = Path(source["model"])
    assert digest(path.with_name("manifest.json")) == candidate["manifest_sha256"]
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(path.parent / "meshes" / name) == expected
    entry = next(entry for entry in source["seeds"] if entry["seed"] == "old_equilibrium")
    assert entry["equilibrium"]["passed"] and entry["hold"]["reached"]
    original = entry["hold"]
    assert digest(original["trajectory"]) == original["trajectory_sha256"]
    destination = ROOT / "output/rl/component_sensitivity" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.initial_report.resolve()), "source_report_sha256": digest(args.initial_report),
              "source_trajectory": original["trajectory"], "source_trajectory_sha256": digest(original["trajectory"]),
              "model": str(path), "model_sha256": digest(path), "mesh_sha256": mesh_hashes(path),
              "mujoco": mujoco.__version__, "training_updates": 0,
              "source_sha256": {**source["source_sha256"], **{name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "model_validity_audit.py")}},
              "protocol": {"profiles": PROFILES, "reoptimization": False, "hold_s": 3,
                           "historical_gate_m": .005, "diagnostic_emergency_m": .02, "fixed_seed": "old_equilibrium"},
              "limitations": "Same initial full physics state and recorded controls in all profiles. Contact time variations are model sensitivity, not hardware calibration. Diagnostic continuation past 5mm does not relax acceptance. Initialized hold only, not reachability or crawling. Adjacent hull filtering retained.", "runs": []}
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with np.load(original["trajectory"], allow_pickle=False) as saved:
            nominal = configure(path, PROFILES[0])
            data = mujoco.MjData(nominal)
            mujoco.mj_setState(nominal, data, saved["initial_physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(nominal, data)
            initial_com = data.subtree_com[1, :2].copy()
            report["initial_com_xy_m"] = initial_com.tolist()
            for profile in PROFILES:
                model = configure(path, profile)
                assert not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
                result = replay_profile(model, saved, destination / f"{profile[0]}.npz")
                with np.load(result["trajectory"], allow_pickle=False) as actual:
                    np.testing.assert_array_equal(actual["initial_physics_state"], saved["initial_physics_state"])
                    np.testing.assert_array_equal(actual["ctrl"], saved["ctrl"][:len(actual["ctrl"])] )
                    if profile[0] == "nominal":
                        for key in ("qpos", "qvel", "ctrl"):
                            np.testing.assert_allclose(actual[key], saved[key], atol=1e-9, rtol=0)
                result.update(hold_gate(result, initial_com))
                result.update(profile=profile[0], timestep_s=float(model.opt.timestep), iterations=int(model.opt.iterations),
                              trajectory_sha256=digest(result["trajectory"]))
                report["runs"].append(result)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: result[key] for key in ("profile", "hold_gate_passed", "max_penetration_m", "final_com_error_m")}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(report_path, flush=True)


if __name__ == "__main__":
    main()
