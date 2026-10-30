import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from ball_contact_assets import VARIANTS, build_variant, compare_models, probe, self_collision_sweep, write_once
from servo_force import validate_position_servos
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed"
    assert report["dynamic_rollouts"] == report["training_updates"] == 0
    for key in ("base_model", "equilibrium_report", "alignment_report"):
        assert digest(report[key]) == report[key + "_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    base_path = Path(report["base_model"])
    base = mujoco.MjModel.from_xml_path(str(base_path))
    nominal = json.loads(base_path.with_name("manifest.json").read_text())["qpos"]
    equilibrium = json.loads(Path(report["equilibrium_report"]).read_text())
    chain = equilibrium
    for key in ("audit_report", "source_report", "baseline_report", "geometry_report", "source_report"):
        assert digest(chain[key]) == chain[key + "_sha256"]
        chain = json.loads(Path(chain[key]).read_text())
    assert Path(chain["model"]).resolve() == base_path.resolve()
    assert chain["model_sha256"] == digest(base_path)
    old_probe = probe(base, equilibrium["solution"]["qpos"])
    assert old_probe == report["old_equilibrium_probe"]
    assert self_collision_sweep(base, nominal) == report["base_sweep"]
    assert sorted(row["variant"] for row in report["variants"]) == sorted(VARIANTS)
    for row in report["variants"]:
        model_path = Path(row["model"])
        assert digest(model_path) == row["model_sha256"]
        assert digest(model_path.with_name("manifest.json")) == row["manifest_sha256"]
        manifest = json.loads(model_path.with_name("manifest.json").read_text())
        for name, expected in manifest["mesh_sha256"].items():
            assert digest(model_path.parent / "meshes" / name) == expected
        assert build_variant(base_path, Path(report["alignment_report"]), row["variant"]) == model_path.parent
        model = mujoco.MjModel.from_xml_path(str(model_path))
        validate_position_servos(model)
        assert model.nu == base.nu == 23
        assert (model.nq, model.nv) == (base.nq, base.nv)
        assert compare_models(base, model) == row["comparison"]
        for field in ("jnt_type", "jnt_qposadr", "jnt_dofadr", "actuator_trnid", "actuator_forcerange", "geom_condim"):
            np.testing.assert_array_equal(getattr(base, field), getattr(model, field))
        for field in ("timestep", "iterations", "gravity", "integrator", "solver", "cone", "tolerance"):
            np.testing.assert_array_equal(getattr(base.opt, field), getattr(model.opt, field))
        for side in ("left", "right"):
            geom = model.geom(f"{side}_hand_collision")
            assert geom.size[0] == .02
            np.testing.assert_array_equal(geom.pos, [VARIANTS[row["variant"]], 0, 0])
        actual = probe(model, equilibrium["solution"]["qpos"])
        assert actual == row["old_equilibrium_geometry_probe"]
        if row["variant"] == "reference_center":
            np.testing.assert_allclose(np.array(actual["gaps_m"]) - old_probe["gaps_m"], [.015, .015, 0, 0], atol=1e-12, rtol=0)
        assert probe(model, nominal) == row["nominal_geometry_probe"]
        assert self_collision_sweep(model, nominal) == row["sweep"]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "immutable"
        write_once(path, b"original")
        write_once(path, b"original")
        try:
            write_once(path, b"changed")
        except ValueError:
            pass
        else:
            raise AssertionError("Immutable asset was overwritten")
        assert path.read_bytes() == b"original"
    assert digest(base_path) == report["base_model_sha256"]
    print("PASS: provenance, mesh hashes, immutable/idempotent assets, servo mapping, unchanged physics, geometric gaps and all 345 collision probes")


if __name__ == "__main__":
    main()
