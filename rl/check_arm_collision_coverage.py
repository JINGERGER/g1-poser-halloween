import argparse
import json
from pathlib import Path

import numpy as np

from arm_collision_coverage import run, summarize, world_points
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    summary = summarize(np.array([[.01, -.02, .03], [.01, .02, .03], [0, 0, 0]]))
    assert summary["outside_1mm_count"] == 1 and summary["max_outside_m"] == .01
    assert summary["worst_sample_index"] == 1
    rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_array_equal(world_points(np.array([[1., 2, 3]]), np.array([4., 5, 6]), rotation), [[7, 7, 5]])
    report = json.loads(args.report.read_text())
    assert digest(report["source_report"]) == report["source_report_sha256"]
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    assert run(Path(report["source_report"])) == report
    assert len(report["rows"]) == 42 and len(report["components"]) == 6
    model_folder = Path(report["model"]).parent
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    for component in report["components"]:
        if component["part"] == "printed_shell_cad":
            continue
        suffix = "shoulder_yaw_link" if component["part"] == "upper_arm_visual" else "elbow_link"
        contents = (model_folder / "meshes" / f"{component['side']}_{suffix}.STL").read_bytes()
        count = int.from_bytes(contents[80:84], "little")
        assert len(contents) == 84 + 50 * count
        raw = np.frombuffer(contents, dtype=dtype, offset=84)["vertices"].reshape(-1, 3).astype(float)
        np.testing.assert_allclose(component["body_bounds_m"], [raw.min(axis=0), raw.max(axis=0)], atol=1e-7, rtol=0)
        if suffix == "elbow_link":
            assert component["own_active_geoms"] == []
    assert digest(report["model"]) == report["model_sha256"]
    print("PASS: union coverage, frame transform, raw STL versus compiled-mesh bounds, provenance and 252 component/pose evaluations; no model changes")


if __name__ == "__main__":
    main()
