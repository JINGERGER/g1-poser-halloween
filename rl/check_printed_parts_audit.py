import argparse
import json
from pathlib import Path

import numpy as np

from printed_parts_audit import mesh_summary, read_part, stl_bounds
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    result = mesh_summary(vertices, triangles)
    assert result["closed_consistently_oriented_edges"]
    np.testing.assert_allclose(result["signed_volume_m3"], 1 / 6, atol=1e-12)
    shifted = mesh_summary(vertices + [4, -2, 3], triangles)
    np.testing.assert_allclose(shifted["signed_volume_m3"], result["signed_volume_m3"], atol=1e-12)
    assert mesh_summary(vertices, triangles[:-1])["enclosed_volume_cm3"] is None
    reversed_result = mesh_summary(vertices, triangles[:, ::-1])
    assert reversed_result["closed_consistently_oriented_edges"]
    np.testing.assert_allclose(reversed_result["signed_volume_m3"], -1 / 6, atol=1e-12)
    report = json.loads(args.report.read_text())
    assert report["status"] == "completed" and len(report["parts"]) == 4
    for name, expected in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected
    for key in ("readme", "source_xml"):
        assert digest(report[key]) == report[key + "_sha256"]
    for part in report["parts"]:
        assert read_part(Path(part["file"])) == part
        assert part["declared_unit"] == "meter"
        assert all(obj["closed_consistently_oriented_edges"] for obj in part["objects"])
    assert stl_bounds(Path(report["current_visual_forearm"]["file"])) == report["current_visual_forearm"]
    print("PASS: closed/open/orientation/translation mesh tests, source hashes, 3MF meter units and geometry, STL bounds")


if __name__ == "__main__":
    main()
