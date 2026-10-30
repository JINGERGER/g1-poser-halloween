import argparse
import json
from pathlib import Path

import numpy as np

from local_arm_geometry import capsule_distance, run
from target_load_ablation import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    first, second = np.zeros(3), np.array([2., 0, 0])
    points = np.array([[1., 0, 0], [1, 2, 0], [-2, 0, 0], [4, 0, 0]])
    expected = [-1, 1, 1, 1]
    np.testing.assert_allclose(capsule_distance(points, first, second, 1), expected)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(capsule_distance(points @ rotation.T + 3, first @ rotation.T + 3,
                                              second @ rotation.T + 3, 1), expected)
    np.testing.assert_allclose(capsule_distance(points, first, first, 1), np.linalg.norm(points, axis=1) - 1)
    report = json.loads(args.report.read_text())
    for name, expected_hash in report["source_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == expected_hash
    assert digest(report["source_report"]) == report["source_report_sha256"]
    assert run(Path(report["source_report"])) == report
    assert len(report["rows"]) == 42
    assert sum(row["joint"] is None for row in report["rows"]) == 2
    assert digest(report["model"]) == report["model_sha256"]
    print("PASS: capsule distances, rigid transform, degenerate segment, provenance and all 42 local geometry probes; model unchanged")


if __name__ == "__main__":
    main()
