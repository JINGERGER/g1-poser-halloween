import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

from run_reference_matrix import RUNS, bounded_process, remaining_budget, source_hashes


def main():
    assert len(set(RUNS)) == 4
    assert source_hashes() == source_hashes()
    with patch("run_reference_matrix.time.monotonic", return_value=10), patch("run_reference_matrix.time.time", return_value=500):
        assert remaining_budget(0, 0, 100) == -400
    with patch("run_reference_matrix.time.monotonic", return_value=500), patch("run_reference_matrix.time.time", return_value=10):
        assert remaining_budget(0, 0, 100) == -400
    plan = json.loads(subprocess.check_output([sys.executable, "-B", str(Path(__file__).with_name("run_reference_matrix.py")), "--dry-run"]))
    assert plan["total_planned_transitions"] == 12002304
    assert plan["transitions_per_run"] == 3000576
    assert [(row["pd"], row["seed"]) for row in plan["runs"]] == list(RUNS)
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "child.log"
        bounded_process([sys.executable, "-c", "print('complete')"], log, 5)
        assert "complete" in log.read_text()
        try:
            bounded_process([sys.executable, "-c", "raise SystemExit(7)"], log, 5)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failed child was accepted")
        started = time.monotonic()
        try:
            bounded_process([sys.executable, "-c", "import time; time.sleep(30)"], log, 0.1)
        except subprocess.TimeoutExpired:
            assert time.monotonic() - started < 5
        else:
            raise AssertionError("Timeout did not stop child")
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("train_reference.py")),
                             "--stage", "screen", "--iterations", "10"], capture_output=True, text=True)
    assert result.returncode == 2 and "Frozen screen" in result.stderr
    print("PASS: four-run budget, ordering, dry-run, child failure, timeout cleanup, dual-clock limits, invalid screen configuration rejected")


if __name__ == "__main__":
    main()
