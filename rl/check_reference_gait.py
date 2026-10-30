from copy import deepcopy
import json
from pathlib import Path

from reference_gait import paired_cycles, summarize_episode


def row(index, kind="loaded"):
    return {"elapsed_s": (index + 1) * 0.02, "loads_n": [10.0 if kind == "loaded" else 0.0] * 4,
            "floor_contacts": [kind == "loaded"] * 4, "gaps_m": [0.0 if kind == "loaded" else 0.02] * 4,
            "slip_mps": [0.0] * 4, "progress_m": 0.0, "lateral_m": 0.0, "penetration_m": 0.0,
            "other_load_n": 0.0, "max_torque_ratio": 0.1, "speed_error_mps": 0.1, "turn_error_radps": 0.0}


def main():
    config = json.loads(Path(__file__).with_name("reference_eval.json").read_text())
    sequence = [row(index, kind) for index, kind in enumerate(["loaded"] * 3 + ["air"] * 3 + ["loaded"] * 3)]
    assert [limb["count"] for limb in paired_cycles(sequence, config, 0.02)] == [1] * 4
    assert all(limb["count"] == 0 for limb in paired_cycles(sequence[:6], config, 0.02))
    assert all(limb["count"] == 0 for limb in paired_cycles(sequence[3:], config, 0.02))
    for field, value in (("gaps_m", 0.0), ("floor_contacts", True), ("loads_n", 2.0)):
        broken = deepcopy(sequence)
        for sample in broken[3:6]:
            sample[field] = [value] * 4
        assert all(limb["count"] == 0 for limb in paired_cycles(broken, config, 0.02))
    flicker = [row(index, kind) for index, kind in enumerate(["loaded"] * 3 + ["air"] * 2 + ["loaded"] * 3)]
    assert all(limb["count"] == 0 for limb in paired_cycles(flicker, config, 0.02))
    repeated = sequence + [row(index + 9, kind) for index, kind in enumerate(["air"] * 3 + ["loaded"] * 3)]
    assert all(limb["count"] == 2 for limb in paired_cycles(repeated, config, 0.02))
    stationary = [row(index) for index in range(1000)]
    assert not summarize_episode(stationary, True, config, 0.02)["passed"]
    assert summarize_episode(sequence, False, config, 0.02)["loaded_slip_mps"] is None
    hovering = [row(index, "air") for index in range(1000)]
    summary = summarize_episode(hovering, True, config, 0.02)
    assert summary["loaded_slip_mps"] is None and not summary["passed"]
    good = [row(index, "air" if index % 20 in (10, 11, 12) else "loaded") for index in range(1000)]
    for sample in good:
        sample["progress_m"] = 0.1 * sample["elapsed_s"]
        sample["speed_error_mps"] = 0.0
    assert summarize_episode(good, True, config, 0.02)["passed"]
    for field, value, gate in (("slip_mps", [0.2] * 4, "loaded_slip_mps"), ("other_load_n", 20.0, "non_support_fraction"),
                               ("penetration_m", 0.02, "penetration"), ("max_torque_ratio", 1.0, "saturation_fraction")):
        bad = deepcopy(good)
        for sample in bad:
            sample[field] = value
        assert not summarize_episode(bad, True, config, 0.02)["gates"][gate]
    try:
        summarize_episode(good[1:], True, config, 0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing samples accepted")
    print("PASS: paired events, initial drop, incomplete flight, contact/load/gap conjunction, flicker, repeated cycles, missing samples and physical gates")


if __name__ == "__main__":
    main()
