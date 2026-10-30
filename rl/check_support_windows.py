from support_windows import summarize_windows


def main():
    trace = [{"elapsed_s": time, "loads_n": loads, "slip_mps": slips,
              "base_speed_mps": 0.0, "drift_m": 0.0}
             for time, loads, slips in (
                 (1.0, [10, 10, 10, 10], [0.1, 0.1, 0.1, 0.1]),
                 (5.0, [10, 0, 10, 10], [0.0, 99, 0.0, 0.0]),
                 (10.0, [0, 0, 0, 0], [99, 99, 99, 99]),
                 (20.0, [10, 10, 10, 10], [0.0, 0.0, 0.0, 0.0]))]
    result = summarize_windows(trace)
    assert [row["samples"] for row in result] == [1, 1, 1, 1]
    assert result[0]["mean_force_weighted_slip_mps"] == 0.1
    assert result[1]["mean_force_weighted_slip_mps"] == 0
    assert result[1]["limb_mean_slip_when_loaded_mps"][1] is None
    assert result[2]["mean_force_weighted_slip_mps"] is None
    assert result[3]["mean_force_weighted_slip_mps"] == 0
    assert all(row["samples"] == 0 for row in summarize_windows([]))
    print("PASS: window boundaries, unloaded contacts, zero-load windows and empty traces")


if __name__ == "__main__":
    main()
