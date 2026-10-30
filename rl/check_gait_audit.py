from gait_audit import summarize_gait


def main():
    peak = {"phase": [0.875] * 4, "gaps_m": [0.02, 0.02, 0.0, 0.02],
            "floor_contacts": [False, True, False, False], "loads_n": [0, 0, 0, 2],
            "progress_m": 0.0, "command_mps": 0.03, "forward_speed_mps": 0.0}
    landing = {**peak, "phase": [0.05] * 4, "loads_n": [10] * 4}
    result = summarize_gait([peak, landing])
    assert [row["peak_clear_15mm_unloaded_fraction"] for row in result["per_limb"]] == [1, 0, 0, 0]
    assert all(row["landing_loaded_fraction"] == 1 for row in result["per_limb"])
    assert result["speed_mae_mps"] == 0.03
    assert all(row["peak_contact_fraction"] is None for row in summarize_gait([])["per_limb"])
    assert summarize_gait([peak])["per_limb"][0]["landing_loaded_fraction"] is None
    print("PASS: geometry/contact/load conjunction, landing loads, missing windows and speed error")


if __name__ == "__main__":
    main()
