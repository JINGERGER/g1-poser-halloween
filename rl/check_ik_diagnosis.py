import numpy as np

from diagnose_ik_policy import cancellation


def main():
    reference = np.arange(20).reshape(10, 2) * .01
    assert cancellation(reference, -reference)["opposing_projection"] == 1
    assert cancellation(reference, -reference)["combined_ac_ratio"] == 0
    assert cancellation(reference, reference)["opposing_projection"] == -1
    assert abs(cancellation(reference, np.ones_like(reference) * .3)["opposing_projection"]) < 1e-10
    assert cancellation(np.ones_like(reference), reference) is None
    assert cancellation(reference[:2], reference[:2]) is None
    print("IK diagnostic cancellation checks passed")


if __name__ == "__main__":
    main()
