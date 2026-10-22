import numpy as np

from verify_crawl_candidate import reconstruct


def fixture(phases=(0., .5, .5, 0.05), moving=True, clearance=.02, airtime=.12):
    times = np.arange(0., 3., .002)
    samples = np.zeros((len(times), 15))
    samples[:, 0] = times
    samples[:, 4:8] = 50.
    samples[:, 8:12] = -.001
    positions = np.zeros((len(times), 4, 3))
    for limb, phase in enumerate(phases):
        for cycle in range(5):
            start = .2 + .5 * (cycle + phase)
            mask = (times >= start) & (times < start + airtime)
            samples[mask, 4 + limb] = 0.
            samples[mask, 8 + limb] = clearance
            positions[:, limb, 0] += np.clip((times - start) / airtime, 0., 1.) * (.08 if moving else 0.)
    return samples, positions


def main():
    samples, positions = fixture()
    assert reconstruct(samples, positions)[2]
    for options in ({"moving": False}, {"clearance": .009}, {"airtime": .06},
                    {"phases": (0., 0., 0., 0.)}, {"phases": (0., .5, 0., .5)}):
        assert not reconstruct(*fixture(**options))[2], options
    bounce = (samples[:, 0] > 1.12) & (samples[:, 0] < 1.13)
    samples[bounce, 6] = 0.
    samples[bounce, 10] = .0001
    assert reconstruct(samples, positions)[2]
    samples, positions = fixture()
    samples[:, 6] = np.where(samples[:, 6] > 5., 2., samples[:, 6])
    assert not reconstruct(samples, positions)[2]
    print("Event verifier checks passed: diagonal cycles, no-drag, clearance, duration, phase, micro-bounce and reload.")


if __name__ == "__main__":
    main()
