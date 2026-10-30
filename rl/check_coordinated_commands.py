import json

import numpy as np

from coordinated_blend_task import CoordinatedBlendTask
from run_coordinated_transition import CONFIG
from search_coordinated_contact import mapping, target_at
from standing_capture_routes import integration_state


def main():
    config = json.loads(CONFIG.read_text())
    task = CoordinatedBlendTask()
    task.make_states([0])
    before = integration_state(task)
    parameters = np.asarray(config["parameters"])
    indices, signs = mapping(task)
    np.testing.assert_array_equal(target_at(task, parameters, 0.), task.target)
    np.testing.assert_allclose(target_at(task, parameters, parameters[-1]), task.crawl_target, atol=1e-12)
    np.testing.assert_allclose(target_at(task, parameters, parameters[-1] + 1.), task.crawl_target, atol=1e-12)
    for knot in (1, 2):
        target = target_at(task, parameters, parameters[-1] * knot / 3.)
        np.testing.assert_allclose(target[indices[0]], target[indices[1]] * signs[1], atol=1e-12)
        before_knot = target_at(task, parameters, parameters[-1] * knot / 3. - 1e-7)
        after_knot = target_at(task, parameters, parameters[-1] * knot / 3. + 1e-7)
        np.testing.assert_allclose(before_knot, after_knot, atol=1e-10)
    for elapsed in np.linspace(0., parameters[-1], 101):
        target = target_at(task, parameters, elapsed)
        assert np.all(target >= task.model.jnt_range[task.joints, 0])
        assert np.all(target <= task.model.jnt_range[task.joints, 1])
    for duration in (0., -.1, .015, float("nan")):
        task.blend_duration = duration
        try:
            task.finish()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid blend duration accepted")
    np.testing.assert_array_equal(integration_state(task), before)
    print("PASS: exact endpoints, mirrored knots, continuous targets, joint bounds, invalid blend rejection, state immutability")


if __name__ == "__main__":
    main()
