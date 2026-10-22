from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from crawl_scene import make_halloween_scene
from crawl_witches import WITCH_NAMES, animate_witches


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "output/rl/models/component_hulls_v1_12d607bfdb588900/flat.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    with TemporaryDirectory() as temporary:
        scene = make_halloween_scene(source, Path(temporary) / "scene.xml")
        themed = mujoco.MjModel.from_xml_path(str(scene))
        assert (original.nq, original.nv, original.nu) == (themed.nq, themed.nv, themed.nu)
        for name in ("jnt_range", "jnt_actfrcrange", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "dof_damping", "dof_armature"):
            np.testing.assert_array_equal(getattr(original, name), getattr(themed, name))
        for name in ("geom_contype", "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp", "geom_pos", "geom_size"):
            np.testing.assert_array_equal(getattr(original, name), getattr(themed, name)[:original.ngeom])
        for name in ("body_mass", "body_inertia"):
            np.testing.assert_array_equal(getattr(original, name), getattr(themed, name)[:original.nbody])
        assert not themed.geom_contype[original.ngeom:].any()
        assert not themed.geom_conaffinity[original.ngeom:].any()
        assert not themed.body_mass[original.nbody:].any()
        assert themed.body("pumpkin_left").id >= original.nbody
        assert themed.body("ghost_left").id >= original.nbody
        assert themed.body("cemetery_grave_2_5").id >= original.nbody
        assert themed.body("pumpkin_left_pedestal").id >= original.nbody
        assert themed.body("cemetery_grass").id >= original.nbody
        assert themed.nmocap == original.nmocap + len(WITCH_NAMES)
        assert mujoco.mj_name2id(themed, mujoco.mjtObj.mjOBJ_GEOM, "ramp_surface") == -1
        trajectory = np.load(root / "output/rl/release_assets/nominal_trajectory.npz")
        original_data, themed_data = mujoco.MjData(original), mujoco.MjData(themed)
        animate_witches(themed, themed_data)
        initial_witches = themed_data.mocap_pos.copy()
        for data in (original_data, themed_data):
            data.qpos[:] = trajectory["qpos"][0]
            data.qvel[:] = trajectory["qvel"][0]
        for control in trajectory["ctrl"]:
            for model, data in ((original, original_data), (themed, themed_data)):
                data.ctrl[:] = control
                for unused in range(10):
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
            animate_witches(themed, themed_data)
            np.testing.assert_array_equal(original_data.qpos, themed_data.qpos)
            np.testing.assert_array_equal(original_data.qvel, themed_data.qvel)
        assert np.all(np.linalg.norm(themed_data.mocap_pos - initial_witches, axis=1) > .1)
        np.testing.assert_allclose(np.linalg.norm(themed_data.mocap_quat, axis=1), 1.)
    print("Halloween scene: original physics parameters unchanged; 10-second paired dynamics exactly equal; decorations have no mass/collision; ramp absent.")


if __name__ == "__main__":
    main()
