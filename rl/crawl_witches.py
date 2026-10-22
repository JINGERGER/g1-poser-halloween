import math
import xml.etree.ElementTree as ET

import mujoco


WITCH_NAMES = tuple(f"cemetery_witch_{index}" for index in range(3))


def add_witches(assets, world):
    vertices = [(math.cos(index * math.tau / 8) * .14,
                 math.sin(index * math.tau / 8) * .14, 0.) for index in range(8)] + [(0., 0., .32)]
    faces = [(index, (index + 1) % 8, 8) for index in range(8)] + [(0, index + 1, index) for index in range(1, 7)]
    ET.SubElement(assets, "mesh", name="witch_pointed_hat",
                  vertex=" ".join(str(value) for vertex in vertices for value in vertex),
                  face=" ".join(str(value) for face in faces for value in face))
    colors = {"witch_cloak": "0.16 0.045 0.23 1", "witch_hat": "0.045 0.025 0.065 1",
              "witch_skin": "0.40 0.62 0.24 1", "witch_wood": "0.32 0.15 0.055 1",
              "witch_straw": "0.72 0.48 0.16 1", "witch_band": "0.95 0.37 0.07 1"}
    for name, rgba in colors.items():
        ET.SubElement(assets, "material", name=name, rgba=rgba)
    for index, name in enumerate(WITCH_NAMES):
        body = ET.SubElement(world, "body", name=name, mocap="true", pos=f"{index * 2} 2 1.4",
                             childclass="halloween_decoration")

        def geom(kind, material, **attributes):
            return ET.SubElement(body, "geom", type=kind, material=material, **attributes)

        geom("capsule", "witch_wood", fromto="-0.55 0 -0.06 0.55 0 -0.06", size="0.016")
        for strand in range(9):
            angle = strand * math.tau / 9
            geom("capsule", "witch_straw", size="0.018",
                 fromto=f"-0.44 0 -0.06 -0.76 {math.cos(angle) * .12} {-0.06 + math.sin(angle) * .09}")
        geom("ellipsoid", "witch_cloak", pos="-0.045 0 0.08", size="0.19 0.13 0.16")
        geom("ellipsoid", "witch_cloak", pos="0.015 0 0.19", size="0.10 0.09 0.14")
        geom("sphere", "witch_skin", pos="0.07 0 0.34", size="0.085")
        geom("capsule", "witch_skin", fromto="0.14 0 0.34 0.22 0 0.31", size="0.02")
        geom("cylinder", "witch_hat", pos="0.055 0 0.405", size="0.19 0.012")
        geom("mesh", "witch_hat", mesh="witch_pointed_hat", pos="0.055 0 0.41")
        geom("cylinder", "witch_band", pos="0.055 0 0.425", size="0.135 0.018")
        for side in (-1, 1):
            geom("sphere", "witch_hat", pos=f"0.142 {side * .04} 0.365", size="0.012")
            geom("capsule", "witch_cloak", fromto=f"0.03 {side * .08} 0.24 0.22 {side * .09} 0.06", size="0.034")
            geom("capsule", "witch_skin", fromto=f"0.22 {side * .09} 0.06 0.30 {side * .025} -0.04", size="0.025")
            geom("capsule", "witch_cloak", fromto=f"0 {side * .08} 0.02 0.13 {side * .14} -0.15", size="0.04")
            geom("capsule", "witch_hat", fromto=f"0.13 {side * .14} -0.15 0.02 {side * .15} -0.30", size="0.029")
            geom("ellipsoid", "witch_hat", pos=f"0.075 {side * .15} -0.31", size="0.085 0.04 0.035")


def animate_witches(model, data):
    updated = False
    for index, name in enumerate(WITCH_NAMES):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        mocap = int(model.body_mocapid[body])
        assert mocap >= 0 and model.body_jntnum[body] == 0
        phase = .65 * data.time + index * math.tau / 3
        side = -1 if index % 2 else 1
        data.mocap_pos[mocap] = (2.5 + 3.2 * math.sin(phase), side * (1.6 + .6 * math.cos(phase)),
                                1.25 + .12 * index + .15 * math.sin(.9 * data.time + index))
        heading = math.atan2(-side * .6 * math.sin(phase), 3.2 * math.cos(phase))
        data.mocap_quat[mocap] = (math.cos(heading / 2), 0., 0., math.sin(heading / 2))
        updated = True
    if updated:
        mujoco.mj_kinematics(model, data)
