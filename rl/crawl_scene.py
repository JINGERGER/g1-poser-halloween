from copy import deepcopy
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from crawl_witches import add_witches


HALLOWEEN = Path(__file__).resolve().parents[1] / "g1_description/halloween.xml"


def add_grassy_cemetery(assets, world):
    texture = assets.find("texture[@name='halloween_tiles']")
    texture.attrib.update(builtin="flat", rgb1="0.12 0.22 0.085", mark="random",
                          markrgb="0.22 0.32 0.12", random="0.18")
    assets.find("material[@name='halloween_floor']").attrib.update(texrepeat="12 12", reflectance="0")
    assets.find("texture[@name='halloween_sky']").attrib.update(rgb1="0.045 0.065 0.085", rgb2="0.25 0.29 0.25")
    for name, color in (("cemetery_grass", "0.19 0.30 0.10 1"), ("cemetery_grass_dark", "0.10 0.19 0.06 1")):
        ET.SubElement(assets, "material", name=name, rgba=color)
    world.find("geom[@name='floor']").set("rgba", "1 1 1 1")
    generator = random.Random(31)
    grass = ET.SubElement(world, "body", name="cemetery_grass", childclass="halloween_decoration")
    for cluster in range(180):
        position_x = generator.uniform(-2., 9.)
        position_y = generator.choice((-1., 1.)) * generator.uniform(.85, 4.)
        for blade in range(3):
            start_x = position_x + generator.uniform(-.045, .045)
            start_y = position_y + generator.uniform(-.045, .045)
            end_x = start_x + generator.uniform(-.025, .025)
            end_y = start_y + generator.uniform(-.025, .025)
            height = generator.uniform(.035, .095)
            ET.SubElement(grass, "geom", type="capsule", size="0.002",
                          fromto=f"{start_x} {start_y} 0.002 {end_x} {end_y} {height}",
                          material="cemetery_grass" if blade % 2 else "cemetery_grass_dark")
    template = world.find("body[@name='grave_left']")
    for row, position_y in enumerate((-2.1, 2.3, 3.7)):
        for column, position_x in enumerate((-.6, 1.2, 3., 4.8, 6.6, 8.4)):
            grave = deepcopy(template)
            grave.set("name", f"cemetery_grave_{row}_{column}")
            grave.set("pos", f"{position_x + generator.uniform(-.16, .16)} {position_y} 0")
            world.append(grave)
    for name in ("pumpkin_left", "pumpkin_right"):
        pumpkin = world.find(f"body[@name='{name}']")
        position = [float(value) for value in pumpkin.get("pos").split()]
        radius = float(pumpkin.find("geom").get("size").split()[2])
        height = position[2] - radius
        pedestal = ET.SubElement(world, "body", name=f"{name}_pedestal",
                                 pos=f"{position[0]} {position[1]} 0", childclass="halloween_decoration")
        ET.SubElement(pedestal, "geom", type="box", material="grave_stone",
                      pos=f"0 0 {height / 2}", size=f"0.29 0.32 {height / 2}")


def make_halloween_scene(base_model, destination):
    base_model, destination = Path(base_model).resolve(), Path(destination)
    root = ET.parse(base_model).getroot()
    assert not root.findall(".//include")
    compiler = root.find("compiler")
    compiler.set("meshdir", str((base_model.parent / compiler.get("meshdir", "")).resolve()))
    theme = ET.parse(HALLOWEEN).getroot()
    assets = root.find("asset")
    existing_names = {element.get("name") for element in assets}
    for element in theme.find("asset"):
        if element.get("name", "").startswith(("ramp_", "halloween_ramp")):
            continue
        assert element.get("name") not in existing_names
        assets.append(deepcopy(element))
    decoration = deepcopy(theme.find("./default/default[@class='halloween_decoration']"))
    decoration.find("geom").set("density", "0")
    root.find("default").append(decoration)
    world = root.find("worldbody")
    world.find("geom[@name='floor']").set("material", "halloween_floor")
    for element in theme.find("worldbody"):
        if element.tag == "light" or element.get("childclass") == "halloween_decoration":
            world.append(deepcopy(element))
    add_grassy_cemetery(assets, world)
    add_witches(assets, world)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination
