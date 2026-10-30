import ast
import hashlib
import json
import operator
import re
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from prepare import ASSETS, ROOT, SUPPORTS


def value(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value(item, names) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {value(key, names): value(item, names) for key, item in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -value(node.operand, names)
    if isinstance(node, ast.BinOp):
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.Pow: operator.pow}
        return operations[type(node.op)](value(node.left, names), value(node.right, names))
    if isinstance(node, ast.Call):
        return {"call": ast.unparse(node.func), "kwargs": {item.arg: value(item.value, names) for item in node.keywords}}
    if isinstance(node, ast.JoinedStr) and all(isinstance(item, ast.Constant) for item in node.values):
        return "".join(item.value for item in node.values)
    return {"expression": ast.unparse(node)}


def assignments(source, selected=None):
    names = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if selected is not None and name not in selected:
                continue
            try:
                names[name] = value(node.value, names)
            except (KeyError, TypeError):
                names[name] = {"expression": ast.unparse(node.value)}
    return names


def matched(setting, name):
    if not isinstance(setting, dict):
        return setting
    matches = [amount for pattern, amount in setting.items() if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one parameter match for {name}: {matches}")
    return matches[0]


def main():
    repository = ROOT.parent / "g1_crawl"
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repository), *arguments])

    before = git("status", "--porcelain=v1")
    commit = git("rev-parse", "HEAD").decode().strip()
    prefix = "source/g1_crawl/g1_crawl/tasks/manager_based/g1_crawl/"
    sources = {}

    def read(path):
        content = git("show", f"{commit}:{path}")
        sources[path] = hashlib.sha256(content).hexdigest()
        return content.decode()

    reference = assignments(read(prefix + "g1.py"))["G1_CFG"]["kwargs"]
    runner = assignments(read("sim2sim_mj/run_sim2sim.py"), {"CONFIG", "PYTORCH_JOINT_ORDER", "TRAINING_DEFAULT_ANGLES"})
    scene = ET.fromstring(read("sim2sim_mj/scene.xml"))
    usd = read(reference["spawn"]["kwargs"]["usd_path"])
    usd_pointer = usd.startswith("version https://git-lfs.github.com/spec/v1")
    env_source = read(prefix + "g1_crawl_env_cfg.py")
    ppo_source = read(prefix + "agents/rsl_rl_ppo_cfg.py")
    reset_source = read(prefix + "mdp/events.py")
    reset_function = next(node for node in ast.parse(reset_source).body
                          if isinstance(node, ast.FunctionDef) and node.name == "reset_from_animation_frame")
    reset_calls = [ast.unparse(node.func) for node in ast.walk(reset_function) if isinstance(node, ast.Call)]
    classes = {}
    for source in (env_source, ppo_source):
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {item.targets[0].id: ast.unparse(item.value) for item in node.body
                                     if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)}
    model_path = ASSETS / "flat.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    home = model.key("home").qpos
    rows = []
    for index, joint in enumerate(model.actuator_trnid[:, 0]):
        name = model.joint(int(joint)).name
        groups = [group["kwargs"] for group in reference["actuators"].values()
                  if any(re.fullmatch(pattern, name) for pattern in group["kwargs"]["joint_names_expr"])]
        if len(groups) != 1:
            raise ValueError(f"Ambiguous actuator group for {name}")
        group = groups[0]
        rows.append({"joint": name, "reference_training_kp": matched(group["stiffness"], name),
                     "reference_training_kd": matched(group["damping"], name),
                     "reference_training_limit_nm": matched(group["effort_limit_sim"], name),
                     "current_kp": float(model.actuator_gainprm[index, 0]),
                     "current_kd": float(-model.actuator_biasprm[index, 2]),
                     "current_limit_nm": model.jnt_actfrcrange[joint].tolist(),
                     "reference_default_angle": runner["TRAINING_DEFAULT_ANGLES"][name],
                     "current_home_angle": float(home[model.jnt_qposadr[joint]])})
    policy_order = runner["PYTORCH_JOINT_ORDER"]
    current_order = [row["joint"] for row in rows]
    scene_actuators = [element.get("joint") for element in scene.find("actuator")]
    mapping = [current_order.index(name) for name in policy_order]
    assert len(mapping) == 23 and len(set(mapping)) == 23
    assert all(np.isclose(row["reference_training_limit_nm"], row["current_limit_nm"][1]) for row in rows)
    collision = {}
    for name in SUPPORTS:
        reference_geom = scene.find(f".//geom[@name='{name}']")
        collision[name] = {"reference_mujoco_xml": reference_geom.attrib if reference_geom is not None else None,
                           "current_size": model.geom(name).size.tolist()}
    assert git("status", "--porcelain=v1") == before
    report = {"reference_commit": commit, "source_sha256": sources,
              "animation_reset_implementation": {"start_line": reset_function.lineno, "end_line": reset_function.end_lineno,
                                                  "calls": reset_calls, "has_write_call": any("write" in name for name in reset_calls),
                                                  "finding": "Function samples frame_indices but does not apply the selected frame. Not evidence that the framework never resets."},
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "reference_usd_is_lfs_pointer": usd_pointer, "reference_usd_pointer": usd if usd_pointer else None,
              "reference_self_collision": reference["spawn"]["kwargs"]["articulation_props"]["kwargs"]["enabled_self_collisions"],
              "reference_xml_actuator_count": len(scene_actuators),
              "policy_joints_missing_from_reference_xml": [name for name in policy_order if name not in scene_actuators],
              "policy_order": policy_order, "policy_to_current_actuator_indices": mapping,
              "actuator_comparison": rows, "support_geometries": collision,
              "reference_active_class_assignments": classes,
              "reference_playback_gain_multipliers": {key: amount for key, amount in runner["CONFIG"].items() if "multiplier" in key},
              "reference_worktree_unchanged": True,
              "limitations": "Static source inspection without executing external Python. USD LFS contents not fetched. Reference XML is a playback model, not proof of training collision/inertial parity. Missing wrists are explicitly zero-filled/dropped by its runner. No policy transfer or hardware calibration claimed."}
    output = ROOT / "output/rl/reference_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("reference_commit", "reference_usd_is_lfs_pointer", "reference_self_collision", "reference_xml_actuator_count", "policy_joints_missing_from_reference_xml", "reference_worktree_unchanged")}, indent=2))
    for row in rows:
        if row["joint"].startswith("left_"):
            print(row["joint"], "reference/current kp", row["reference_training_kp"], row["current_kp"], "kd", row["reference_training_kd"], row["current_kd"])
    print(output)


if __name__ == "__main__":
    main()
