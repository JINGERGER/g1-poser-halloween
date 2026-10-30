import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from joint_shift_equilibrium import ShiftEquilibrium, replay
from load_transfer import physical_metrics
from prepare import ROOT, SUPPORTS
from target_load_ablation import digest


class ContactGeometry:
    def __init__(self, model, seed):
        self.model, self.data = model, mujoco.MjData(model)
        self.seed = np.array(seed)
        self.scale = np.r_[[.02, .02, .04], [.1 / np.sqrt(3)] * 3, [.35] * 23]
        self.ids = [model.geom(name).id for name in SUPPORTS]
        self.data.qpos[:] = seed
        mujoco.mj_forward(model, self.data)
        self.xy = self.data.geom_xpos[self.ids, :2].copy()
        self.normals = self.data.geom_xmat[self.ids[2:]].reshape(2, 3, 3)[:, :, 2].copy()
        self.deadline, self.calls, self.cache = float("inf"), 0, None

    def evaluate(self, values):
        if time.monotonic() > self.deadline:
            raise TimeoutError("Geometry 30s budget exhausted")
        if self.cache is not None and np.array_equal(values, self.cache[0]):
            return self.cache[1]
        self.calls += 1
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.seed
        mujoco.mj_integratePos(self.model, self.data.qpos, values * self.scale, 1)
        self.data.ctrl[:] = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]]
        mujoco.mj_forward(self.model, self.data)
        row = physical_metrics(self.model, self.data)
        floor = self.model.geom("floor").id
        unexpected = [{"pair": [self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name],
                       "distance_m": float(contact.dist)} for contact in self.data.contact
                      if not (floor in (contact.geom1, contact.geom2) and any(geom in self.ids for geom in (contact.geom1, contact.geom2)))]
        worst = min((entry["distance_m"] for entry in unexpected), default=0)
        joints = self.model.actuator_trnid[:, 0]
        angles = self.data.qpos[self.model.jnt_qposadr[joints]]
        gaps = np.array(row["gaps_m"])
        normals = self.data.geom_xmat[self.ids[2:]].reshape(2, 3, 3)[:, :, 2]
        constraints = np.r_[(gaps + .0006) / .001, (-.0004 - gaps) / .001,
                            (worst + .001) / .001,
                            1 - np.linalg.norm(self.data.geom_xpos[self.ids, :2] - self.xy, axis=1) / .02,
                            1 - np.linalg.norm(normals - self.normals, axis=1) / .02,
                            angles - self.model.jnt_range[joints, 0], self.model.jnt_range[joints, 1] - angles,
                            self.data.qpos[2] - .16, .65 - self.data.qpos[2]]
        result = {"qpos": self.data.qpos.copy(), "ineq": constraints, "gaps_m": gaps,
                  "unexpected_contacts": unexpected, "objective": float(np.mean(values ** 2))}
        self.cache = values.copy(), result
        return result

    def solve(self):
        self.deadline = time.monotonic() + 30
        last = np.zeros(self.model.nv)

        def callback(values):
            nonlocal last
            last = values.copy()

        try:
            result = minimize(lambda values: self.evaluate(values)["objective"], last, method="SLSQP",
                              bounds=[(-1, 1)] * self.model.nv,
                              constraints=[{"type": "ineq", "fun": lambda values: self.evaluate(values)["ineq"]}],
                              callback=callback, options={"maxiter": 100, "ftol": 1e-8, "eps": 1e-6})
            last, success, message = result.x, bool(result.success), str(result.message)
        except TimeoutError as error:
            success, message = False, str(error)
        self.deadline = float("inf")
        measured = self.evaluate(last)
        return {"passed": bool(np.min(measured["ineq"]) >= -1e-5), "solver_success": success, "message": message,
                "values": last.tolist(), "qpos": measured["qpos"].tolist(), "gaps_m": measured["gaps_m"].tolist(),
                "minimum_constraint": float(np.min(measured["ineq"])), "unexpected_contacts": measured["unexpected_contacts"],
                "evaluations": self.calls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_report", type=Path)
    args = parser.parse_args()
    candidate = json.loads(args.candidate_report.read_text())
    assert candidate["status"] == "completed"
    model_path = Path(candidate["model"])
    assert digest(model_path) == candidate["model_sha256"]
    assert digest(model_path.with_name("manifest.json")) == candidate["manifest_sha256"]
    manifest = json.loads(model_path.with_name("manifest.json").read_text())
    for name, expected in manifest["mesh_sha256"].items():
        assert digest(model_path.parent / "meshes" / name) == expected
    chain = candidate
    for unused in range(3):
        assert digest(chain["source_report"]) == chain["source_report_sha256"]
        chain = json.loads(Path(chain["source_report"]).read_text())
    assert digest(chain["equilibrium_report"]) == chain["equilibrium_report_sha256"]
    old_equilibrium = json.loads(Path(chain["equilibrium_report"]).read_text())["solution"]
    nominal_path = Path(chain["base_model"]).with_name("manifest.json")
    nominal = json.loads(nominal_path.read_text())["qpos"]
    seeds = [("old_nominal", nominal), ("old_equilibrium", old_equilibrium["qpos"])]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    destination = ROOT / "output/rl/component_initial_state" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    destination.mkdir(parents=True)
    report = {"status": "running", "source_report": str(args.candidate_report.resolve()), "source_report_sha256": digest(args.candidate_report),
              "model": str(model_path), "model_sha256": digest(model_path), "nominal_manifest": str(nominal_path),
              "nominal_manifest_sha256": digest(nominal_path), "training_updates": 0, "seeds": [],
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                (Path(__file__).name, "joint_shift_equilibrium.py", "load_transfer.py", "diagnose_short_crawl.py", "servo_force.py")},
              "protocol": {"seeds": 2, "restarts": 0, "geometry_budget_s_each": 30, "geometry_maxiter": 100,
                           "equilibrium_budget_s_each": 60, "hold_s_each": 3, "geometry_gap_band_m": [-.0006, -.0004]},
              "limitations": "Candidate-model diagnostic only. Adjacent hull filtering retained. Geometry/static states are initialized, not dynamically reached. Holds use native motor controls only; no claim of crawling or hardware feasibility."}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        for label, seed in seeds:
            entry = {"seed": label, "seed_qpos": seed, "geometry": ContactGeometry(model, seed).solve(), "equilibrium": None, "hold": None}
            report["seeds"].append(entry)
            if entry["geometry"]["passed"]:
                qpos = entry["geometry"]["qpos"]
                ctrl = np.clip(np.array(qpos)[7:] + np.array(old_equilibrium["ctrl"]) - np.array(old_equilibrium["qpos"])[7:],
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                entry["equilibrium"] = ShiftEquilibrium(model, qpos, ctrl).solve()
                if entry["equilibrium"]["passed"]:
                    entry["hold"] = replay(model, {}, entry["equilibrium"], "hypothetical_hold", destination / f"{label}_hold.npz")
                    entry["hold"]["trajectory_sha256"] = digest(entry["hold"]["trajectory"])
            print(json.dumps({"seed": label, "geometry": entry["geometry"]["passed"],
                              "equilibrium": entry["equilibrium"] and entry["equilibrium"]["passed"],
                              "hold": entry["hold"] and entry["hold"]["reached"]}), flush=True)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(destination / "report.json", flush=True)


if __name__ == "__main__":
    main()
