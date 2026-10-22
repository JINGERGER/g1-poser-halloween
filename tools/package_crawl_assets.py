import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "935e8b9c857bd494b952603c0453800a0cf5c9d2"
ARCHIVE = "crawl-assets-v0.1.0-halloween.zip"


def main():
    files = {}
    model = ROOT / "output/rl/models/component_hulls_v1_12d607bfdb588900"
    reference = ROOT / "output/rl/reference_assets" / f"g1_crawl_{COMMIT}"
    for path in [model / "flat.xml", *sorted((model / "meshes").iterdir()),
                 reference / "scene.xml", reference / "crawl-pose.json", reference / "policy_crawl.pt",
                 reference / "reference_manifest.json", *sorted((reference / "meshes").iterdir())]:
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    source_path = reference / "run_sim2sim.py"
    license_path = ROOT / "output/rl/release_assets/g1_crawl_LICENSE.txt"
    source = source_path.read_bytes() if source_path.exists() else subprocess.check_output(
        ["git", "-C", str(ROOT.parent / "g1_crawl"), "show", f"{COMMIT}:sim2sim_mj/run_sim2sim.py"])
    license_text = license_path.read_bytes() if license_path.exists() else subprocess.check_output(
        ["git", "-C", str(ROOT.parent / "g1_crawl"), "show", f"{COMMIT}:LICENSE"])
    files[(reference / "run_sim2sim.py").relative_to(ROOT).as_posix()] = source
    files["output/rl/release_assets/g1_crawl_LICENSE.txt"] = license_text
    bundle = ROOT / "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json"
    initial = json.loads(bundle.read_text())
    files[bundle.relative_to(ROOT).as_posix()] = (json.dumps({
        "initial_qpos": initial["initial_qpos"],
        "source_bundle_sha256": initial.get("source_bundle_sha256", hashlib.sha256(bundle.read_bytes()).hexdigest()),
        "scope": "Frozen crawl runtime initial qpos only; research task assets are not included."}, indent=2) + "\n").encode()
    nominal = ROOT / "output/rl/crawl_candidate_verification/2026-10-22_14-45-04-443094/case_0_contact_0.005_dt_0.002_ankle_-0.2_boost_0.08.npz"
    if not nominal.exists():
        nominal = ROOT / "output/rl/release_assets/nominal_trajectory.npz"
    files["output/rl/release_assets/nominal_trajectory.npz"] = nominal.read_bytes()
    manifest = {"version": "v0.1.0-halloween", "reference_commit": COMMIT,
                "scope": "Flat-ground simulation preview; explicit PD/contact profile, not hardware or terrain certification.",
                "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in sorted(files.items())}}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = ROOT / "output/releases" / ARCHIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, contents in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {ARCHIVE}\n")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": checksum, "files": len(files)}))


if __name__ == "__main__":
    main()
