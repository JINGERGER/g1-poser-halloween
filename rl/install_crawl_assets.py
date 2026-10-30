import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = "v0.1.0-halloween"
ARCHIVE = "crawl-assets-v0.1.0-halloween.zip"
SHA256 = "fcc6cd93d2e4160b7ba1edf4159e2d570d2d8a11c14605c5d9160d4af4791622"
URL = f"https://github.com/JINGERGER/g1-poser-halloween/releases/download/{RELEASE}/{ARCHIVE}"
MAX_BYTES = 150 * 1024 * 1024


def install(archive, root=ROOT):
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != SHA256:
        raise ValueError("Asset archive SHA256 mismatch; no files were installed")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or sum(info.file_size for info in bundle.infolist()) > MAX_BYTES:
            raise ValueError("Invalid or oversized asset archive")
        manifest = json.loads(bundle.read("manifest.json"))
        assert set(names) == set(manifest["files"]) | {"manifest.json"}
        verified = []
        for name, expected in manifest["files"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not name.startswith("output/rl/") or "\\" in name:
                raise ValueError(f"Unsafe asset path: {name}")
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f"Asset path leaves workspace: {name}")
            contents = bundle.read(name)
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError(f"Corrupt asset: {name}")
            if target.exists() and target.read_bytes() != contents:
                if name == "output/rl/short_step_assets/e5924df6cfa804e6/bundle.json":
                    packaged = json.loads(contents)
                    existing = target.read_bytes()
                    if (hashlib.sha256(existing).hexdigest() == packaged.get("source_bundle_sha256")
                            and json.loads(existing).get("initial_qpos") == packaged["initial_qpos"]):
                        continue
                raise FileExistsError(f"Refusing to overwrite different local asset: {name}. Use a clean checkout.")
            verified.append((target, contents))
        for target, contents in verified:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(contents)
    print(f"Verified and installed {len(verified)} assets for {RELEASE}")


def main():
    parser = argparse.ArgumentParser(description="Install checksum-pinned crawl assets without a neighboring reference checkout.")
    parser.add_argument("--archive", type=Path, help="Install a previously downloaded release archive offline.")
    args = parser.parse_args()
    if args.archive:
        install(args.archive)
        return
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / ARCHIVE
        request = urllib.request.Request(URL, headers={"User-Agent": "g1-poser-halloween-assets"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("Asset download exceeds size limit")
                output.write(chunk)
        install(archive)


if __name__ == "__main__":
    main()
