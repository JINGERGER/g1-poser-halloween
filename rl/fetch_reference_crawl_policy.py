from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request


COMMIT = "935e8b9c857bd494b952603c0453800a0cf5c9d2"
EXPECTED = "1616790c7a1f56146d8a7ed07c32e414e76a5599a0d420f3811200fd36d9b1c8"
SIZE = 299360


def main():
    root = Path(__file__).resolve().parents[1]
    reference = root.parent / "g1_crawl"
    before = subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain=v1"])
    pointer = subprocess.check_output(["git", "-C", str(reference), "show", f"{COMMIT}:sim2sim_mj/policies/policy_crawl.pt"], text=True)
    assert f"oid sha256:{EXPECTED}" in pointer and f"size {SIZE}" in pointer
    destination = root / "output/rl/reference_assets" / f"g1_crawl_{COMMIT}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "policy_crawl.pt"
    url = f"https://media.githubusercontent.com/media/jloganolson/g1_crawl/{COMMIT}/sim2sim_mj/policies/policy_crawl.pt"
    if path.exists():
        contents = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "g1-poser-reference-audit"})
        with urllib.request.urlopen(request, timeout=30) as response:
            contents = response.read(SIZE + 1)
    assert len(contents) == SIZE and hashlib.sha256(contents).hexdigest() == EXPECTED
    if not path.exists():
        path.write_bytes(contents)
    after = subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain=v1"])
    assert before == after
    report = {"status": "downloaded_and_hash_verified", "commit": COMMIT, "url": url,
              "path": str(path), "sha256": EXPECTED, "bytes": SIZE,
              "verified_at": datetime.now(timezone.utc).isoformat(), "reference_worktree_unchanged": True,
              "limitations": "Artifact retrieval only; not deserialized, not simulated, not proven compatible with the audited component model or current controller."}
    (destination / "policy_download.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
