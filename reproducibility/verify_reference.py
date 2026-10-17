"""Check the released aggregate reports against their SHA-256 manifest."""
import hashlib
import json
from pathlib import Path


def verify(root=Path("reference_results")):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["reports"].items():
        path = root / f"{name}.json"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["released_sha256"]:
            raise ValueError(f"Reference report changed: {path}")
        print(f"OK {name}")
    return len(manifest["reports"])


if __name__ == "__main__":
    verify()
