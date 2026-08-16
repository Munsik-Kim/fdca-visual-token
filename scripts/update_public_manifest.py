#!/usr/bin/env python3
"""Regenerate the non-self-referential public file manifest."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"release/PUBLIC_MANIFEST.json", "release/SHA256SUMS.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return (
        path.is_file()
        and ".git" not in path.parts
        and "build" not in path.parts
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and not (path.parent == ROOT / "release" and path.suffix == ".zip")
        and rel not in EXCLUDED
    )


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if included(path):
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema": "FDCA_PUBLIC_MANIFEST_V1",
        "version": "v1.0.0",
        "non_self_referential": True,
        "files": files,
    }
    release = ROOT / "release"
    release.mkdir(exist_ok=True)
    (release / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (release / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files),
        encoding="utf-8",
    )
    print(json.dumps({"pass": True, "files": len(files)}))


if __name__ == "__main__":
    main()
