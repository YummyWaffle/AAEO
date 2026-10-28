"""Read-only checks against the source checkout; all reports stay in this folder."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
SOURCE = PARENT / "tlg_franky_reconstruct"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    records = {}
    for directory, dirs, files in os.walk(PARENT, followlinks=False):
        dirs[:] = [name for name in dirs
                   if Path(directory, name).resolve() != ROOT
                   and not Path(directory, name).is_symlink()
                   and not getattr(Path(directory, name), "is_junction", lambda: False)()]
        for name in files:
            path = Path(directory, name)
            stat = path.lstat()
            records[path.relative_to(PARENT).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    report_dir = ROOT / "artifacts" / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = report_dir / "original_inventory.json"
    if args.baseline:
        if baseline_path.exists():
            raise FileExistsError("Refusing to replace the original baseline")
        baseline_path.write_text(json.dumps(inventory(), indent=2), encoding="utf-8")
        print("Original file size/mtime inventory recorded (distribution folder excluded).")
        return
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = inventory()
    changed = sorted(name for name in before.keys() | after.keys()
                     if before.get(name) != after.get(name))
    manifest = json.loads((ROOT / "docs" / "source_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for item in manifest["files"]:
        if (sha256(SOURCE / item["source"]) != item["sha256"]
                or sha256(ROOT / item["destination"]) != item.get("distribution_sha256", item["sha256"])):
            mismatches.append(item["destination"])
    result = {"original_files_checked_by_size_and_mtime": len(before),
              "original_inventory_changes": changed,
              "copied_files_checked_by_sha256": len(manifest["files"]),
              "source_or_copy_hash_mismatches": mismatches}
    (report_dir / "source_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if changed or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
