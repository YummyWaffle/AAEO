"""Offline wheel/install smoke check; writes only beneath this distribution.

Uses dependencies already installed in the active Python runtime. This verifies
package isolation, not automatic dependency installation in a clean environment.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faiss", action="store_true", help="Also verify FAISS using the existing runtime")
    args = parser.parse_args()
    work = ROOT / ".work"
    reports = ROOT / "artifacts" / "validation"
    for directory in (work / "tmp", work / "external-cwd", work / "matplotlib", ROOT / "dist", reports):
        directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", TEMP=str(work / "tmp"), TMP=str(work / "tmp"),
                       TMPDIR=str(work / "tmp"), MPLCONFIGDIR=str(work / "matplotlib"),
                       PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1",
                       PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    environment.pop("PYTHONPATH", None)
    prefix = Path(sys.prefix)
    for variable, directory in (("GDAL_DATA", "gdal"), ("PROJ_DATA", "proj")):
        candidate = prefix / "Library" / "share" / directory
        if candidate.is_dir():
            environment[variable] = str(candidate)

    def run(label, arguments, cwd=ROOT):
        completed = subprocess.run([sys.executable, "-I", "-B", *arguments], cwd=cwd,
                                   env=environment, capture_output=True, text=True)
        (reports / (label + ".log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            print(completed.stdout + completed.stderr)
            raise RuntimeError(f"{label} failed; see artifacts/validation/{label}.log")
        print(f"{label}: passed", flush=True)
        return completed.stdout

    run("build", ["-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"])
    wheel = ROOT / "dist" / "aaeo_tlg-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        allowed = {"tlg_api", "tlg_core", "aaeo_tlg-0.3.0.dist-info"}
        assert all(name.split("/")[0] in allowed for name in members), members
        assert not any("tlg_data" in name for name in members)
        manifest = json.loads((ROOT / "docs" / "source_manifest.json").read_text(encoding="utf-8"))
        source_files = [item for item in manifest["files"] if item["destination"].startswith(("tlg_api/", "tlg_core/"))]
        assert {name for name in members if name.endswith(".py")} == {item["destination"] for item in source_files}
        for item in source_files:
            assert hashlib.sha256(archive.read(item["destination"])).hexdigest() == item.get("distribution_sha256", item["sha256"])

    installed = work / "installed"
    run("install", ["-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                    "--no-cache-dir", "--upgrade", "--target", str(installed), str(wheel)])
    # -I excludes the current directory, PYTHONPATH and the user site. In addition,
    # block the unpublished data package, and assert both imported package paths.
    smoke = r'''
import importlib.abc, json, pathlib, runpy, sys
installed, example, output, extractor = sys.argv[1:]
class NoDataPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tlg_data" or fullname.startswith("tlg_data."):
            raise ModuleNotFoundError("tlg_data is intentionally unavailable", name=fullname)
sys.meta_path.insert(0, NoDataPackage())
sys.path.insert(0, installed)
import tlg_api, tlg_core
for module in (tlg_api, tlg_core):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path(installed).resolve()), module.__file__
grid = tlg_core.GridSpec("EPSG:32611", (500000, 30, 0, 4100000, 0, -30), 64, 64)
assert grid.same_grid(grid)
assert tlg_core.BandSpec("red", "image", 2).index == 2
print(json.dumps({"tlg_api": tlg_api.__file__, "tlg_core": tlg_core.__file__}))
sys.argv = [example, "--output", output, "--extractor", extractor]
runpy.run_path(example, run_name="__main__")
assert not any(name == "tlg_data" or name.startswith("tlg_data.") for name in sys.modules)
'''
    variants = ["minibatch", "faiss"] if args.faiss else ["minibatch"]
    results = {}
    for extractor in variants:
        output = ROOT / "artifacts" / ("installed-" + extractor)
        run("isolated-" + extractor, ["-c", smoke, str(installed), str(ROOT / "examples" / "minimal_example.py"),
                                    str(output), extractor], cwd=work / "external-cwd")
        results[extractor] = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "GDAL", "faiss-cpu", "setuptools", "wheel", "pip"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if args.faiss:
        versions["faiss_runtime"] = run("faiss-version", ["-c", "import faiss; print(faiss.__version__)"]).strip()
    report = {"python": sys.version, "platform": platform.platform(), "dependencies": versions,
              "wheel": wheel.name, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
              "wheel_members": members, "source_files_verified": len(source_files),
              "tlg_data_blocked": True, "import_origin": ".work/installed", "results": results,
              "limitation": "Uses existing dependency runtime; no clean dependency installation or real-data experiment tested."}
    (reports / "distribution_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Distribution validation passed; reports: artifacts/validation", flush=True)


if __name__ == "__main__":
    main()
