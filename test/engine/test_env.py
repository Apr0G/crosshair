"""
The engine venv is the one running and every Appendix E package imports (P0.1 test loop).

    .venv\\Scripts\\python.exe test\\engine\\test_env.py [--torch]

Exit 2 (not 1) when run from the wrong interpreter (Python < 3.12). torch is a SKIP by
default: it is installed separately (AMD ROCm wheels, P0.1.3) and its device check is
P0.1's job; `--torch` imports it in a subprocess and records the device. When
`requirements-engine.lock` exists, installed versions are compared against it.
"""
import importlib
import importlib.metadata
import subprocess
import sys

from _harness import ROOT, check, section, skip, summary

# (import name, distribution name) for Appendix E minus torch.
PACKAGES = (
    ("awpy", "awpy"), ("polars", "polars"), ("pyarrow", "pyarrow"), ("numpy", "numpy"),
    ("pandas", "pandas"), ("lightgbm", "lightgbm"), ("sklearn", "scikit-learn"),
    ("faiss", "faiss-cpu"), ("matplotlib", "matplotlib"), ("yaml", "pyyaml"),
    ("jsonschema", "jsonschema"), ("zstandard", "zstandard"), ("requests", "requests"),
)
LOCK = ROOT / "requirements-engine.lock"


def dist_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def locked_versions() -> dict[str, str]:
    """name -> version from `name==version` lines of the lock file (lowercased names)."""
    out = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            out[name.strip().lower().replace("_", "-")] = ver.strip()
    return out


def check_torch(want: bool) -> None:
    ver = dist_version("torch")
    if not want:
        skip("torch imports and sees the GPU", f"P0.1 check, run with --torch (installed: {ver})")
        return
    if ver is None:
        skip("torch imports and sees the GPU", "torch not installed")
        return
    code = ("import torch; ok = torch.cuda.is_available(); "
            "print(torch.__version__, ok, torch.cuda.get_device_name(0) if ok else None)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    check("torch imports", proc.returncode == 0, (proc.stdout or proc.stderr).strip()[-200:])


def main() -> int:
    section("interpreter")
    print(f"  {sys.executable}\n  Python {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        print("  wrong interpreter: the engine needs Python >= 3.12 (.venv\\Scripts\\python.exe)")
        return 2
    check("Python >= 3.12", sys.version_info >= (3, 12))

    section("Appendix E packages")
    for mod, dist in PACKAGES:
        try:
            importlib.import_module(mod)
            ok, why = True, ""
        except Exception as e:          # report every broken import, not just the first
            ok, why = False, f"{type(e).__name__}: {e}"
        check(f"import {mod}", ok, why or f"{dist} {dist_version(dist)}")
    check_torch("--torch" in sys.argv[1:])

    section("locked versions")
    if not LOCK.is_file():
        skip("installed versions match requirements-engine.lock", "lock file not frozen yet (P0.1.6)")
    else:
        locked = locked_versions()
        for _, dist in PACKAGES:
            want, have = locked.get(dist), dist_version(dist)
            if want is None:
                skip(f"{dist} locked", "not in the lock file")
            else:
                check(f"{dist} == {want}", have == want, f"installed {have}")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
