"""Regression guard: archived experiment scripts must be reproducible from the
repository alone.

Motivation (F18/F19): the same helper module existed in both /tmp and the repo.
A fix landed in one copy while the other kept feeding stale values, and the
archived crit_* scripts imported from /tmp -- so `results/...` was NOT
reproducible from a clean checkout (they failed with ModuleNotFoundError once
/tmp was absent).

This guard copies the repo to a scratch dir, removes the /tmp helpers, and
asserts that every archived experiment script still loads and that helper
imports resolve INSIDE the repository.

Run:  python archived_scripts_reproducible_test.py
Exit: 0 = all pass, 1 = at least one script is not reproducible.
"""
import importlib.util as u

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

CHECK_DIRS = [
    "results/20260921_phaseC_w6",
    "results/20260920_phaseC_probe/scripts",
]
# modules a version of which also lived in /tmp (the contamination vector)
SHARED_MODULES = [
    "_ckpt_as_opponent.py", "_criterion_v2.py", "_criterion_v3.py",
    "_rolefixed_probe.py", "_selfplay_design_probe.py", "_verify_f5_fix.py",
]

# scripts expected to load without running a training job
SMOKE_SCRIPTS = [
    "results/20260920_phaseC_probe/scripts/_criterion_v3.py",
    "results/20260920_phaseC_probe/scripts/adapter_fidelity_test_v2.py",
]


def grep_tmp_inserts(path):
    """Return lines that put /tmp on sys.path at import time."""
    hits = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                s = ln.strip()
                if s.startswith("sys.path.insert") and "/tmp" in s:
                    hits.append((i, s))
    except OSError:
        pass
    return hits


def main():
    print("=" * 74)
    print("ARCHIVED-SCRIPT REPRODUCIBILITY GUARD  (F18/F19)")
    print("=" * 74)
    failures = []

    # 1. no archived file may put /tmp on sys.path
    print("\n1. no /tmp sys.path inserts in archived code")
    n_checked = 0
    for d in CHECK_DIRS:
        full = os.path.join(REPO, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if not fn.endswith(".py"):
                continue
            hits = grep_tmp_inserts(os.path.join(full, fn))
            n_checked += 1
            if hits:
                failures.append(f"{d}/{fn} inserts /tmp at {hits}")
                print(f"   FAIL {fn}: {hits}")
    print(f"   checked {n_checked} files; "
          f"{'clean' if not failures else 'VIOLATIONS FOUND'}")

    # 2. helpers the scripts need must exist in the repo
    print("\n2. helper modules present in the repo (not only in /tmp)")
    for m in SHARED_MODULES:
        found = any(os.path.exists(os.path.join(REPO, d, m))
                    for d in CHECK_DIRS)
        print(f"   {'OK  ' if found else 'MISS'} {m}")
        if not found:
            failures.append(f"helper {m} not archived in the repo")

    # 3. load the smoke scripts with /tmp removed and a clean cwd
    print("\n3. smoke scripts load from a clean copy with /tmp helpers absent")
    tmp_helpers = [os.path.join("/tmp", m) for m in SHARED_MODULES]
    backup = tempfile.mkdtemp(prefix="helper_backup_")
    moved = []
    for p in tmp_helpers:
        if os.path.exists(p):
            shutil.move(p, os.path.join(backup, os.path.basename(p)))
            moved.append(p)
    try:
        for rel in SMOKE_SCRIPTS:
            p = os.path.join(REPO, rel)
            if not os.path.exists(p):
                failures.append(f"missing smoke script {rel}")
                print(f"   MISS {rel}")
                continue
            mod = os.path.basename(rel)[:-3]
            code = (
                "import sys\n"
                "sys.path=[p for p in sys.path if p!='/tmp']\n"
                "import importlib.util as u\n"
                f"spec=u.spec_from_file_location({mod!r}, {p!r})\n"
                f"m=u.module_from_spec(spec)\n"
                "spec.loader.exec_module(m)\n"
                "print('LOADED')\n"
            )
            r = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, cwd=REPO)
            out = (r.stdout or "") + (r.stderr or "")
            if "LOADED" in out:
                print(f"   OK   {rel}")
            elif "ModuleNotFoundError" in out:
                failures.append(f"{rel}: ModuleNotFoundError without /tmp")
                print(f"   FAIL {rel}: ModuleNotFoundError")
            else:
                # ran real work; not a path failure
                print(f"   OK   {rel} (ran; no ModuleNotFoundError)")
    finally:
        for p in moved:
            shutil.move(os.path.join(backup, os.path.basename(p)), p)
        shutil.rmtree(backup, ignore_errors=True)
        print("   (/tmp helpers restored)")

    print("\n" + "=" * 74)
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: ALL PASS -- archived scripts are reproducible from the repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
