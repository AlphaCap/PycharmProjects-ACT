# Python
"""
Auto Debug and Repair System (Manual Commit Mode)
- Detects newly created or modified Python files
- Auto-fixes:
  • Ensures package structure (__init__.py)
  • Inserts package docstrings in __init__.py to satisfy D104
  • Resolves import errors by installing missing packages and updating requirements.txt
  • Removes simple Git merge conflict markers (keeps first side)
  • Formats code via Black/Isort (if available)
  • Runs mypy (if available) and retries basic fixes on invalid syntax
  • Runs pytest (if available) and optionally stubs missing names to unblock CI
- Outputs "Repairs made." or "No repairs needed."
- Does not auto-stage, commit, or push changes.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ------------- Logging -------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("auto_debug_repair.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------- Constants -------------
PROJECT_ROOT = Path(".").resolve()
STATE_FILE = PROJECT_ROOT / ".auto_debug_state.json"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

ALLOW_STUBS = os.getenv("AUTOREPAIR_ALLOW_STUBS", "true").lower() in {"1", "true", "yes"}

IMPORT_TO_PACKAGE_MAP: Dict[str, str] = {
    "pkg_resources": "setuptools",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "Image": "pillow",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "pandas_market_calendars": "pandas_market_calendars",
    "polygon": "polygon-api-client",
    "requests": "requests",
    "urllib3": "urllib3",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "tornado": "tornado",
    "protobuf": "protobuf",
    "pyparsing": "pyparsing",
    "pytz": "pytz",
    "scipy": "scipy",
    "six": "six",
}

# ------------- Utilities -------------
def read_state() -> Dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def write_state(state: Dict) -> None:
    try:
        content = json.dumps(state, indent=2) + "\n"
        STATE_FILE.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write state file: {e}")

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = False, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    try:
        res = subprocess.run(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env or os.environ.copy(),
        )
        if check and res.returncode != 0:
            logger.debug(res.stdout)
            logger.debug(res.stderr)
            raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)
        return res.returncode, res.stdout, res.stderr
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"

def is_git_repo() -> bool:
    code, out, _ = run_cmd(["git", "rev-parse", "--is-inside-work-tree"])
    return code == 0 and out.strip() == "true"

def git_status_porcelain() -> str:
    code, out, _ = run_cmd(["git", "status", "--porcelain"])
    return out if code == 0 else ""

def get_changed_files_since_last_run() -> Set[Path]:
    changed: Set[Path] = set()
    last_ts = read_state().get("last_run_ts", 0)

    if is_git_repo():
        _, out1, _ = run_cmd(["git", "ls-files", "--modified", "*.py"])
        _, out2, _ = run_cmd(["git", "diff", "--name-only", "--cached", "*.py"])
        _, out3, _ = run_cmd(["git", "ls-files", "--others", "--exclude-standard", "*.py"])

        for out in (out1, out2, out3):
            for line in out.splitlines():
                p = (PROJECT_ROOT / line.strip()).resolve()
                if p.exists() and p.suffix == ".py":
                    changed.add(p)
    else:
        for p in PROJECT_ROOT.rglob("*.py"):
            try:
                if p.is_file() and p.stat().st_mtime > last_ts:
                    changed.add(p.resolve())
            except Exception:
                pass

    return changed

# ------------- Fixers -------------
def ensure_init_files() -> List[Path]:
    created: List[Path] = []
    dirs: Set[Path] = set(p.parent for p in PROJECT_ROOT.rglob("*.py") if p.is_file())
    for d in sorted(dirs):
        init_path = d / "__init__.py"
        if not init_path.exists():
            try:
                init_path.write_text("# Auto-created to ensure package imports\n", encoding="utf-8")
                created.append(init_path)
                logger.info(f"Created {init_path}")
            except Exception as e:
                logger.warning(f"Failed to create {init_path}: {e}")
    return created

def ensure_package_docstrings() -> List[Path]:
    updated: List[Path] = []
    for init_file in PROJECT_ROOT.rglob("__init__.py"):
        try:
            text = init_file.read_text(encoding="utf-8")
        except Exception:
            continue

        has_doc = False
        try:
            tree = ast.parse(text or "")
            if isinstance(tree, ast.Module) and tree.body:
                first = tree.body[0]
                if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Str):
                    has_doc = True
        except Exception:
            stripped = text.lstrip()
            has_doc = stripped.startswith('"""') or stripped.startswith("'''")

        if not has_doc:
            try:
                new_text = '"""Package initializer."""\n' + ("" if text.startswith("\n") else "\n") + text
                init_file.write_text(new_text, encoding="utf-8")
                updated.append(init_file)
                logger.info(f"Inserted package docstring into {init_file}")
            except Exception as e:
                logger.warning(f"Failed to insert docstring into {init_file}: {e}")
    return updated

def run_formatter_for_files(files: Set[Path]) -> None:
    if not files:
        return
    paths = [str(f) for f in files]
    code, _, _ = run_cmd(["isort", "--profile", "black", *paths])
    if code != 0:
        logger.info("isort not available or failed. Skipping import sort.")
    code, _, _ = run_cmd(["black", *paths])
    if code != 0:
        logger.info("black not available or failed. Skipping code formatting.")

def parse_imports_from_file(pyfile: Path) -> Set[str]:
    try:
        src = pyfile.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return set()

    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = (n.name or "").split(".")[0]
                if root:
                    modules.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root:
                    modules.add(root)
    return {m for m in modules if not m.startswith("_") and m not in {"."}}

def try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False

def ensure_requirements_entry(pkg_name: str) -> None:
    try:
        existing = REQUIREMENTS_FILE.read_text(encoding="utf-8") if REQUIREMENTS_FILE.exists() else ""
        pattern = re.compile(rf"^\s*{re.escape(pkg_name)}([<>=].*)?$", re.IGNORECASE | re.MULTILINE)
        if not pattern.search(existing):
            with REQUIREMENTS_FILE.open("a", encoding="utf-8") as f:
                f.write(f"\n{pkg_name}\n")
            logger.info(f"Added '{pkg_name}' to requirements.txt")
    except Exception as e:
        logger.warning(f"Failed to update requirements.txt for {pkg_name}: {e}")

def pip_install(pkg_name: str) -> bool:
    logger.info(f"Installing missing package: {pkg_name}")
    code, out, err = run_cmd([sys.executable, "-m", "pip", "install", pkg_name])
    if code == 0:
        return True
    logger.error(f"pip install failed for {pkg_name}: {err or out}")
    return False

def resolve_and_install_missing_packages(files: Set[Path]) -> List[str]:
    all_modules: Set[str] = set()
    for f in files:
        all_modules |= parse_imports_from_file(f)

    local_roots = {p.name for p in PROJECT_ROOT.glob("*") if p.is_dir()}
    missing: Set[str] = set()
    for mod in sorted(all_modules):
        if mod in local_roots:
            continue
        if not try_import(mod):
            missing.add(mod)

    installed: List[str] = []
    for mod in sorted(missing):
        pkg = IMPORT_TO_PACKAGE_MAP.get(mod, mod)
        if pip_install(pkg):
            ensure_requirements_entry(pkg)
            installed.append(pkg)
    return installed

def run_mypy_collect_output() -> Tuple[bool, str]:
    code, _, _ = run_cmd(["mypy", "--version"])
    if code != 0:
        logger.info("mypy not available. Skipping type check.")
        return True, "mypy not available"
    code, out, err = run_cmd(["mypy"])
    ok = code == 0
    output = (out or "") + ("\n" + err if err else "")
    if ok:
        logger.info("mypy passed")
    else:
        logger.warning("mypy failed")
    return ok, output

def fix_merge_conflicts_in_text(text: str) -> Tuple[str, bool]:
    if "<<<" not in text or ">>>" not in text:
        return text, False

    lines = text.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    changed = False
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            changed = True
            i += 1
            head_block: List[str] = []
            while i < len(lines) and not lines[i].startswith("======="):
                head_block.append(lines[i])
                i += 1
            while i < len(lines) and not lines[i].startswith(">>>>>>>"):
                i += 1
            i += 1
            out.extend(head_block)
        else:
            out.append(lines[i])
            i += 1
    return "".join(out), changed

def fix_merge_conflict_markers(pyfile: Path) -> bool:
    try:
        text = pyfile.read_text(encoding="utf-8")
    except Exception:
        return False
    new_text, changed = fix_merge_conflicts_in_text(text)
    if changed:
        try:
            pyfile.write_text(new_text, encoding="utf-8")
            logger.info(f"Removed merge conflict markers in {pyfile}")
            return True
        except Exception as e:
            logger.warning(f"Failed to write merge conflict fix into {pyfile}: {e}")
    return False

def run_pytest_and_collect_failures() -> Tuple[bool, str]:
    code, _, _ = run_cmd(["pytest", "--version"])
    if code != 0:
        logger.info("pytest not available. Skipping tests.")
        return True, "pytest not available"
    code, out, err = run_cmd(["pytest", "-q"])
    success = code == 0
    output = (out or "") + ("\n" + err if err else "")
    if success:
        logger.info("pytest passed")
    else:
        logger.warning("pytest failed")
    return success, output

def extract_missing_names_from_pytest_output(output: str) -> List[Tuple[Path, str]]:
    results: List[Tuple[Path, str]] = []
    file_re = re.compile(r'File "(.+?\.py)", line \d+, in .+')
    name_re = re.compile(r"NameError:\s+name '([A-Za-z_][A-Za-z0-9_]*)' is not defined")
    last_file: Optional[Path] = None
    for line in output.splitlines():
        mfile = file_re.search(line)
        if mfile:
            p = Path(mfile.group(1)).resolve()
            if p.exists():
                last_file = p
        mname = name_re.search(line)
        if mname and last_file:
            results.append((last_file, mname.group(1)))
            last_file = None
    return results

def insert_stub_if_missing(pyfile: Path, name: str) -> bool:
    try:
        src = pyfile.read_text(encoding="utf-8")
    except Exception:
        return False
    try:
        tree = ast.parse(src)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return False
            if isinstance(node, ast.ClassDef) and node.name == name:
                return False
    except Exception:
        pass
    stub = f"\n\ndef {name}(*args, **kwargs):\n    raise NotImplementedError('Auto-stubbed by auto_debug_repair')\n"
    try:
        pyfile.write_text(src + stub, encoding="utf-8")
        logger.info(f"Inserted stub '{name}' into {pyfile}")
        return True
    except Exception as e:
        logger.warning(f"Failed to insert stub '{name}' into {pyfile}: {e}")
        return False

# ------------- Orchestration -------------
def auto_repair_cycle(changed_files: Set[Path]) -> bool:
    before = git_status_porcelain()

    ensure_init_files()
    ensure_package_docstrings()
    target_files = changed_files or set(PROJECT_ROOT.rglob("*.py"))
    resolve_and_install_missing_packages(target_files)
    for f in target_files:
        fix_merge_conflict_markers(f)
    run_formatter_for_files(target_files)
    ok, mypy_out = run_mypy_collect_output()
    if not ok and "invalid syntax" in mypy_out:
        suspect_files: Set[Path] = set()
        for line in mypy_out.splitlines():
            m = re.search(r"^(.+?\.py):\d+: error: invalid syntax", line)
            if m:
                p = Path(m.group(1)).resolve()
                if p.exists():
                    suspect_files.add(p)
        if suspect_files:
            fixed_any = False
            for f in suspect_files:
                fixed_any |= fix_merge_conflict_markers(f)
            if fixed_any:
                run_formatter_for_files(suspect_files)
                run_mypy_collect_output()
    test_ok, test_out = run_pytest_and_collect_failures()
    if not test_ok and ALLOW_STUBS:
        stubs_added = False
        for fpath, missing_name in extract_missing_names_from_pytest_output(test_out):
            stubs_added |= insert_stub_if_missing(fpath, missing_name)
        if stubs_added:
            run_formatter_for_files({f for f, _ in extract_missing_names_from_pytest_output(test_out)})
            run_pytest_and_collect_failures()
    after = git_status_porcelain()
    return before != after

def main():
    changed_files = get_changed_files_since_last_run()
    changed = auto_repair_cycle(changed_files)
    if changed:
        print("Repairs made.")
    else:
        print("No repairs needed.")
    state = read_state()
    state["last_run_ts"] = time.time()
    state["last_run"] = datetime.now().isoformat()
    write_state(state)

if __name__ == "__main__":
    main()
