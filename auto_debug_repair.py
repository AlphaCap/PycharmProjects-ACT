# Python
"""
Auto Debug and Repair System (Hands-Free, Commit-Enabled)
- Triggers automatically via Git hooks (commit/push), not on file save
- Detects newly created or modified Python files
- Auto-fixes common issues:
  • Ensures package structure (__init__.py)
  • Resolves import errors by installing missing packages and updating requirements.txt
  • Formats code via Black/Isort (if available)
  • Runs tests via pytest (if available) and optionally stubs missing names to unblock CI
- Automatically stages and commits changes when possible

Notes:
- In pre-commit, hooks run before the commit is created. We stage changes and
  attempt to commit; if the environment blocks commit inside the hook, we still
  stage the changes so the next commit includes them. We also avoid infinite loops.
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
from dataclasses import dataclass
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

# Allow simple auto-stub insertion for missing names in tests
ALLOW_STUBS = os.getenv("AUTOREPAIR_ALLOW_STUBS", "true").lower() in {"1", "true", "yes"}

# Avoid recursive commits within this tool
AUTO_REPAIR_COMMIT_FLAG = os.getenv("AUTOREPAIR_MADE_COMMIT", "false").lower() in {"1", "true", "yes"}

# Map import names to pip packages (when names differ)
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
    "yfinance": "yfinance",
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
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
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


def in_pre_commit_env() -> bool:
    # Common env markers pre-commit sets
    return any(os.getenv(k) for k in ("PRE_COMMIT", "PRE_COMMIT_HOME", "PRE_COMMIT_COLOR"))


def git_status_porcelain() -> str:
    code, out, err = run_cmd(["git", "status", "--porcelain"])
    return out if code == 0 else ""


def git_has_changes() -> bool:
    return bool(git_status_porcelain().strip())


def git_stage_all() -> bool:
    code, _, err = run_cmd(["git", "add", "-A"])
    if code != 0:
        logger.error(f"git add failed: {err}")
        return False
    return True


def ensure_git_identity() -> None:
    # Ensure user.name and user.email exist for local repo
    code_name, out_name, _ = run_cmd(["git", "config", "--get", "user.name"])
    code_email, out_email, _ = run_cmd(["git", "config", "--get", "user.email"])
    if code_name != 0 or not out_name.strip():
        run_cmd(["git", "config", "user.name", "Auto Repair Bot"])
    if code_email != 0 or not out_email.strip():
        run_cmd(["git", "config", "user.email", "auto-repair@example.com"])


def git_commit_auto(message: Optional[str] = None) -> bool:
    """
    Try to create a commit for staged changes.
    If running inside pre-commit and the environment blocks commit, we still consider staging a success.
    """
    if not git_has_changes():
        return False

    ensure_git_identity()
    msg = message or f"[auto-repair] Apply fixes {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    # Prevent infinite hook loops by setting a one-off env var for this commit subprocess
    env = os.environ.copy()
    env["AUTOREPAIR_MADE_COMMIT"] = "true"

    code, out, err = run_cmd(["git", "commit", "-m", msg], env=env)
    if code == 0:
        logger.info("Auto-commit created.")
        return True

    # If commit not allowed in this context (pre-commit), we keep changes staged
    if in_pre_commit_env():
        logger.info("Commit blocked in hook environment. Staged changes will be included in the next commit.")
        return False

    # Other commit errors
    logger.warning(f"git commit failed: {err or out}")
    return False


def get_changed_files_since_last_run() -> Set[Path]:
    """
    Determine changed .py files:
    - If git available: use staged + unstaged + untracked
    - Else: use mtime > last_run_ts (from state), plus any new .py
    """
    changed: Set[Path] = set()
    last_ts = read_state().get("last_run_ts", 0)

    if is_git_repo():
        # Unstaged
        _, out1, _ = run_cmd(["git", "ls-files", "--modified", "*.py"])
        # Staged
        _, out2, _ = run_cmd(["git", "diff", "--name-only", "--cached", "*.py"])
        # Untracked
        _, out3, _ = run_cmd(["git", "ls-files", "--others", "--exclude-standard", "*.py"])

        for out in (out1, out2, out3):
            for line in out.splitlines():
                p = (PROJECT_ROOT / line.strip()).resolve()
                if p.exists() and p.suffix == ".py":
                    changed.add(p)
    else:
        # Timestamp-based detection
        for p in PROJECT_ROOT.rglob("*.py"):
            try:
                if p.is_file() and p.stat().st_mtime > last_ts:
                    changed.add(p.resolve())
            except Exception:
                pass

    return changed


def ensure_init_files() -> List[Path]:
    """
    Ensure __init__.py exists in any directory that contains .py files.
    """
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


def run_formatter_for_files(files: Set[Path]) -> None:
    """
    Run black and isort if available.
    """
    if not files:
        return
    paths = [str(f) for f in files]
    # isort
    code, _, _ = run_cmd(["isort", "--profile", "black", *paths])
    if code != 0:
        logger.info("isort not available or failed. Skipping import sort.")
    # black
    code, _, _ = run_cmd(["black", *paths])
    if code != 0:
        logger.info("black not available or failed. Skipping code formatting.")


def parse_imports_from_file(pyfile: Path) -> Set[str]:
    """
    Parse import module names from a Python file.
    Returns top-level module names (e.g., 'numpy' from 'numpy as np', or 'pkg_resources').
    """
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
    # Ignore relative or local same-project modules
    return {m for m in modules if not m.startswith("_") and m not in {"."}}


def try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def ensure_requirements_entry(pkg_name: str) -> None:
    """
    Append a package to requirements.txt if not present.
    """
    try:
        existing = ""
        if REQUIREMENTS_FILE.exists():
            existing = REQUIREMENTS_FILE.read_text(encoding="utf-8")
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
    """
    For imports in the given files, attempt to import; if missing, install.
    Returns list of packages installed.
    """
    all_modules: Set[str] = set()
    for f in files:
        all_modules |= parse_imports_from_file(f)

    # Filter out local modules (present in project)
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


def run_pytest_and_collect_failures() -> Tuple[bool, str]:
    """
    Run pytest if available; return (success, output).
    """
    code, _, _ = run_cmd(["pytest", "--version"])
    if code != 0:
        logger.info("pytest not available. Skipping tests.")
        return True, "pytest not available"

    code, out, err = run_cmd(["pytest", "-q"])
    success = code == 0
    output = (out or "") + "\n" + (err or "")
    if success:
        logger.info("pytest passed")
    else:
        logger.warning("pytest failed")
    return success, output


def extract_missing_names_from_pytest_output(output: str) -> List[Tuple[Path, str]]:
    """
    Very basic extraction: look for NameError: name 'X' is not defined in a file.
    Returns list of (file_path, missing_name).
    """
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
    """
    Insert a simple stub for a missing name into the module if it does not already exist.
    Stub raises NotImplementedError to make failure explicit but unblock imports/tests.
    """
    try:
        src = pyfile.read_text(encoding="utf-8")
    except Exception:
        return False

    # If name already defined, skip
    try:
        tree = ast.parse(src)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return False
            if isinstance(node, ast.ClassDef) and node.name == name:
                return False
    except Exception:
        # If parse fails, still try to append at EOF
        pass

    stub = f"\n\ndef {name}(*args, **kwargs):\n    raise NotImplementedError('Auto-stubbed by auto_debug_repair')\n"
    try:
        pyfile.write_text(src + stub, encoding="utf-8")
        logger.info(f"Inserted stub '{name}' into {pyfile}")
        return True
    except Exception as e:
        logger.warning(f"Failed to insert stub '{name}' into {pyfile}: {e}")
        return False


def auto_repair_cycle(changed_files: Set[Path]) -> bool:
    """
    Full repair cycle:
      1) Ensure package structure (__init__.py)
      2) Resolve imports -> install missing packages -> update requirements
      3) Format code (isort/black)
      4) Run tests (pytest). If failing with NameError, optionally stub missing names, re-run tests once.
    Returns True if any file changes were made (to working tree).
    """
    if not changed_files:
        logger.info("No new or modified Python files detected.")
    else:
        logger.info(f"Changed files: {', '.join(str(p.relative_to(PROJECT_ROOT)) for p in sorted(changed_files))}")

    before = git_status_porcelain()

    # 1) Ensure __init__.py
    ensure_init_files()

    # 2) Resolve and install missing packages (based on changed files, but might fix broader env)
    installed = resolve_and_install_missing_packages(changed_files or set(PROJECT_ROOT.rglob("*.py")))
    if installed:
        logger.info(f"Installed packages: {installed}")

    # 3) Format changed files (or whole tree if unknown)
    run_formatter_for_files(changed_files or set(PROJECT_ROOT.rglob("*.py")))

    # 4) Run tests
    success, output = run_pytest_and_collect_failures()
    if not success and ALLOW_STUBS:
        stubs_added = False
        for fpath, missing_name in extract_missing_names_from_pytest_output(output):
            if insert_stub_if_missing(fpath, missing_name):
                stubs_added = True
        if stubs_added:
            run_formatter_for_files({f for f, _ in extract_missing_names_from_pytest_output(output)})
            success2, _ = run_pytest_and_collect_failures()
            if success2:
                logger.info("pytest passed after stubbing.")

    after = git_status_porcelain()
    return before != after


def main():
    # Determine changed files (newly created or modified since last run)
    changed_files = get_changed_files_since_last_run()

    # Run repair cycle
    changed = auto_repair_cycle(changed_files)

    # Stage and commit automatically if there are changes
    if is_git_repo() and changed:
        # Always stage everything the tool changed
        if git_stage_all():
            # Try to commit, unless we already committed in this tool invocation
            if not AUTO_REPAIR_COMMIT_FLAG:
                committed = git_commit_auto()
                if not committed and in_pre_commit_env():
                    # If commit blocked in hook, keep changes staged;
                    # pre-commit may require re-running commit, but staged changes are ready.
                    pass

    # Update last run timestamp
    state = read_state()
    state["last_run_ts"] = time.time()
    state["last_run"] = datetime.now().isoformat()
    write_state(state)


if __name__ == "__main__":
    main()
