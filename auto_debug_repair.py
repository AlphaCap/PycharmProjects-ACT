# Python
"""
Enhanced Auto Debug and Repair System (Comprehensive Mode)
- Detects and auto-repairs issues in Python files
- Added functionality to:
  • Detect and analyze runtime and logical issues.
  • Generate dynamic test cases for validating logic.
  • Perform runtime validation and self-healing corrections.
  • Integrate comprehensive error analysis and suggestions.
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
from traceback import format_exception
from typing import Any, Dict, List, Optional, Set, Tuple

# ------------ Logging ------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("auto_debug_repair.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------ Constants ------------
PROJECT_ROOT = Path(".").resolve()
STATE_FILE = PROJECT_ROOT / ".auto_debug_state.json"
DYNAMIC_TESTS_DIR = PROJECT_ROOT / "tests" / "auto_generated"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

ALLOW_STUBS = os.getenv("AUTOREPAIR_ALLOW_STUBS", "true").lower() in {
    "1",
    "true",
    "yes",
}

IMPORT_TO_PACKAGE_MAP: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pytest": "pytest",
    "matplotlib": "matplotlib",
    "requests": "requests",
    # Add more common mappings as needed
}


# ------------ Utilities ------------
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


def is_git_repo() -> bool:
    code, out, _ = run_cmd(["git", "rev-parse", "--is-inside-work-tree"])
    return code == 0 and out.strip() == "true"

def detect_changed_files() -> Generator[Path, None, None]:
    """
    Detect all Python files that have been modified or newly created.
    Returns a generator of Path objects.
    """
    try:
        # Check if we're in a Git repository
        if is_git_repo():
            # Use Git to find modified or new Python files
            code, output, _ = run_cmd(
                ["git", "diff", "--name-only", "--diff-filter=d", "HEAD"]
            )
            if code == 0:
                # Yield only Python files
                for line in output.splitlines():
                    path = PROJECT_ROOT / line.strip()
                    if path.suffix == ".py" and path.exists():
                        yield path
        else:
            # Fallback: Use file timestamps for non-Git-tracked projects
            logger.warning("Not a Git repository. Falling back to file timestamps.")
            for file in PROJECT_ROOT.rglob("*.py"):
                # Get files modified in the last 24 hours
                if file.stat().st_mtime > time.time() - 24 * 60 * 60:
                    yield file
    except Exception as e:
        logger.error(f"Failed to detect changed files: {e}")
        return []

def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env or os.environ.copy(),
        )
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"

def parse_python_imports(file_path: Path) -> Set[str]:
    try:
        content = file_path.read_text(encoding="utf-8")
        module_names = set()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names.add(node.module.split(".")[0])
        return module_names
    except Exception as e:
        logger.warning(f"Failed to parse imports in {file_path}: {e}")
        return set()


def pip_install(package: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", package]
    code, _, _ = run_cmd(cmd, check=False)
    return code == 0


def ensure_requirements(pkg_name: str) -> None:
    try:
        if not REQUIREMENTS_FILE.exists():
            with open(REQUIREMENTS_FILE, "w") as f:
                f.write("")
        with open(REQUIREMENTS_FILE, "r+") as f:
            content = f.read()
            if pkg_name not in content:
                f.write(f"{pkg_name}\n")
                logger.info(f"Added {pkg_name} to {REQUIREMENTS_FILE}")
    except Exception as e:
        logger.warning(f"Failed to update requirements.txt for {pkg_name}: {e}")


# ------------ Enhanced Repairs ------------
def detect_runtime_issues() -> None:
    """
    Globally capture runtime issues for analysis and repair.
    """

    def global_exception_handler(exc_type, exc_value, exc_traceback):
        error_message = "".join(format_exception(exc_type, exc_value, exc_traceback))
        logger.error(f"Unhandled Runtime Exception:\n{error_message}")
        repair_runtime_issues(error_message)

    sys.excepthook = global_exception_handler


def repair_runtime_issues(error_message: str) -> None:
    """
    Analyze and resolve runtime issues based on error messages.
    """
    if "ModuleNotFoundError" in error_message:
        missing_module = re.search(r"No module named '([^']+)'", error_message)
        if missing_module:
            module_name = missing_module.group(1)
            logger.info(f"Installing missing module: {module_name}")
            pip_install(module_name)
            ensure_requirements(module_name)
    elif "KeyError" in error_message:
        logger.warning(f"Suggesting fix for KeyError:\n{error_message}")
        # Optionally: Generate suggestions for missing keys.
    else:
        logger.warning(f"Unhandled runtime issue detected:\n{error_message}")


# ------------ Dynamic Test Generation ------------
def generate_dynamic_tests() -> None:
    """
    Generate dynamic tests for all modified or new Python functions.
    """
    modified_files = detect_changed_files()
    DYNAMIC_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    for file in modified_files:
        try:
            # Parse AST to extract function definitions
            content = file.read_text(encoding="utf-8")
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    test_case = generate_test_stub(node, file)
                    if test_case:
                        test_path = DYNAMIC_TESTS_DIR / f"test_{file.stem}.py"
                        with open(test_path, "a", encoding="utf-8") as f:
                            f.write(test_case)
                        logger.info(f"Generated test for {node.name} in {test_path}")
        except Exception as e:
            logger.warning(f"Failed to generate tests for {file}: {e}")


def generate_test_stub(func, file) -> Optional[str]:
    """
    Generate a test stub for a given function in a file.
    """
    test_name = f"test_{func.name}"
    params = [arg.arg for arg in func.args.args]
    param_str = ", ".join(params) if params else ""

    return f"""
def {test_name}():
    \"\"\"Auto-generated test for {func.name} from {file}\"\"\"
    try:
        result = {func.name}({param_str})
        # Add assertions based on logic
        assert result is not None, "Result should not be None"
    except Exception as ex:
        assert False, f"Test failed due to exception: {{str(ex)}}"
"""


# ------------ Main Auto-Repair Cycle ------------
def auto_repair_cycle() -> None:
    """
    Main enhanced repair workflow.
    """
    logger.info("Starting Comprehensive Auto-Repair Cycle")

    # Step 1: Run Static Code Fixers
    logger.info("Running static code formatters")
    run_cmd(["black", str(PROJECT_ROOT)])
    run_cmd(["isort", str(PROJECT_ROOT)])

    # Step 2: Generate Tests for Coverage
    logger.info("Generating dynamic tests")
    generate_dynamic_tests()

    # Step 3: Runtime Diagnostics
    logger.info("Monitoring runtime errors")
    detect_runtime_issues()

    # Step 4: Run Tests and Capture Failures
    logger.info("Running all tests")
    test_code, test_out, _ = run_cmd(["pytest", "--maxfail=5", str(DYNAMIC_TESTS_DIR)])
    if test_code != 0:
        logger.warning(f"Tests failed:\n{test_out}")

    logger.info("Auto-repair cycle completed.")


if __name__ == "__main__":
    auto_repair_cycle()
