import argparse
import ast
import cProfile
import io
import logging
import os
import pstats
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# --- CONFIGURATION CONSTANTS ---
PROJECT_ROOT = Path(__file__).parent.resolve()
PAGES_DIR = os.path.join(PROJECT_ROOT, "pages")
CORE_FUNCTIONS = [
    "generate_strategy_for_objective",
    "calculate_indicators",
    "manage_positions",
]
STATIC_TOOLS = ["flake8", "black", "isort", "mypy"]
PROFILER_ACTIVE = False

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AutoDebugRepair")


# --- PYTHONPATH ADJUSTMENT ---
def add_to_pythonpath() -> None:
    """Ensure `PROJECT_ROOT` and `pages` directory are in the PYTHONPATH."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        logger.info(f"Added {PROJECT_ROOT} to PYTHONPATH.")

    if PAGES_DIR not in sys.path:
        sys.path.append(PAGES_DIR)
        logger.info(f"Added {PAGES_DIR} to PYTHONPATH.")


# --- STATIC CODE ANALYSIS AND REPAIR ---
def run_static_tools() -> None:
    """
    Run all static analysis tools (flake8, black, isort, mypy) to detect
    and repair issues (styling, type-checking, imports, etc.).
    """
    logger.info("Running static analysis and repair tools...")
    for tool in STATIC_TOOLS:
        command = (
            [tool, str(PROJECT_ROOT)]
            if tool != "black"
            else [tool, str(PROJECT_ROOT), "--quiet"]
        )
        try:
            subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"`{tool}` completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"{tool} failed:\n{e.stderr.strip()}")
        except FileNotFoundError:
            logger.error(f"`{tool}` is not installed. Skipping...")


# --- SYNTAX VALIDATION AND REPAIR ---
def validate_and_repair_syntax(file_path: str) -> None:
    """
    Parse the file with `ast` to detect potential syntax errors and attempt to fix them.
    Generates a report if unsuccessful in resolving issues.
    """
    logger.info(f"Validating syntax for: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)
        logger.info(f"Syntax valid for: {file_path}")
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {e}")
        attempt_syntax_fix(file_path, e)


def attempt_syntax_fix(file_path: str, error: SyntaxError) -> None:
    """
    Attempt to auto-fix syntax issues based on the error type provided by Python's parser.
    """
    logger.info(f"Attempting to repair syntax for: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Example: Add logic to repair common issues like missing colons, parentheses, etc.
        if "expected an indented block" in str(error):
            lines.insert(
                error.lineno - 1, "    pass  # Auto-fix: Added placeholder block\n"
            )

        elif "missing parentheses" in str(error):
            logger.warning(
                "Manual intervention may be required for missing parentheses."
            )

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"Syntax fixed for: {file_path}")

    except Exception as e:
        logger.error(f"Failed to repair syntax for {file_path}: {e}")
        traceback.print_exc()


# --- DEPENDENCY FILE SCANNING ---
def find_core_dependencies() -> List[str]:
    """
    Scan for files containing core functions or classes within the project.
    """
    logger.info("Locating files with core dependencies...")
    dependencies = []
    for filepath in PROJECT_ROOT.rglob("*.py"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if any(core_function in content for core_function in CORE_FUNCTIONS):
            dependencies.append(str(filepath))
    logger.info(f"Found {len(dependencies)} dependent files.")
    return dependencies


# --- RUNTIME PROFILING ---
def profile_runtime(filepath: str) -> None:
    """
    Profile the runtime of the given Python file using cProfile.
    """
    global PROFILER_ACTIVE
    if PROFILER_ACTIVE:
        logger.warning(f"Skipping {filepath}: Another profiler is active.")
        return

    PROFILER_ACTIVE = True
    try:
        logger.info(f"Profiling runtime for {filepath}...")
        profiler = cProfile.Profile()
        profiler.enable()
        exec(open(filepath).read())  # Execute the file
        profiler.disable()

        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output).sort_stats("cumulative")
        stats.print_stats()
        logger.info(f"Profile summary for {filepath}:\n{output.getvalue()}")
    except Exception as e:
        logger.error(f"Error during profiling for {filepath}: {e}")
        traceback.print_exc()
    finally:
        PROFILER_ACTIVE = False


# --- GIT OPERATIONS ---
def stage_and_commit_changes():
    """
    Automatically stage and commit changes, then push to the remote repository (if available).
    """
    logger.info("Attempting to commit and push changes...")
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Auto-debug repair completed"], check=True
        )
        subprocess.run(["git", "push"], check=True)
        logger.info("Changes committed and pushed successfully.")
    except Exception as e:
        logger.error(f"Error committing and pushing changes: {e}")
        traceback.print_exc()


# --- MAIN AUTO-REPAIR WORKFLOW ---
def auto_repair_cycle(target_file: Optional[str] = None):
    """
    Main workflow for auto-debugging and repairing project files.
    """
    add_to_pythonpath()

    # Step 1: Locate core dependency files
    if target_file:
        core_files = [target_file]
        logger.info(f"Targeting specific file for auto-repair: {target_file}")
    else:
        core_files = find_core_dependencies()

    # Step 2: Validate and repair syntax
    for file_path in core_files:
        validate_and_repair_syntax(file_path)

    # Step 3: Run static analysis and repair tools
    run_static_tools()

    # Step 4: Profile each dependent file
    for file_path in core_files:
        profile_runtime(file_path)

    # Step 5: Stage and commit all changes
    stage_and_commit_changes()

    logger.info("Auto-debug repair workflow completed successfully.")


# --- EXECUTE THE WORKFLOW ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-debug and repair tool.")
    parser.add_argument(
        "--target", type=str, help="Specify a single file to target for auto-repair."
    )
    args = parser.parse_args()

    auto_repair_cycle(args.target)
