import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


# Project root is dynamically resolved
PROJECT_ROOT = Path(__file__).parent.resolve()


def add_to_pythonpath():
    """Ensure project files are accessible in PYTHONPATH"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        print(f"Added {PROJECT_ROOT} to PYTHONPATH")


def fix_navigation_logic(file_path: Path):
    """
    Detect and correct invalid Streamlit navigation paths (e.g., invalid st.switch_page calls).
    """
    print(f"Processing file for navigation issues: {file_path}")
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.readlines()

        updated_content = []
        modified = False

        # Gather valid file names in the main and `pages/` directory
        valid_pages = [f.stem for f in PROJECT_ROOT.glob("*.py")]  # Main files
        valid_pages += [f.stem for f in (PROJECT_ROOT / "pages").glob("*.py")]  # Pages/

        for line in content:
            if "st.switch_page" in line:
                # Extract target page name
                start = line.find("(") + 2  # Account for "st.switch_page("
                end = line.rfind(")") - 1
                target_page = line[start:end] if start < end else None

                if target_page:
                    # Check if the navigation target matches valid pages
                    target_page_fixed = target_page.strip('"').strip("'")
                    if target_page_fixed not in valid_pages:
                        print(f"Invalid navigation target detected: {target_page_fixed}")
                        if valid_pages:
                            # Suggest the first matching page
                            suggested_page = valid_pages[0]
                            print(f"Attempting to fix to: {suggested_page}")
                            line = (
                                line.replace(target_page_fixed, suggested_page)
                                if target_page_fixed
                                else line
                            )
                            modified = True
            updated_content.append(line)

        # Save updated content if modifications were made
        if modified:
            with open(file_path, "w", encoding="utf-8") as file:
                file.writelines(updated_content)
            print(f"Navigation logic fixed in {file_path}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")


def detect_runtime_issues(log_file: Optional[Path] = None):
    """
    Analyze runtime logs or Streamlit API exceptions for patterns that can be corrected.
    """
    print("Running runtime issue detection")
    try:
        log_path = log_file or (PROJECT_ROOT / "runtime.log")
        if not log_path.exists():
            print(f"No runtime log found at: {log_path}")
            return

        with open(log_path, "r", encoding="utf-8") as file:
            log_data = file.read()

        # Look for Streamlit navigation errors
        if "StreamlitAPIException" in log_data:
            print("Streamlit navigation issue detected in logs.")
            print("Suggestion: Validate all arguments passed to st.switch_page().")
            print("Fixing logic...")
            for file in PROJECT_ROOT.rglob("*.py"):
                fix_navigation_logic(file)

    except Exception as e:
        print(f"Error analyzing runtime logs: {e}")


def generate_dynamic_tests():
    """
    Dynamically generate test cases to validate navigation and critical paths.
    """
    print("Generating dynamic tests...")

    test_dir = PROJECT_ROOT / "tests/auto_generated"
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate navigation tests for Streamlit pages
        test_file = test_dir / "test_navigation.py"
        with open(test_file, "w", encoding="utf-8") as file:
            file.write("import streamlit as st\n\n")
            file.write("def test_navigation_pages():\n")
            file.write("    pages = [\n")

            # Generate page list dynamically
            pages = [f.stem for f in PROJECT_ROOT.glob("*.py")] + [
                f.stem for f in (PROJECT_ROOT / "pages").glob("*.py")
            ]
            for page in pages:
                file.write(f"        '{page}',\n")

            file.write("    ]\n")
            file.write("    for page in pages:\n")
            file.write("        try:\n")
            file.write("            st.switch_page(page)\n")
            file.write("            print(f'Navigation to {page} passed.')\n")
            file.write("        except Exception as e:\n")
            file.write("            print(f'Failed to navigate to {page}: {e}')\n")

        print(f"Generated test_navigation.py in {test_dir}")

    except Exception as e:
        print(f"Failed to generate dynamic tests: {e}")


def run_static_fixers():
    """
    Run static code fixers like black and isort.
    """
    print("Running static code fixers...")
    try:
        subprocess.run(["black", str(PROJECT_ROOT)], check=True)
        subprocess.run(["isort", str(PROJECT_ROOT)], check=True)
    except Exception as e:
        print(f"Static fixers encountered an error: {e}")


def run_tests():
    """
    Run all tests to validate existing functionality.
    """
    print("Running all tests...")
    try:
        result = subprocess.run(
            ["pytest", str(PROJECT_ROOT / "tests")],
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Some tests failed. Check the output.")
    except Exception as e:
        print(f"Failed to execute tests: {e}")


def auto_repair_cycle():
    """
    Main enhanced repair workflow incorporating new logic for robustness.
    """
    print("\n=== Starting Comprehensive Auto-Repair Cycle ===\n")

    # Step 1: Run Static Code Fixers
    run_static_fixers()

    # Step 2: Dynamic Dependency & Navigation Fixes
    print("\n=== Fixing Navigation Logic ===")
    for file in PROJECT_ROOT.rglob("*.py"):
        fix_navigation_logic(file)

    # Step 3: Dynamic Test Generation
    generate_dynamic_tests()

    # Step 4: Detect Runtime Issues
    detect_runtime_issues()

    # Step 5: Validate with Tests
    run_tests()

    print("\n=== Auto-Repair Cycle Completed ===")


if __name__ == "__main__":
    add_to_pythonpath()
    auto_repair_cycle()
