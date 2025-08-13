import ast
import os
import sys
import traceback
from pathlib import Path

# Configure your project root
PROJECT_ROOT = Path(__file__).parent.resolve()


def add_to_pythonpath():
    """Add project root to PYTHONPATH if not already added."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        print(f"✅ Added {PROJECT_ROOT} to PYTHONPATH")


def parse_imports(filepath):
    """
    Parse the AST of a Python file to extract top-level imports.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=filepath)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            return imports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def fix_import(filepath):
    """
    Attempt to fix imports in a given Python file.

    - Fixes relative imports to be absolute based on project structure.
    - Ignores invalid or malformed imports gracefully.
    """
    try:
        # Parse existing imports from the file
        existing_imports = parse_imports(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Gather all Python module paths in the project
        project_modules = {
            str(mod.relative_to(PROJECT_ROOT)).replace("/", ".").replace(".py", ""): mod
            for mod in PROJECT_ROOT.glob("**/*.py")
            if "__init__" not in mod.name
        }

        updated_lines = []
        modified = False

        for line in lines:
            # Only attempt a fix for lines starting with "from ." or "import ."
            if line.strip().startswith("from .") or line.strip().startswith("import ."):
                # Split safely and handle unexpected cases
                parts = line.strip().split()
                if len(parts) < 2:
                    updated_lines.append(line)  # Skip malformed lines
                    continue

                keyword, module_relative = parts[0], parts[1]
                module_relative = module_relative.lstrip(".")  # Remove leading dots
                abs_module = f"{PROJECT_ROOT.name}.{module_relative}"

                # If the relative module exists, fix the import
                if module_relative in project_modules:
                    if keyword == "from":
                        updated_line = line.replace(
                            f"from .{module_relative}", f"from {abs_module}"
                        )
                    elif keyword == "import":
                        updated_line = line.replace(
                            f"import .{module_relative}", f"import {abs_module}"
                        )
                    else:
                        updated_line = line  # Unknown keyword, keep as is
                    modified = True  # Mark as modified
                else:
                    # If the module does not exist, keep the line unchanged
                    updated_line = line
            else:
                updated_line = line  # Keep unmodified lines

            updated_lines.append(updated_line)

        # Save the updated file only if modifications were made
        if modified:
            with open(filepath, "w", encoding="utf-8") as file:
                file.writelines(updated_lines)
                print(f"✅ Fixed imports in {filepath}")

    except Exception as e:
        print(f"❌ Error fixing imports in {filepath}: {e}")
        traceback.print_exc()


def batch_fix_imports():
    """
    Iterate over Python files to detect and fix import issues.
    """
    print(f"Scanning Python files in {PROJECT_ROOT}...")
    for py_file in PROJECT_ROOT.glob("**/*.py"):
        if "venv" not in str(py_file):  # Skip files in virtual environments
            print(f"Processing file: {py_file}")
            fix_import(py_file)


if __name__ == "__main__":
    add_to_pythonpath()
    batch_fix_imports()
    print("\n✅ Completed fixing imports in all files.")
