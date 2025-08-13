import os
import sys
import ast
import traceback
from pathlib import Path

# Configure your project root
PROJECT_ROOT = Path(__file__).parent.resolve()


def add_to_pythonpath():
    """Add project root to PYTHONPATH (if not already added)."""
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
                    imports.append(node.module)
            return imports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def fix_import(filepath):
    """
    Fix imports in a given file.
    """
    try:
        imports = parse_imports(filepath)
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        project_modules = {
            mod.name: mod for mod in PROJECT_ROOT.glob("**/*.py")
            if mod.name != "__init__.py" and mod.name != os.path.basename(filepath)
        }

        updated_lines = []
        modified = False

        # Rewrite imports
        for line in lines:
            if line.startswith("from .") or line.startswith("import ."):
                module_relative = line.split(" ")[1].strip()
                if module_relative.endswith(";"):
                    ...
            # Fix imports logic with AST,
