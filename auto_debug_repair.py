"""
Auto Debug and Repair System
Scans repository for modified files and their dependencies, performs analysis,
and applies automatic fixes where appropriate.
"""

from __future__ import annotations
import os
import ast
import logging
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("auto_debug_repair.log"),  # Save log to a file
        logging.StreamHandler(),  # Print log to console
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a detected code issue."""
    file: str
    line: Optional[int]
    column: Optional[int]
    message: str
    severity: str
    auto_fixable: bool = False

class CodeAnalyzer:
    """Analyzes Python files and detects issues."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()

    def scan_modified_files(self) -> List[str]:
        """Scans and finds modified Python files in the repository."""
        logger.info("Scanning for modified files...")
        # This example assumes all `.py` files are modified. Adjust logic as needed.
        python_files = [str(file) for file in self.root_path.glob("**/*.py") if file.is_file()]
        logger.info(f"Found {len(python_files)} Python files.")
        return python_files

    @staticmethod
    def analyze_file(file_path: str) -> List[CodeIssue]:
        """Performs analysis on a single Python file and detects issues."""
        issues = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            logger.info(f"Analyzing file: {file_path}")

            # Example issue: Check for lines exceeding 120 characters
            for i, line in enumerate(code.splitlines(), start=1):
                if len(line) > 120:
                    issues.append(CodeIssue(
                        file=file_path,
                        line=i,
                        column=None,
                        message=f"Line exceeds 120 characters (found {len(line)})",
                        severity="warning",
                        auto_fixable=True,
                    ))

            # Example: Complexity analysis using AST
            tree = ast.parse(code)
            issues.extend(CodeAnalyzer.detect_complexity(file_path, tree))

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")

        return issues

    @staticmethod
    def detect_complexity(file_path: str, tree: ast.AST) -> List[CodeIssue]:
        """Detects functions with high cyclomatic complexity."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = sum(isinstance(n, (ast.If, ast.For, ast.While)) for n in ast.walk(node))
                if complexity > 10:
                    issues.append(CodeIssue(
                        file=file_path,
                        line=node.lineno,
                        column=None,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        severity="warning",
                        auto_fixable=False,
                    ))
        return issues

    @staticmethod
    def find_dependencies(file_path: str) -> List[str]:
        """Finds imported files referenced in this file."""
        dependencies = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("import") or line.strip().startswith("from"):
                        # Attempt to resolve the module to a file
                        parts = line.split()
                        if len(parts) > 1:
                            module = parts[-1].split(".")[0] + ".py"
                            possible_path = str(Path(file_path).parent / module)
                            if os.path.isfile(possible_path):
                                dependencies.append(possible_path)
        except Exception as e:
            logger.warning(f"Failed to find dependencies for {file_path}: {e}")
        return dependencies


class AutoRepair:
    """Applies automatic fixes to files when possible."""

    def repair_issues(self, file_path: str, issues: List[CodeIssue]):
        """Fixes auto-fixable issues in a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            for issue in issues:
                if issue.auto_fixable:
                    # Example fix: Break long lines
                    if "Line exceeds 120 characters" in issue.message:
                        logger.info(f"Fixing line {issue.line} in {file_path}")
                        code = self.fix_long_line(code, issue.line)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(f"Repaired file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to repair {file_path}: {e}")

    @staticmethod
    def fix_long_line(code: str, line_number: int) -> str:
        """Splits long lines into multiple shorter lines."""
        lines = code.splitlines()
        if 1 <= line_number <= len(lines):
            line = lines[line_number - 1]
            split_index = line[:120].rfind(" ")
            if split_index != -1:
                lines[line_number - 1] = line[:split_index]
                lines.insert(line_number, line[split_index:])
        return "\n".join(lines)


@dataclass
class AnalysisReport:
    """Represents the results of analyzing a file."""
    file_path: str
    issues: List[CodeIssue]


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description="Auto Debug and Repair System")
    parser.add_argument(
        "--root", default=".", help="Root directory of the project"
    )
    parser.add_argument(
        "--repair", action="store_true", help="Enable auto-repair for fixable issues"
    )
    args = parser.parse_args()

    root_path = args.root
    repair_mode = args.repair

    analyzer = CodeAnalyzer(root_path)
    repairer = AutoRepair()

    # Scan for modified files
    modified_files = analyzer.scan_modified_files()

    # Analyze files and track dependencies
    reports: List[AnalysisReport] = []
    analyzed_files = set()

    for file_path in modified_files:
        if file_path in analyzed_files:
            continue  # Avoid duplicate analysis
        analyzed_files.add(file_path)

        # Analyze this file
        issues = analyzer.analyze_file(file_path)
        reports.append(AnalysisReport(file_path, issues))

        # Find and analyze dependencies
        dependencies = analyzer.find_dependencies(file_path)
        for dep in dependencies:
            if dep not in analyzed_files:
                dep_issues = analyzer.analyze_file(dep)
                reports.append(AnalysisReport(dep, dep_issues))
                analyzed_files.add(dep)

        # Repair issues in the file if repair mode is enabled
        if repair_mode and issues:
            auto_fixable_issues = [i for i in issues if i.auto_fixable]
            if auto_fixable_issues:
                repairer.repair_issues(file_path, auto_fixable_issues)

    # Print analysis summary
    for report in reports:
        logger.info(f"\nReport for {report.file_path}")
        logger.info("=" * 50)
        if not report.issues:
            logger.info("No issues found!")
        for issue in report.issues:
            logger.info(f" - {issue.message} (Line: {issue.line}, Severity: {issue.severity})")


if __name__ == "__main__":
    main()