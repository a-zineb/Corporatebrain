"""Immutable pre-integration app.py reference used by parity harnesses."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "a125cfa"


def source() -> str:
    """Return the exact pre-integration production app source from Git."""

    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:app.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def tree() -> ast.Module:
    """Parse the immutable baseline source without importing Streamlit."""

    return ast.parse(source())


def top_level_function(name: str, namespace: dict[str, object]) -> object:
    """Compile one exact baseline top-level function with supplied dependencies."""

    function = next(
        node for node in tree().body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = copy.deepcopy(function)
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"{BASELINE_COMMIT}:app.py", "exec"), namespace)
    return namespace[name]


def assignment_nodes(name: str) -> list[ast.Assign]:
    """Return baseline assignments in source order for an exact variable name."""

    return sorted(
        [
            node
            for node in ast.walk(tree())
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ],
        key=lambda node: node.lineno,
    )


def execute_assignment(name: str, namespace: dict[str, object], occurrence: int = 0) -> object:
    """Execute one baseline assignment in an isolated, supplied namespace."""

    node = copy.deepcopy(assignment_nodes(name)[occurrence])
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"{BASELINE_COMMIT}:app.py", "exec"), namespace)
    return namespace[name]
