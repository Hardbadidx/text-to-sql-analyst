"""Test dependency synchronization between pyproject.toml and requirements.txt.

Streamlit Community Cloud deploys from requirements.txt, while local and CI
environments install from pyproject.toml. This test ensures that all runtime
dependencies in pyproject.toml are present in requirements.txt (by normalized package name).
"""

import re
import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"


def normalize_package_name(raw_name: str) -> str:
    """Extract and normalize package name by stripping version specifiers and extras."""
    # Strip env markers, extras, and version operators
    base = re.split(r"[<>=!~;]", raw_name)[0].strip()
    # Strip any extra specification e.g. pkg[extra] -> pkg
    base = re.sub(r"\[.*?\]", "", base).strip()
    return base.lower().replace("_", "-")


def test_requirements_txt_contains_all_pyproject_runtime_dependencies():
    """Assert all runtime dependencies in pyproject.toml exist in requirements.txt."""
    assert PYPROJECT_PATH.exists(), f"Missing {PYPROJECT_PATH}"
    assert REQUIREMENTS_PATH.exists(), f"Missing {REQUIREMENTS_PATH}"

    with open(PYPROJECT_PATH, "rb") as f:
        pyproject = tomllib.load(f)

    runtime_deps_raw = pyproject["project"]["dependencies"]
    pyproject_packages = {normalize_package_name(dep) for dep in runtime_deps_raw}

    requirements_packages = set()
    with open(REQUIREMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            requirements_packages.add(normalize_package_name(stripped))

    missing = pyproject_packages - requirements_packages
    assert not missing, (
        f"Deployment drift! The following runtime dependencies from pyproject.toml "
        f"are missing in requirements.txt: {sorted(missing)}"
    )
