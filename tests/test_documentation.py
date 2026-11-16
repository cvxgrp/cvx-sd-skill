"""Keep the active skill routes and runnable reference recipes usable."""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_skill_reference_links_exist():
    sources = [ROOT / "SKILL.md", *sorted((ROOT / "reference").glob("*.md"))]
    for source in sources:
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            assert (source.parent / path).exists(), (source, target)


@pytest.mark.parametrize("name", ["time-axis", "periodic-and-time", "downstream"])
def test_reference_recipes(name):
    source = ROOT / "reference" / f"{name}.md"
    namespace = {"__name__": "__documentation__"}
    for block in re.findall(r"```python\n(.*?)```", source.read_text(), re.DOTALL):
        exec(compile(block, str(source), "exec"), namespace)
