"""Structure and narrative contracts for the exact retained notebook inventory."""

import ast
from pathlib import Path

import nbformat
import pytest

from tests.notebook_smoke import NOTEBOOK_NAMES

ROOT = Path(__file__).absolute().parents[2]


@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebook_has_valid_structure_and_explanatory_contract(name):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown").lower()
    assert "prerequisites" in markdown
    assert "input" in markdown and "output" in markdown
    assert "interpretation" in markdown and "conclusion" in markdown
    assert "limitations" in markdown or "heuristics" in markdown
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert "sys.path" not in cell.source
            assert 'else Path(".")' not in cell.source
