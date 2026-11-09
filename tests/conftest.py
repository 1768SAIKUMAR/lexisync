import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_synthetic_pdfs import build_all  # noqa: E402


@pytest.fixture(scope="session")
def pdfs(tmp_path_factory):
    """All synthetic PDFs, generated once per test session."""
    return build_all(tmp_path_factory.mktemp("synthetic"))
