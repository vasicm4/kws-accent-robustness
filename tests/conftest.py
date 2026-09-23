import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fake_data import make_corpus  # noqa: E402


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return make_corpus(tmp_path)
