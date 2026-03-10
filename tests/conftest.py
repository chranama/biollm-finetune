import json
import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def set_pythonpath(repo_root):
    original = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(repo_root / "src")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original


@pytest.fixture(scope="session")
def samples_dir(repo_root) -> pathlib.Path:
    return repo_root / "tests" / "data"


@pytest.fixture(scope="session")
def sample_questions_path(samples_dir) -> pathlib.Path:
    return samples_dir / "sample_questions.jsonl"


@pytest.fixture(scope="session")
def sample_gold_path(samples_dir) -> pathlib.Path:
    return samples_dir / "sample_gold.json"
