"""Test the BaseChef class."""

from pathlib import Path
from typing import List

import pytest

from chonkie.chef.base import BaseChef


class TestChef(BaseChef):

    """Test implementation of BaseChef."""

    def clean(self, text: str) -> str:
        """Clean the text."""
        return text


@pytest.fixture
def test_chef():
    """Create a test chef instance with .txt and .md extensions."""
    return TestChef(extensions=[".txt", ".md"])


@pytest.fixture
def tmp_path() -> Path:
    """Create a temporary path."""
    return Path("tests/data")


@pytest.fixture
def temp_files(tmp_path) -> List[Path]:
    """Create temporary test files with different extensions."""
    # Create some test files
    files = [
        ("file1.txt", "Content 1"),
        ("file2.md", "Content 2"),
        ("file3.pdf", "Content 3"),  # Unsupported extension
        ("nested/file4.txt", "Content 4"),
        ("nested/deep/file5.md", "Content 5"),
    ]

    created_files = []
    for filepath, content in files:
        full_path = tmp_path / filepath
        # Ensure parent directories exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # Write content to file
        full_path.write_text(content)
        created_files.append(full_path)

    return created_files


def test_fetch_single_file(test_chef, temp_files):
    """Test fetching a single file."""
    result = test_chef.fetch(str(temp_files[0]))
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "Content 1"


def test_fetch_multiple_files(test_chef, temp_files):
    """Test fetching multiple files."""
    files = [str(temp_files[0]), str(temp_files[1])]
    result = test_chef.fetch(files)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result == ["Content 1", "Content 2"]


def test_fetch_directory(test_chef, tmp_path):
    """Test fetching all files from a directory recursively."""
    result = test_chef.fetch(directory=str(tmp_path))
    assert isinstance(result, list)
    assert len(result) == 4  # Should find all .txt and .md files, including nested ones
    assert "Content 1" in result
    assert "Content 2" in result
    assert "Content 4" in result
    assert "Content 5" in result


def test_fetch_unsupported_extension(test_chef, temp_files):
    """Test fetching a file with unsupported extension raises error."""
    with pytest.raises(ValueError, match="not a supported extension"):
        test_chef.fetch(str(temp_files[2]))  # file3.pdf


def test_fetch_missing_file(test_chef):
    """Test fetching a non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        test_chef.fetch("nonexistent.txt")


def test_fetch_missing_directory(test_chef):
    """Test fetching from a non-existent directory raises error."""
    with pytest.raises(FileNotFoundError):
        test_chef.fetch(directory="nonexistent_dir")


def test_fetch_no_inputs(test_chef):
    """Test fetch with no inputs raises error."""
    with pytest.raises(ValueError, match="Either files or directory must be provided"):
        test_chef.fetch()


def test_fetch_both_inputs(test_chef, temp_files, tmp_path):
    """Test fetch with both file and directory inputs raises error."""
    with pytest.raises(
        ValueError, match="Only one of files or directory can be provided"
    ):
        test_chef.fetch(files=str(temp_files[0]), directory=str(tmp_path))
