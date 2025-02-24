"""Base class for all chefs."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union


class BaseChef(ABC):

    """Base class for all chefs.

    This class provides a base for all chefs. Chefs are used to process text
    and handle issues that might cause indigestion to Chonkie.

    Args:
        extensions: The supported extensions.

    """

    def __init__(self, extensions: Optional[List[str]] = None) -> None:
        """Initialize the BaseChef class.

        Args:
            extensions: The supported extensions.

        """
        self._extensions = extensions if extensions is not None else []

    def _read_file(self, file_path: str) -> str:
        """Read a single file and return its content."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if the file is a supported extension
        if not any(file_path.endswith(ext) for ext in self._extensions):
            raise ValueError(f"File {file_path} is not a supported extension")

        # Open and read the file
        with open(file_path, "r", encoding="utf-8") as f:
            return str(f.read())

    def _read_files(self, file_paths: List[str]) -> List[str]:
        """Read multiple files and return their contents."""
        return [self._read_file(f) for f in file_paths]

    def _get_files_from_directory(self, directory: str) -> List[str]:
        """Recursively get all files with supported extensions from directory.

        Args:
            directory: The directory to search in

        Returns:
            List of file paths that match supported extensions

        """
        matching_files = []

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if any(file.endswith(ext) for ext in self._extensions):
                    matching_files.append(file_path)

        return matching_files

    def fetch(
        self,
        files: Union[str, List[str], None] = None,
        directory: Union[str, None] = None,
    ) -> Union[str, List[str]]:
        """Fetch the files from the given directory or file paths.

        Reads the file or files from the given directory or file paths and
        returns the content of the file or files. When a directory is provided,
        it recursively searches for files with supported extensions.

        Args:
            files: The files to fetch.
            directory: The directory to fetch the files from.

        Returns:
            The content of the file or files.

        """
        if files is None and directory is None:
            raise ValueError("Either files or directory must be provided")
        if files is not None and directory is not None:
            raise ValueError("Only one of files or directory can be provided")

        # If dir is provided, recursively fetch all files with supported extensions
        if directory is not None:
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Directory not found: {directory}")

            files = self._get_files_from_directory(directory)
            
        # If single file, then convert to list
        if isinstance(files, str):
            files = [files]

        # Read the files
        return self._read_files(files)

    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean the text.

        This method is used to clean the text.

        Args:
            text: The text to clean.

        Returns:
            The cleaned text.

        """
        return text

    def process(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        files: Optional[Union[str, List[str]]] = None,
        directory: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Process the text or the files/dir.

        Args:
            texts: The text to process.
            files: The files to process.
            directory: The directory to process the files from.

        Returns:
            The processed text.

        """
        # Text and files are mutually exclusive
        if texts is not None and (files is not None or directory is not None):
            raise ValueError("Either text or files/dir must be provided, not both")

        # If files is provided, fetch the files
        if files is not None or directory is not None:
            texts = self.fetch(files, directory)

        # Clean the texts
        texts = [self.clean(text) for text in texts]

        # return the cleaned texts
        return texts

    def __call__(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        files: Optional[Union[str, List[str]]] = None,
        directory: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Call the chef."""
        return self.process(texts, files, directory)
