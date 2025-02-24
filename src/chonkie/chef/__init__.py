"""A module for handling text files."""

from .base import BaseChef
from .patterns import ABBREVIATIONS, UnicodeReplacements
from .text import TextChef

__all__ = ["BaseChef", "TextChef", "ABBREVIATIONS", "UnicodeReplacements"]
