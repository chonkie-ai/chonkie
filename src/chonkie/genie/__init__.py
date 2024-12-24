"""Genie classes which are used for all generative tasks.

Specifically, the genie classes are being used for contextualizing the chunks 
via the ContextualRefinery class.
"""

from .base import BaseGenie
from .claude import ClaudeGenie


__all__ = [
    "BaseGenie",
    "ClaudeGenie"
]