"""Read and write GelSVG documents (`.gel.svg`)."""

from .container import Container, PanelSlot, read_container
from .document import Assembly, Document, read
from .notices import Kind, Notice
from .panel import Panel
from .writer import Rewrite, Saved, Tool, WriteError, WriteNote, save

__version__ = "0.0.1.dev0"

__all__ = [
    "Assembly", "Container", "Document", "Kind", "Notice", "Panel", "PanelSlot", "Rewrite", "Saved", "Tool",
    "WriteError", "WriteNote", "read", "read_container", "save", "__version__",
]
