"""Read and write GelSVG documents (`.gel.svg`)."""

from .container import Container, PanelSlot, read_container
from .document import Assembly, Document, read
from .notices import Kind, Notice
from .panel import Panel
from .writer import (Content, Saved, Tool, WriteError, WriteNote, assembly_only, image_element, new_assembly_id,
                     new_panel, rewrite_panel, save, sign)

__version__ = "0.0.1.dev0"

__all__ = [
    "Assembly", "Container", "Content", "Document", "Kind", "Notice", "Panel", "PanelSlot", "Saved", "Tool",
    "WriteError", "WriteNote", "assembly_only", "image_element", "new_assembly_id", "new_panel", "read",
    "read_container", "rewrite_panel", "save", "sign", "__version__",
]
