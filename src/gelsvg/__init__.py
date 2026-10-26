"""Read and write GelSVG documents (`.gel.svg`)."""

from .container import Container, PanelSlot, read_container
from .document import Assembly, Document, read
from .notices import Kind, Notice
from .panel import Panel

__version__ = "0.0.1.dev0"

__all__ = [
    "Assembly", "Container", "Document", "Kind", "Notice", "Panel", "PanelSlot",
    "read", "read_container", "__version__",
]
