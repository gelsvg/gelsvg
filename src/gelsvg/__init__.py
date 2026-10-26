"""Read and write GelSVG documents (`.gel.svg`)."""

from .container import Container, PanelSlot, read_container
from .notices import Kind, Notice

__version__ = "0.0.1.dev0"

__all__ = ["Container", "Kind", "Notice", "PanelSlot", "read_container", "__version__"]
