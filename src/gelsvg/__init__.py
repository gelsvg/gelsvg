"""Read and write GelSVG documents (`.gel.svg`)."""

from .container import Container, PanelSlot, read_container
from ._hashes import pixel_hash, rendered_hash
from .document import Assembly, Document, read
from .drawing import aligned_image, aligned_size, default_crop, render, to_channel, to_panel
from .notices import Kind, Notice
from .panel import Panel
from .sources import SourceError, SourceImage, read_source
from .writer import (Content, Saved, Tool, WriteError, WriteNote, assembly_only, image_element, new_assembly_id,
                     new_panel, rewrite_panel, save, sign)

__version__ = "0.0.1.dev0"

__all__ = [
    "Assembly", "Container", "Content", "Document", "Kind", "Notice", "Panel", "PanelSlot", "Saved", "Tool",
    "WriteError", "WriteNote", "aligned_image", "aligned_size", "assembly_only", "default_crop", "image_element",
    "new_assembly_id", "new_panel", "pixel_hash", "read", "read_container", "render", "rendered_hash",
    "rewrite_panel", "save", "sign", "to_channel", "to_panel", "SourceError", "SourceImage", "read_source",
    "__version__",
]
