# gelsvg

Read and write GelSVG (`.gel.svg`): gel and blot figures that carry their own
data. A GelSVG document is an ordinary SVG figure; each panel's rendered image
is a PNG that also holds the panel's JSON and its embedded channel images and
source metadata.

The format is defined by its specification, which is being drafted and will be
published with the first stable release. This library follows the
specification and does not define it.

Status: early development. Nothing here is stable yet, and the format may still
change before its first release.
