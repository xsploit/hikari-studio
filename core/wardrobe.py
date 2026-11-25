"""Wardrobe artwork tied to the original Hikari v28 meshes."""
from functools import cached_property
import json
from pathlib import Path

from core.paths import studio_root

ASSETS = studio_root() / "assets" / "wardrobe"
TOP_TEXTURE = 5
SHOE_TEXTURES = (6, 7)
WARDROBE_STYLES = {
    "outfitStyle": ("original", "hikari-stripe", "refined-vrm"),
    "shoeStyle": ("original", "d3-high-top"),
}


def _nodes(node):
    yield node
    for child in node.get("children", ()):
        yield from _nodes(child)


class Wardrobe:
    def __init__(self, original_header: bytes) -> None:
        self.original_header = original_header

    @cached_property
    def stripe_top(self) -> bytes:
        return (ASSETS / "hikari-stripe-top.png").read_bytes()

    @cached_property
    def refined_top(self) -> bytes:
        return (ASSETS / "hikari-vrm-top.png").read_bytes()

    @cached_property
    def high_tops(self) -> tuple[bytes, bytes]:
        return tuple((ASSETS / f"high-top-{side}.png").read_bytes() for side in ("r", "l"))

    @cached_property
    def raised_shoe_header(self) -> bytes:
        changes = json.loads((ASSETS / "high-top-meshes.json").read_text(encoding="utf-8"))
        model = json.loads(self.original_header[12:])
        meshes = changes["meshes"]
        replacements = changes["values"]
        for node in _nodes(model["nodes"]):
            key = str(node["uuid"])
            if key in meshes:
                node["mesh"] = meshes[key]

        for param in model["param"]:
            key = str(param["uuid"])
            for binding in param["bindings"]:
                values = replacements.get(str(binding["node"]), {})
                if binding["param_name"] == "deform" and key in values:
                    binding["values"] = values[key]

        content = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.original_header[:8] + len(content).to_bytes(4, "big") + content
