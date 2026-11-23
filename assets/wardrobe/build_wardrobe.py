"""Fit the wardrobe source art to Hikari v28, preserving original skin and arm seams."""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator


WARDROBE = Path(__file__).resolve().parent
STUDIO = WARDROBE.parents[1]
ROOT = STUDIO.parent
sys.path.insert(0, str(STUDIO))
from core.texture_upscaler import _unpack_inp

REFINED = Path(r"D:\Waifu Studio\assets\characters\models\a4f45d37bdbb-hikari-refined.vrm")
MODEL = ROOT / "v28-inochi" / "Hikari-v28.inp"
SOURCE = WARDROBE / "stripe-and-high-tops-source.png"
SOURCE_CROPS = ((72, 28, 676, 883), (800, 124, 1223, 810), (1276, 124, 1700, 810))
TOP_POINTS = [
    (269, 6, 245, 6), (367, 27, 365, 27), (476, 4, 479, 4),
    (209, 108, 211, 108), (533, 110, 517, 113),
    (41, 156, 23, 159), (694, 154, 706, 164),
    (193, 289, 177, 293), (370, 279, 367, 282), (550, 288, 550, 295),
    (144, 351, 148, 373), (373, 508, 369, 528), (595, 356, 604, 378),
    (11, 555, 17, 556), (724, 555, 721, 553),
    (331, 651, 337, 659), (402, 650, 401, 655),
    (171, 831, 171, 844), (430, 727, 446, 733),
    (101, 881, 88, 898), (369, 879, 369, 891), (638, 883, 639, 900),
    (88, 999, 83, 1000), (647, 999, 643, 1005), (370, 1038, 375, 1039),
]


def refined_images(path: Path) -> dict[int, Image.Image]:
    raw = path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20:20 + length])
    start = 28 + length
    result = {}
    for index in (14, 16):
        view = document["bufferViews"][document["images"][index]["bufferView"]]
        offset = start + view.get("byteOffset", 0)
        result[index] = Image.open(io.BytesIO(raw[offset:offset + view["byteLength"]])).convert("RGBA")
    return result


def read_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")).astype(np.float32)


def save_texture(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(np.clip(np.rint(rgba), 0, 255).astype(np.uint8)).save(path)


def build_top(original_bytes: bytes, generated: Image.Image, real_fabric: Image.Image) -> None:
    original = read_image(original_bytes)
    height, width = original.shape[:2]
    reference = np.asarray(generated.resize((740, 1048), Image.Resampling.LANCZOS), dtype=np.float32)
    points = np.array(TOP_POINTS, dtype=np.float64)
    warp = RBFInterpolator(points[:, :2], points[:, 2:], kernel="thin_plate_spline", smoothing=.05)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    coords = np.column_stack([x.ravel() * 740 / width, y.ravel() * 1048 / height])
    uv = warp(coords).reshape(height, width, 2).astype(np.float32)
    painted = cv2.remap(reference, uv[..., 0].copy(), uv[..., 1].copy(), cv2.INTER_LINEAR)
    alpha = original[..., 3:4] / 255
    straight = np.clip(original[..., :3] / np.maximum(alpha, 1/255), 0, 255)

    def is_skin(rgb: np.ndarray) -> np.ndarray:
        r, g, b = rgb.transpose(2, 0, 1)
        return (r > 145) & (r > g + 10) & (g > b - 4) & (r > b + 16)

    # Protect every drawn skin pixel and feather into the existing ink outlines.
    fabric = (alpha[..., 0] > .03) & ~is_skin(straight)
    protected_edge = cv2.distanceTransform(fabric.astype(np.uint8), cv2.DIST_L2, 3)
    painted_fabric = (painted[..., 3] > 205) & ~is_skin(painted[..., :3])
    weight = np.clip((protected_edge - 1) / (width * .009), 0, 1) * painted_fabric

    # This sample comes from the Refined top's diffuse texture between its eyelets.
    swatch = np.asarray(real_fabric.crop((768, 1090, 1136, 1438)).convert("L"), dtype=np.float32)
    material = cv2.resize(swatch, (width, height), interpolation=cv2.INTER_LINEAR)
    material -= cv2.GaussianBlur(material, (0, 0), max(2, width * .015))
    luminance = painted[..., :3].mean(axis=2)
    painted[..., :3] += (material * .50 * (luminance < 110))[..., None]
    blend = weight[..., None]
    top = original.copy()
    top[..., :3] = original[..., :3] * (1 - blend) + np.clip(painted[..., :3], 0, 255) * alpha * blend
    save_texture(WARDROBE / "hikari-stripe-top.png", top)


def build_shoe(generated: Image.Image, real_shoe: Image.Image, output: Path) -> None:
    # Keep the largest shoe component. Generated islands outside it cannot
    # follow the edge of the leg when the ankle moves.
    rgba = np.array(generated.convert("RGBA"), dtype=np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((rgba[..., 3] > 103).astype(np.uint8))
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = cv2.dilate((labels == index).astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
    rgba[..., 3] *= region
    sample = np.asarray(real_shoe, dtype=np.float32)
    colors = sample[..., :3]
    cool = (sample[..., 3] > 250) & (colors[..., 0] > 67) & (colors[..., 2] - colors[..., 0] > 30)
    cyan = np.median(colors[cool], axis=0)
    cyan_chroma = cyan - cyan.mean()
    result = np.asarray(Image.fromarray(rgba).resize((440, 692), Image.Resampling.LANCZOS), dtype=np.float32).copy()
    rgb = result[..., :3]
    trim = np.clip((rgb[..., 2] - rgb[..., 0] - 22) / 33, 0, 1)[..., None]
    luminance = rgb.mean(axis=2, keepdims=True)
    rgb[:] = np.clip(rgb * (1 - trim * .35) + (luminance + cyan_chroma) * trim * .35, 0, 255)
    rgb *= result[..., 3:4] / 255
    save_texture(output, result)


def walk(node):
    yield node
    for child in node.get("children", []):
        yield from walk(child)


def build_meshes(header: bytes) -> None:
    document = json.loads(header[12:])
    new_meshes = {}
    new_values = {}
    for node in walk(document["nodes"]):
        if node["name"] not in {"ArtMeshFootwearR", "ArtMeshFootwearL"}:
            continue
        mesh = node["mesh"]
        old_uv = np.reshape(mesh["uvs"], (-1, 2))
        old_xy = np.reshape(mesh["verts"], (-1, 2))
        # Use the original mapping for ankle alignment. Extend the high-top up
        # by 22% with the heel held in place. A full UV grid fits the new edge.
        rows, columns = 7, 5
        u, v = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
        grid = np.column_stack([u.ravel(), v.ravel()])
        affine = np.linalg.lstsq(np.column_stack([old_uv, np.ones(len(old_uv))]), old_xy, rcond=None)[0]
        xy = np.column_stack([grid, np.ones(len(grid))]) @ affine
        sole = float(xy[:, 1].max())
        xy[:, 1] = sole + (xy[:, 1] - sole) * 1.22
        center = float((xy[:, 0].max() + xy[:, 0].min()) / 2)
        xy[:, 0] = center + (xy[:, 0] - center) * 1.045
        triangles = []
        for row in range(rows - 1):
            for col in range(columns - 1):
                a = row * columns + col
                b, c = a + 1, a + columns
                triangles.extend([a, b, c, b, c + 1, c])
        old_tri = old_xy[np.array(mesh["indices"][:3])]
        old_sign = np.linalg.det(old_tri[1:] - old_tri[0])
        if old_sign < 0:
            triangles = np.array(triangles).reshape(-1, 3)[:, [0, 2, 1]].ravel().tolist()
        new_meshes[str(node["uuid"])] = {
            **mesh, "uvs": grid.ravel().tolist(), "verts": xy.ravel().tolist(), "indices": triangles,
        }

        def move_key(offsets):
            values = np.asarray(offsets)
            linear = LinearNDInterpolator(old_uv, values)(grid)
            outside = np.isnan(linear[:, 0])
            linear[outside] = NearestNDInterpolator(old_uv, values)(grid[outside])
            posed = (xy + linear)[np.reshape(triangles, (-1, 3))]
            determinants = np.linalg.det(posed[:, 1:] - posed[:, :1])
            if not np.all(determinants * old_sign > 0):
                raise ValueError(f"Shoe deformation flips a triangle on {node['name']}")
            return linear.tolist()

        values = {}
        for parameter in document["param"]:
            for binding in parameter["bindings"]:
                if binding["node"] == node["uuid"] and binding["param_name"] == "deform":
                    values[str(parameter["uuid"])] = [
                        [move_key(key) for key in row] for row in binding["values"]
                    ]
        new_values[str(node["uuid"])] = values
    if len(new_meshes) != 2:
        raise ValueError("Expected both shoe parts")
    (WARDROBE / "high-top-meshes.json").write_text(
        json.dumps({"meshes": new_meshes, "values": new_values}, separators=(",", ":")), encoding="utf-8",
    )


def main() -> None:
    header, _, _ = _unpack_inp(MODEL)
    _, hd, _ = _unpack_inp(STUDIO / "models" / "Hikari-v28-BaseHD.inp")
    source = Image.open(SOURCE).convert("RGBA")
    top, right, left = [source.crop(region) for region in SOURCE_CROPS]
    refined = refined_images(REFINED)
    build_top(hd[5][1], top, refined[16])
    build_shoe(right, refined[14], WARDROBE / "high-top-r.png")
    build_shoe(left, refined[14], WARDROBE / "high-top-l.png")
    build_meshes(header)


if __name__ == "__main__":
    main()
