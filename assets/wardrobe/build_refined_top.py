"""Fit the Refined VRM front bake and redraw to the original 2D halter."""
import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator

from build_wardrobe import STUDIO, WARDROBE, _unpack_inp, read_image, save_texture

# 2D texture landmarks followed by their counterparts on the VRM front bake.
VRM_LANDMARKS = [
    (269, 6, 289, 132), (367, 27, 355, 150), (476, 4, 421, 132),
    (209, 108, 218, 201), (533, 110, 492, 201),
    (193, 289, 226, 302), (370, 279, 355, 300), (550, 288, 483, 303),
    (144, 351, 225, 326), (373, 508, 355, 423), (595, 356, 485, 327),
    (11, 555, 137, 451), (724, 555, 573, 451),
    (65, 755, 160, 544), (678, 755, 548, 544),
    (82, 814, 175, 596), (654, 814, 537, 596),
    (101, 881, 180, 643), (369, 879, 355, 635), (638, 883, 528, 643),
]


def skin(rgb: np.ndarray) -> np.ndarray:
    r, g, b = np.moveaxis(rgb, -1, 0)
    return (r > 145) & (r > g + 10) & (g > b - 4) & (r > b + 16)


def skin_regions(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    warm = skin(rgb) & (alpha > .035)
    _, labels = cv2.connectedComponents(warm.astype(np.uint8))
    height, width = warm.shape
    points = [(80, 205), (653, 205), (370, 385), (370, 980)]
    regions = [int(labels[round(y * height / 1048), round(x * width / 740)]) for x, y in points]
    if not all(regions):
        raise ValueError("Could not locate the original skin islands")
    return np.isin(labels, regions)


def main() -> None:
    _, textures, _ = _unpack_inp(STUDIO / "models" / "Hikari-v28-BaseHD.inp")
    original = read_image(textures[5][1])
    height, width = original.shape[:2]
    alpha = original[..., 3:4] / 255
    original_rgb = original[..., :3] / np.maximum(alpha, 1/255)
    original_skin = skin_regions(original_rgb, alpha[..., 0])
    fabric = (alpha[..., 0] > .035) & ~original_skin
    distance = cv2.distanceTransform(fabric.astype(np.uint8), cv2.DIST_L2, 3)
    fit = np.clip((distance - 1) / 5, 0, 1)[..., None]

    redraw = Image.open(WARDROBE / "hikari-vrm-top-source.png").convert("RGBA")
    redraw = np.array(redraw.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)
    redraw_skin = skin_regions(redraw[..., :3], redraw[..., 3] / 255)
    usable_redraw = (redraw[..., 3] > 210) & ~redraw_skin
    redraw_weight = fit * usable_redraw[..., None]
    paint = original[..., :3] * (1 - redraw_weight) + redraw[..., :3] * alpha * redraw_weight

    # Project the actual texture back onto the 2D UVs. Use its graphically
    # distinct hem hardware and seam panels as the primary fabric surface.
    source = np.array(Image.open(WARDROBE / "refined-vrm-front.png").convert("RGBA"), dtype=np.float32)
    landmarks = np.array(VRM_LANDMARKS, dtype=np.float64)
    warp = RBFInterpolator(landmarks[:, :2], landmarks[:, 2:], kernel="thin_plate_spline", smoothing=.1)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    coords = np.column_stack((xx.ravel() * 740 / width, yy.ravel() * 1048 / height))
    mapping = warp(coords).reshape(height, width, 2).astype(np.float32)
    projected = cv2.remap(source, mapping[..., 0].copy(), mapping[..., 1].copy(), cv2.INTER_LINEAR)
    real_fabric = projected[..., 3] > 220
    baked_distance = cv2.distanceTransform(real_fabric.astype(np.uint8), cv2.DIST_L2, 3)
    bake_weight = fit * np.clip((baked_distance - 1) / 6, 0, 1)[..., None]
    paint = paint * (1 - bake_weight) + projected[..., :3] * alpha * bake_weight

    # The previous silver ring hangs past the VRM hem at the lower left.
    # Continue the clean redraw's edge through the ring without repainting skin.
    x = xx * 740 / width
    y = yy * 1048 / height
    edge = np.minimum(np.clip((x - 108) / 8, 0, 1), np.clip((223 - x) / 8, 0, 1))
    edge *= np.clip((y - 858) / 10, 0, 1) * np.clip((899 - y) / 8, 0, 1)
    edge *= fabric
    seam = redraw[round(851 * height / 1048), :, :3]
    paint = paint * (1 - edge[..., None]) + seam * alpha * edge[..., None]

    output = original.copy()
    output[..., :3] = paint
    protected = original_skin | (alpha[..., 0] < .035)
    assert np.array_equal(output[protected], original[protected])
    assert np.array_equal(output[..., 3], original[..., 3])
    save_texture(WARDROBE / "hikari-vrm-top.png", output)


if __name__ == "__main__":
    main()
