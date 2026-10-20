"""Display-only point glyphs. Never expand measured depth or modify RGB/XYZ."""
import cv2
import numpy as np


def point_overlay(image, pixels, colors, depths, *, size=1, shape='dot', opacity=.7):
    """Draw odd-sized disks or + markers in original Color-image pixels.

    Expanded glyphs may overlap: the nearer point is drawn last. Blend once,
    so overlap/density does not change opacity. Out-of-view centers are ignored.
    """
    if not isinstance(size, (int, np.integer)) or size not in range(1, 22, 2) or shape not in ('dot', 'cross'):
        raise ValueError('점 크기는 1~21의 홀수, 모양은 dot 또는 cross여야 합니다.')
    if not 0 <= opacity <= 1:
        raise ValueError('opacity는 0~1이어야 합니다.')
    pixels = np.asarray(pixels).reshape(-1, 2)
    colors = np.asarray(colors, np.uint8).reshape(-1, 3)
    depths = np.asarray(depths).reshape(-1)
    if not len(pixels) == len(colors) == len(depths):
        raise ValueError('투영 좌표·색·깊이 개수가 다릅니다.')
    height, width = image.shape[:2]
    good = np.isfinite(pixels).all(1) & np.isfinite(depths) & (depths > 0)
    good &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    good &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    ids = np.flatnonzero(good)
    xy = np.rint(pixels[ids]).astype(int)
    ids = ids[(xy[:, 0] < width) & (xy[:, 1] < height)]
    out = image.copy()
    if not len(ids):
        return out
    ids = ids[np.argsort(-depths[ids], kind='stable')]  # Far to near.
    xy = np.rint(pixels[ids]).astype(int)
    layer, mask = image.copy(), np.zeros((height, width), np.uint8)
    for (x, y), color in zip(xy.tolist(), colors[ids].tolist()):
        if shape == 'cross':
            cv2.drawMarker(layer, (x, y), color, cv2.MARKER_CROSS, size, 1, cv2.LINE_8)
            cv2.drawMarker(mask, (x, y), 1, cv2.MARKER_CROSS, size, 1, cv2.LINE_8)
        else:
            cv2.circle(layer, (x, y), size//2, color, -1, cv2.LINE_8)
            cv2.circle(mask, (x, y), size//2, 1, -1, cv2.LINE_8)
    drawn = mask != 0
    background = round(1-opacity, 15)  # Preserve the original .3/.7 blend rounding.
    out[drawn] = np.rint(image[drawn]*background+layer[drawn]*opacity).astype(np.uint8)
    return out
