"""Menon 2007 / DDFAPD 的加速实现：算法不变，卷积改走 OpenCV float32。"""

from __future__ import annotations

import cv2
import numpy as np

_MASKS: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

_H_GREEN = np.array([[-0.25, 0.5, 0.5, 0.5, -0.25]], dtype=np.float32)
_V_GREEN = _H_GREEN.T
_K_DEC = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_K_DEC_T = _K_DEC.T
_K_B = np.array([[0.5, 0.0, 0.5]], dtype=np.float32)
_K_B_T = _K_B.T
_FIR = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32)
_FIR_T = _FIR.T

_PATTERN_ROWS = {
    "RGGB": ("RG", "GB"),
    "GRBG": ("GR", "BG"),
    "GBRG": ("GB", "RG"),
    "BGGR": ("BG", "GR"),
}


def _conv_h(x: np.ndarray, k: np.ndarray, border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    return cv2.filter2D(x, cv2.CV_32F, k, borderType=border)


def _conv_v(x: np.ndarray, k: np.ndarray, border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    return cv2.filter2D(x, cv2.CV_32F, k, borderType=border)


def bayer_masks(h: int, w: int, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (h, w, pattern)
    cached = _MASKS.get(key)
    if cached is not None:
        return cached
    rows = _PATTERN_ROWS[pattern]
    r = np.zeros((h, w), np.uint8)
    g = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            sl = (slice(y, None, 2), slice(x, None, 2))
            if ch == "R":
                r[sl] = 1
            elif ch == "G":
                g[sl] = 1
            else:
                b[sl] = 1
    _MASKS[key] = (r, g, b)
    return r, g, b


def menon2007_fast(
    cfa: np.ndarray,
    pattern: str = "RGGB",
    refining_step: bool = False,
) -> np.ndarray:
    """输入 HxW uint16/float，输出 HxWx3 float32，数值尺度与输入相同。"""
    src = np.ascontiguousarray(cfa, dtype=np.float32)
    h, w = src.shape
    R_m, G_m, B_m = bayer_masks(h, w, pattern)
    Rm = R_m.astype(bool)
    Gm = G_m.astype(bool)
    Bm = B_m.astype(bool)

    R = src * R_m
    G = src * G_m
    B = src * B_m

    green_h = _conv_h(src, _H_GREEN)
    green_v = _conv_v(src, _V_GREEN)
    G_H = np.where(Gm, G, green_h)
    G_V = np.where(Gm, G, green_v)

    C_H = np.where(Rm, R - G_H, 0.0)
    C_H = np.where(Bm, B - G_H, C_H)
    C_V = np.where(Rm, R - G_V, 0.0)
    C_V = np.where(Bm, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])

    d_H = cv2.filter2D(D_H, cv2.CV_32F, _K_DEC, borderType=cv2.BORDER_CONSTANT)
    d_V = cv2.filter2D(D_V, cv2.CV_32F, _K_DEC_T, borderType=cv2.BORDER_CONSTANT)

    mask_h = d_V >= d_H
    G = np.where(mask_h, G_H, G_V)
    M = mask_h

    R_row = np.any(Rm, axis=1)[:, None]
    B_row = np.any(Bm, axis=1)[:, None]
    at_g_redrow = Gm & R_row
    at_g_bluerow = Gm & B_row

    R = np.where(at_g_redrow, G + _conv_h(R, _K_B) - _conv_h(G, _K_B), R)
    R = np.where(at_g_bluerow, G + _conv_v(R, _K_B_T) - _conv_v(G, _K_B_T), R)
    B = np.where(at_g_bluerow, G + _conv_h(B, _K_B) - _conv_h(G, _K_B), B)
    B = np.where(at_g_redrow, G + _conv_v(B, _K_B_T) - _conv_v(G, _K_B_T), B)

    at_b = B_row & Bm
    at_r = R_row & Rm
    R = np.where(
        at_b,
        np.where(M, B + _conv_h(R, _K_B) - _conv_h(B, _K_B), B + _conv_v(R, _K_B_T) - _conv_v(B, _K_B_T)),
        R,
    )
    B = np.where(
        at_r,
        np.where(M, R + _conv_h(B, _K_B) - _conv_h(R, _K_B), R + _conv_v(B, _K_B_T) - _conv_v(R, _K_B_T)),
        B,
    )

    if refining_step:
        R, G, B = _refine(R, G, B, Rm, Gm, Bm, M, R_row, B_row)

    return np.dstack((R, G, B))


def _refine(R, G, B, Rm, Gm, Bm, M, R_row, B_row):
    R_G = R - G
    B_G = B - G
    R_G_h = _conv_h(R_G, _FIR)
    R_G_v = _conv_v(R_G, _FIR_T)
    B_G_h = _conv_h(B_G, _FIR)
    B_G_v = _conv_v(B_G, _FIR_T)
    R_G_m = np.where(Rm, np.where(M, R_G_h, R_G_v), 0.0)
    B_G_m = np.where(Bm, np.where(M, B_G_h, B_G_v), 0.0)
    G = np.where(Rm, R - R_G_m, G)
    G = np.where(Bm, B - B_G_m, G)

    R_col = np.any(Rm, axis=0)[None, :]
    B_col = np.any(Bm, axis=0)[None, :]
    R_G = R - G
    B_G = B - G
    R_G_h = _conv_h(R_G, _K_B)
    R_G_v = _conv_v(R_G, _K_B_T)
    B_G_h = _conv_h(B_G, _K_B)
    B_G_v = _conv_v(B_G, _K_B_T)

    sel = Gm & B_row
    R_G_m = np.where(sel, R_G_v, R_G_m)
    R = np.where(sel, G + R_G_m, R)
    sel = Gm & B_col
    R_G_m = np.where(sel, R_G_h, R_G_m)
    R = np.where(sel, G + R_G_m, R)
    sel = Gm & R_row
    B_G_m = np.where(sel, B_G_v, B_G_m)
    B = np.where(sel, G + B_G_m, B)
    sel = Gm & R_col
    B_G_m = np.where(sel, B_G_h, B_G_m)
    B = np.where(sel, G + B_G_m, B)

    R_B = R - B
    R_B_h = _conv_h(R_B, _FIR)
    R_B_v = _conv_v(R_B, _FIR_T)
    R_B_m = np.where(Bm, np.where(M, R_B_h, R_B_v), 0.0)
    R = np.where(Bm, B + R_B_m, R)
    R_B_m = np.where(Rm, np.where(M, R_B_h, R_B_v), 0.0)
    B = np.where(Rm, R - R_B_m, B)
    return R, G, B
