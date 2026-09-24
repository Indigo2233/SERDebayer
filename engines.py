"""把现成开源 demosaic 接到 numpy 帧上。这里不实现算法，只做调用。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
# colour-demosaicing: Menon 2007 DDFAPD / Malvar 2004
# Siril CLI: RCD（SER 转色固定用 RCD）

_OPENCV_CODE = {
    "RGGB": {
        "bilinear": cv2.COLOR_BayerRGGB2RGB,
        "ea": cv2.COLOR_BayerRGGB2RGB_EA,
        "vng": cv2.COLOR_BayerRGGB2RGB_VNG,
    },
    "GRBG": {
        "bilinear": cv2.COLOR_BayerGRBG2RGB,
        "ea": cv2.COLOR_BayerGRBG2RGB_EA,
        "vng": cv2.COLOR_BayerGRBG2RGB_VNG,
    },
    "GBRG": {
        "bilinear": cv2.COLOR_BayerGBRG2RGB,
        "ea": cv2.COLOR_BayerGBRG2RGB_EA,
        "vng": cv2.COLOR_BayerGBRG2RGB_VNG,
    },
    "BGGR": {
        "bilinear": cv2.COLOR_BayerBGGR2RGB,
        "ea": cv2.COLOR_BayerBGGR2RGB_EA,
        "vng": cv2.COLOR_BayerBGGR2RGB_VNG,
    },
}


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    source: str
    max_output_depth: int  # 8 or 16
    impl: str  # opencv / colour
    recommend: bool = False
    note: str = ""


def _has_malvar() -> bool:
    try:
        import colour_demosaicing  # noqa: F401

        return True
    except ImportError:
        return False


ALGORITHMS = [
    Algorithm(
        "ddfapd",
        "DDFAPD / Menon 2007",
        "OpenCV",
        16,
        "colour",
        True,
        "抗网格。OpenCV float32 加速，可开更多进程",
    ),
    Algorithm(
        "ddfapd_refine",
        "DDFAPD + 细化",
        "OpenCV",
        16,
        "colour",
        False,
        "更慢一点，边缘更干净",
    ),
]
if _has_malvar():
    ALGORITHMS.append(
        Algorithm(
            "malvar",
            "Malvar 2004",
            "colour-science",
            16,
            "colour",
            False,
            "比双线性干净，比 DDFAPD 快",
        )
    )
ALGORITHMS.extend(
    [
        Algorithm(
            "ea",
            "EA 边缘感知",
            "OpenCV",
            16,
            "opencv",
            False,
            "16bit 快速，网格比双线性轻",
        ),
        Algorithm(
            "vng",
            "VNG",
            "OpenCV",
            8,
            "opencv",
            False,
            "和 dcraw/Fitswork 相机 RAW 同族；OpenCV 只支持 8bit",
        ),
        Algorithm(
            "bilinear",
            "双线性",
            "OpenCV",
            16,
            "opencv",
            False,
            "AS!4 默认同类，拉饱和容易出彩色网格",
        ),
    ]
)

ALG_BY_KEY = {a.key: a for a in ALGORITHMS}


def _as_uint16(cfa: np.ndarray) -> np.ndarray:
    if cfa.dtype == np.uint16:
        return cfa
    if cfa.dtype == np.uint8:
        return cfa.astype(np.uint16) << 8
    raise TypeError(cfa.dtype)


def _scale_to_unit(cfa: np.ndarray) -> tuple[np.ndarray, float]:
    src = cfa.astype(np.float32, copy=False)
    peak = 65535.0 if cfa.dtype == np.uint16 or src.max() > 255 else 255.0
    return src / peak, peak


def _from_unit(rgb: np.ndarray, peak: float, depth: int) -> np.ndarray:
    out = np.clip(np.nan_to_num(rgb, nan=0.0) * peak, 0, peak)
    if depth <= 8:
        return np.clip(out * (255.0 / peak), 0, 255).astype(np.uint8)
    return out.astype(np.uint16)


def demosaic_frame(
    cfa: np.ndarray,
    pattern: str,
    algo_key: str,
    output_depth: int,
) -> np.ndarray:
    """输入 HxW Bayer，输出 HxWx3 RGB。"""
    if pattern not in _OPENCV_CODE:
        raise ValueError(f"不支持的 Bayer: {pattern}")
    algo = ALG_BY_KEY[algo_key]
    depth = min(output_depth, algo.max_output_depth)

    if algo.impl == "opencv":
        code = _OPENCV_CODE[pattern][algo_key]
        if algo_key == "vng":
            if cfa.dtype != np.uint8:
                shift = 8 if cfa.dtype == np.uint16 else 0
                src8 = (cfa >> shift).astype(np.uint8) if shift else cfa.astype(np.uint8)
            else:
                src8 = cfa
            rgb = cv2.cvtColor(src8, code)
            return rgb if depth <= 8 else (rgb.astype(np.uint16) << 8)
        src = cfa if cfa.dtype in (np.uint8, np.uint16) else _as_uint16(cfa)
        rgb = cv2.cvtColor(src, code)
        if depth <= 8 and rgb.dtype == np.uint16:
            return (rgb >> 8).astype(np.uint8)
        if depth > 8 and rgb.dtype == np.uint8:
            return rgb.astype(np.uint16) << 8
        return rgb

    if algo_key in ("ddfapd", "ddfapd_refine"):
        from menon_fast import menon2007_fast

        src = _as_uint16(cfa) if cfa.dtype != np.uint8 or output_depth > 8 else cfa
        rgb_f = menon2007_fast(
            src,
            pattern=pattern,
            refining_step=(algo_key == "ddfapd_refine"),
        )
        if depth <= 8:
            if src.dtype == np.uint16:
                return np.clip(rgb_f * (255.0 / 65535.0), 0, 255).astype(np.uint8)
            return np.clip(rgb_f, 0, 255).astype(np.uint8)
        return np.clip(rgb_f, 0, 65535).astype(np.uint16)

    if algo_key == "malvar":
        from colour.utilities import set_default_float_dtype
        from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

        set_default_float_dtype(np.float32)
        unit, peak = _scale_to_unit(_as_uint16(cfa) if output_depth > 8 else cfa)
        rgb = demosaicing_CFA_Bayer_Malvar2004(unit, pattern=pattern)
        return _from_unit(np.asarray(rgb), peak, depth)

    raise ValueError(f"算法 {algo_key} 不能按帧调用，请走整段转换")


def estimate_wb_gains(rgb: np.ndarray) -> tuple[float, float, float]:
    """用画面亮部（月亮/行星盘）做灰世界，把 G 当基准。"""
    x = rgb.astype(np.float64, copy=False)
    lum = x.mean(axis=2)
    thr = float(np.percentile(lum, 75))
    mask = lum >= max(thr, 1.0)
    if int(mask.sum()) < 256:
        mask = lum > 0
    means = np.array([float(x[:, :, c][mask].mean()) for c in range(3)], dtype=np.float64)
    means = np.maximum(means, 1.0)
    gains = means[1] / means
    peak = float((x * gains).max())
    limit = 255.0 if rgb.dtype == np.uint8 else 65535.0
    if peak > limit:
        gains *= limit / peak
    return float(gains[0]), float(gains[1]), float(gains[2])


def apply_wb(rgb: np.ndarray, gains: tuple[float, float, float] | None) -> np.ndarray:
    if not gains:
        return rgb
    if abs(gains[0] - 1.0) + abs(gains[1] - 1.0) + abs(gains[2] - 1.0) < 1e-6:
        return rgb
    out = rgb.astype(np.float32, copy=True)
    out *= np.asarray(gains, dtype=np.float32)
    if rgb.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(out, 0, 65535).astype(np.uint16)
