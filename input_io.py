"""统一读取 SER、8-bit 灰度 AVI 和 8/16-bit FITS。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ser_io import (
    BAYER_PATTERNS,
    SerHeader,
    detect_pixel_dtype,
    read_frame as read_ser_frame,
    read_header as read_ser_header,
    read_timestamps as read_ser_timestamps,
)


SUPPORTED_INPUT_SUFFIXES = {".ser", ".avi", ".fit", ".fits", ".fts"}


@dataclass
class InputHeader:
    path: str
    format_name: str
    width: int
    height: int
    depth: int
    frames: int
    planes: int
    color_name: str
    file_size: int
    observer: str = ""
    instrument: str = ""
    telescope: str = ""
    datetime: int = 0
    datetime_utc: int = 0
    bayer_pattern: str | None = None
    fits_hdu: int = 0
    fits_layout: str = "mono"
    fits_bscale: float = 1.0
    fits_bzero: float = 0.0

    @property
    def is_bayer(self) -> bool:
        return self.bayer_pattern is not None

    @property
    def is_rgb(self) -> bool:
        return self.planes == 3

    @property
    def can_debayer(self) -> bool:
        return self.planes == 1

    @property
    def bytes_per_sample(self) -> int:
        return 1 if self.depth <= 8 else 2

    @property
    def bytes_per_pixel(self) -> int:
        return self.bytes_per_sample * self.planes

    @property
    def frame_size(self) -> int:
        return self.width * self.height * self.bytes_per_pixel


Header = SerHeader | InputHeader


class FrameSource(Protocol):
    header: Header

    def read_frame(self, index: int) -> np.ndarray: ...

    def close(self) -> None: ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...


def detect_input_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".ser":
        return "SER"
    if suffix == ".avi":
        return "AVI"
    if suffix in {".fit", ".fits", ".fts"}:
        return "FITS"
    supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
    raise ValueError(f"不支持的输入格式 {suffix or '(无扩展名)'}；支持: {supported}")


def _normalise_pattern(value: object) -> str | None:
    if value is None:
        return None
    text = "".join(ch for ch in str(value).upper() if ch.isalpha())
    return text if text in set(BAYER_PATTERNS.values()) else None


def _avi_gray(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        raise ValueError(f"AVI 必须解码为 8-bit，当前 dtype={frame.dtype}")
    if frame.ndim == 2:
        return np.ascontiguousarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.ascontiguousarray(frame[:, :, 0])
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        rgb = frame[:, :, :3]
        spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        if int(spread.max(initial=0)) <= 1:
            return np.ascontiguousarray(rgb[:, :, 0])
    raise ValueError("AVI 不是 8-bit 灰度视频，无法作为 Bayer CFA 输入")


def _read_avi_header(path: str) -> InputHeader:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开 AVI: {path}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"AVI 没有可读取的视频帧: {path}")
        gray = _avi_gray(frame)
        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if frames <= 0:
            frames = 1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _avi_gray(frame)
                frames += 1
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        return InputHeader(
            path=path,
            format_name="AVI",
            width=int(gray.shape[1]),
            height=int(gray.shape[0]),
            depth=8,
            frames=frames,
            planes=1,
            color_name="MONO/BAYER",
            file_size=os.path.getsize(path),
            instrument=f"AVI {fourcc}".strip(),
        )
    finally:
        cap.release()


def _first_image_hdu(hdul) -> tuple[int, object, np.ndarray]:
    for index, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and np.ndim(data) in (2, 3):
            return index, hdu, data
    raise ValueError("FITS 中没有二维或三维图像 HDU")


def _read_fits_header(path: str) -> InputHeader:
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError("读取 FITS 需要安装 astropy: python -m pip install astropy") from exc

    with fits.open(path, memmap=True, do_not_scale_image_data=True, uint=False) as hdul:
        hdu_index, hdu, data = _first_image_hdu(hdul)
        bitpix = int(hdu.header.get("BITPIX", 0))
        if bitpix not in (8, 16):
            raise ValueError(f"当前只支持 8/16-bit 整数 FITS，BITPIX={bitpix}")
        shape = tuple(int(v) for v in data.shape)
        if len(shape) == 2:
            frames, planes, height, width, layout = 1, 1, shape[0], shape[1], "mono"
        elif shape[0] in (3, 4):
            frames, planes, height, width, layout = 1, 3, shape[1], shape[2], "rgb-first"
        elif shape[-1] in (3, 4):
            frames, planes, height, width, layout = 1, 3, shape[0], shape[1], "rgb-last"
        else:
            frames, planes, height, width, layout = shape[0], 1, shape[1], shape[2], "mono-cube"
        pattern = None
        for key in ("BAYERPAT", "BAYERPATN", "BAYER"):
            pattern = _normalise_pattern(hdu.header.get(key))
            if pattern:
                break
        return InputHeader(
            path=path,
            format_name="FITS",
            width=width,
            height=height,
            depth=bitpix,
            frames=frames,
            planes=planes,
            color_name="RGB" if planes == 3 else (pattern or "MONO/BAYER"),
            file_size=os.path.getsize(path),
            observer=str(hdu.header.get("OBSERVER", "")).strip(),
            instrument=str(hdu.header.get("INSTRUME", "")).strip(),
            telescope=str(hdu.header.get("TELESCOP", "")).strip(),
            bayer_pattern=pattern,
            fits_hdu=hdu_index,
            fits_layout=layout,
            fits_bscale=float(hdu.header.get("BSCALE", 1.0)),
            fits_bzero=float(hdu.header.get("BZERO", 0.0)),
        )


def read_input_header(path: str) -> Header:
    fmt = detect_input_format(path)
    if fmt == "SER":
        return read_ser_header(path)
    if fmt == "AVI":
        return _read_avi_header(path)
    return _read_fits_header(path)


def input_format_name(header: Header) -> str:
    return "SER" if isinstance(header, SerHeader) else header.format_name


def read_input_timestamps(path: str, header: Header) -> bytes | None:
    if isinstance(header, SerHeader):
        return read_ser_timestamps(path, header)
    return None


class _BaseSource:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SerSource(_BaseSource):
    def __init__(self, path: str):
        self.header = read_ser_header(path)
        self._file = open(path, "rb")
        self._file.seek(178)
        probe = self._file.read(self.header.frame_size)
        self._dtype = detect_pixel_dtype(self.header, probe)

    def read_frame(self, index: int) -> np.ndarray:
        return read_ser_frame(self._file, self.header, index, self._dtype)

    def close(self) -> None:
        self._file.close()


class AviSource(_BaseSource):
    def __init__(self, path: str):
        import cv2

        self._cv2 = cv2
        self.header = _read_avi_header(path)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"无法打开 AVI: {path}")
        self._next_index = 0

    def read_frame(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.header.frames:
            raise IndexError(index)
        if index != self._next_index:
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise IOError(f"AVI 第 {index} 帧读取失败")
        self._next_index = index + 1
        return _avi_gray(frame)

    def close(self) -> None:
        self._cap.release()


class FitsSource(_BaseSource):
    def __init__(self, path: str):
        from astropy.io import fits

        self.header = _read_fits_header(path)
        self._hdul = fits.open(path, memmap=True, do_not_scale_image_data=True, uint=False)
        self._data = self._hdul[self.header.fits_hdu].data

    def _scale(self, frame: np.ndarray) -> np.ndarray:
        h = self.header
        if h.depth == 8 and h.fits_bscale == 1.0 and h.fits_bzero == 0.0:
            return np.array(frame, dtype=np.uint8, order="C", copy=True)
        values = frame.astype(np.float64)
        values = values * h.fits_bscale + h.fits_bzero
        if h.depth <= 8:
            return np.clip(values, 0, 255).astype(np.uint8)
        return np.clip(values, 0, 65535).astype(np.uint16)

    def read_frame(self, index: int) -> np.ndarray:
        h = self.header
        if index < 0 or index >= h.frames:
            raise IndexError(index)
        if h.fits_layout == "mono":
            frame = self._data
        elif h.fits_layout == "mono-cube":
            frame = self._data[index]
        elif h.fits_layout == "rgb-first":
            frame = np.moveaxis(self._data[:3], 0, -1)
        else:
            frame = self._data[:, :, :3]
        return self._scale(np.asarray(frame))

    def close(self) -> None:
        self._data = None
        self._hdul.close()


def open_input(path: str) -> FrameSource:
    fmt = detect_input_format(path)
    if fmt == "SER":
        return SerSource(path)
    if fmt == "AVI":
        return AviSource(path)
    return FitsSource(path)
