"""SER v3 读写。只处理容器格式，不做 demosaic。"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

HEADER_SIZE = 178
FILE_ID = b"LUCAM-RECORDER"

COLOR_NAMES = {
    0: "MONO",
    8: "RGGB",
    9: "GRBG",
    10: "GBRG",
    11: "BGGR",
    16: "CYYM",
    17: "YCMY",
    18: "YMCY",
    19: "MYYC",
    100: "RGB",
    101: "BGR",
}
BAYER_PATTERNS = {8: "RGGB", 9: "GRBG", 10: "GBRG", 11: "BGGR"}
PATTERN_TO_COLOR_ID = {v: k for k, v in BAYER_PATTERNS.items()}


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<i", buf, off)[0]


def _fix_str(buf: bytes) -> str:
    return buf.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


@dataclass
class SerHeader:
    path: str
    color_id: int
    little_endian_flag: int
    width: int
    height: int
    depth: int
    frames: int
    observer: str
    instrument: str
    telescope: str
    datetime: int
    datetime_utc: int
    file_size: int

    @property
    def color_name(self) -> str:
        return COLOR_NAMES.get(self.color_id, f"UNKNOWN({self.color_id})")

    @property
    def is_bayer(self) -> bool:
        return self.color_id in BAYER_PATTERNS

    @property
    def is_rgb(self) -> bool:
        return self.color_id in (100, 101)

    @property
    def can_debayer(self) -> bool:
        """单通道就能按 Bayer 解。很多采集软件把 CFA 标成 MONO。"""
        return self.planes == 1

    @property
    def planes(self) -> int:
        return 3 if self.is_rgb else 1

    @property
    def bytes_per_sample(self) -> int:
        return 1 if self.depth <= 8 else 2

    @property
    def bytes_per_pixel(self) -> int:
        return self.bytes_per_sample * self.planes

    @property
    def frame_size(self) -> int:
        return self.width * self.height * self.bytes_per_pixel

    @property
    def bayer_pattern(self) -> str | None:
        return BAYER_PATTERNS.get(self.color_id)

    @property
    def has_trailer(self) -> bool:
        return self.file_size >= HEADER_SIZE + self.frames * self.frame_size + 8 * self.frames


def read_header(path: str) -> SerHeader:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = f.read(HEADER_SIZE)
    if len(buf) < HEADER_SIZE or buf[:14] != FILE_ID:
        raise ValueError(f"不是有效的 SER 文件: {path}")
    return SerHeader(
        path=path,
        color_id=_u32(buf, 18),
        little_endian_flag=_u32(buf, 22),
        width=_u32(buf, 26),
        height=_u32(buf, 30),
        depth=_u32(buf, 34),
        frames=_u32(buf, 38),
        observer=_fix_str(buf[42:82]),
        instrument=_fix_str(buf[82:122]),
        telescope=_fix_str(buf[122:162]),
        datetime=struct.unpack_from("<q", buf, 162)[0],
        datetime_utc=struct.unpack_from("<q", buf, 170)[0],
        file_size=size,
    )


def detect_pixel_dtype(header: SerHeader, probe: bytes) -> np.dtype:
    """FireCapture 常把 LittleEndian 写成 0，但像素实际是 LE。用一帧数据判别。"""
    if header.bytes_per_sample == 1:
        return np.dtype(np.uint8)
    le = np.frombuffer(probe, dtype="<u2")
    be = np.frombuffer(probe, dtype=">u2")
    # 行星素材天空占大部分，错误字节序通常把低字节抬到高位，中位数会虚高。
    if float(np.median(be)) > float(np.median(le)) * 1.35:
        return np.dtype("<u2")
    if header.little_endian_flag == 1:
        return np.dtype("<u2")
    return np.dtype(">u2")


def read_frame(f: BinaryIO, header: SerHeader, index: int, dtype: np.dtype) -> np.ndarray:
    if index < 0 or index >= header.frames:
        raise IndexError(index)
    f.seek(HEADER_SIZE + index * header.frame_size)
    raw = f.read(header.frame_size)
    if len(raw) != header.frame_size:
        raise IOError(f"第 {index} 帧不完整")
    if header.planes == 1:
        return np.frombuffer(raw, dtype=dtype).reshape(header.height, header.width).astype(
            np.uint16 if header.bytes_per_sample == 2 else np.uint8, copy=True
        )
    arr = np.frombuffer(raw, dtype=dtype).reshape(header.height, header.width, 3)
    if header.color_id == 101:
        arr = arr[..., ::-1]
    return arr.astype(np.uint16 if header.bytes_per_sample == 2 else np.uint8, copy=True)


def read_timestamps(path: str, header: SerHeader) -> bytes | None:
    if not header.has_trailer:
        return None
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE + header.frames * header.frame_size)
        data = f.read(8 * header.frames)
    return data if len(data) == 8 * header.frames else None


def _pad40(text: str) -> bytes:
    raw = (text or "").encode("ascii", errors="replace")[:40]
    return raw + b"\x00" * (40 - len(raw))


def write_rgb_header(
    f: BinaryIO,
    src: SerHeader,
    frames: int,
    depth: int,
) -> None:
    """写出 ColorID=100 的 RGB SER。

    像素始终按小端写入。LittleEndian 标志写成 0，和 FireCapture / AS!4
    的习惯一致（规范里 1=LE，但圈里大量软件把 0 当 LE 用）。
    """
    little_endian = 0
    buf = bytearray(HEADER_SIZE)
    buf[0:14] = FILE_ID
    struct.pack_into(
        "<iiiiiii",
        buf,
        14,
        0,
        100,
        little_endian,
        src.width,
        src.height,
        depth,
        frames,
    )
    buf[42:82] = _pad40(src.observer)
    buf[82:122] = _pad40(src.instrument)
    buf[122:162] = _pad40(src.telescope)
    struct.pack_into("<qq", buf, 162, src.datetime, src.datetime_utc)
    f.write(buf)


def slice_timestamps(blob: bytes | None, start: int, count: int) -> bytes | None:
    if not blob:
        return None
    return blob[start * 8 : (start + count) * 8]
