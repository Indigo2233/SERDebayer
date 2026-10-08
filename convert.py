"""SER、灰度 Bayer AVI、FITS → RGB SER。"""

from __future__ import annotations

import os

# 必须在 import numpy 之前：否则每个进程再乘 OpenBLAS 线程，内存会直接打满。
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

import mmap
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from engines import ALG_BY_KEY, apply_wb, demosaic_frame, estimate_wb_gains
from input_io import (
    Header,
    input_format_name,
    open_input,
    read_input_header,
)
from ser_io import (
    HEADER_SIZE,
    SerHeader,
    detect_pixel_dtype,
    read_frame,
    read_header,
    read_timestamps,
    slice_timestamps,
    write_rgb_header,
)

_WORKER: dict = {}


def _available_ram_gb() -> float:
    try:
        import ctypes

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        info = _MEMSTAT()
        info.dwLength = ctypes.sizeof(_MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
            return info.ullAvailPhys / 1024**3
    except Exception:
        pass
    return 8.0


def recommended_workers(algo_key: str, requested: int | None = None) -> int:
    cpu = max(1, os.cpu_count() or 4)
    ram = _available_ram_gb()
    algo = ALG_BY_KEY.get(algo_key)
    if algo is not None and algo.key.startswith("ddfapd"):
        cap = max(1, min(cpu, int(max(ram - 2.0, 1.0) / 0.5)))
    elif algo is not None and algo.impl == "colour":
        cap = max(1, min(cpu, int(max(ram - 2.0, 1.0) / 0.8)))
    else:
        cap = max(1, min(cpu, int(max(ram - 2.0, 1.0) / 0.35)))
    if requested is None:
        return cap
    return max(1, min(int(requested), cap, cpu))


def init_worker(
    src_path: str,
    dst_path: str,
    width: int,
    height: int,
    cfa_frame: int,
    rgb_frame: int,
    dtype_str: str,
    pattern: str,
    algo_key: str,
    output_depth: int,
    wb_r: float,
    wb_g: float,
    wb_b: float,
) -> None:
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    inf = open(src_path, "rb")
    outf = open(dst_path, "r+b")
    _WORKER.update(
        inf=inf,
        outf=outf,
        in_mm=mmap.mmap(inf.fileno(), 0, access=mmap.ACCESS_READ),
        out_mm=mmap.mmap(outf.fileno(), 0, access=mmap.ACCESS_WRITE),
        width=width,
        height=height,
        cfa_frame=cfa_frame,
        rgb_frame=rgb_frame,
        dtype=np.dtype(dtype_str),
        pattern=pattern,
        algo_key=algo_key,
        output_depth=output_depth,
        wb=(wb_r, wb_g, wb_b),
    )


def process_index(pair: tuple[int, int]) -> None:
    src_idx, out_idx = pair
    w = _WORKER
    raw = w["in_mm"][HEADER_SIZE + src_idx * w["cfa_frame"] : HEADER_SIZE + (src_idx + 1) * w["cfa_frame"]]
    cfa = np.frombuffer(raw, dtype=w["dtype"]).reshape(w["height"], w["width"]).copy()
    rgb = demosaic_frame(cfa, w["pattern"], w["algo_key"], w["output_depth"])
    rgb = apply_wb(rgb, w["wb"])
    blob = np.ascontiguousarray(rgb).tobytes()
    start = HEADER_SIZE + out_idx * w["rgb_frame"]
    w["out_mm"][start : start + len(blob)] = blob


def estimate_output_bytes(header: Header, count: int, depth: int) -> int:
    bpp = 3 if depth <= 8 else 6
    return HEADER_SIZE + count * header.width * header.height * bpp + count * 8


def frame_to_rgb(
    frame: np.ndarray,
    is_rgb: bool,
    pattern: str,
    algo_key: str,
    output_depth: int,
) -> np.ndarray:
    """把一个输入帧规范化为指定输出位深的 RGB。"""
    if not is_rgb:
        return demosaic_frame(frame, pattern, algo_key, output_depth)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"RGB 输入帧形状无效: {frame.shape}")
    if output_depth <= 8:
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)
        if frame.dtype == np.uint16:
            return (frame >> 8).astype(np.uint8)
    else:
        if frame.dtype == np.uint16:
            return np.ascontiguousarray(frame)
        if frame.dtype == np.uint8:
            return frame.astype(np.uint16) << 8
    raise TypeError(f"不支持的像素类型: {frame.dtype}")


def _convert_generic(
    src_path: str,
    dst_path: str,
    header: Header,
    pattern: str,
    algo_key: str,
    output_depth: int,
    start: int,
    count: int,
    white_balance: bool,
    progress: Callable[[int, int, str], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    tmp_path = dst_path + ".partial"
    wb = (1.0, 1.0, 1.0)
    if white_balance:
        with open_input(src_path) as source:
            sample = source.read_frame(start)
        sample_rgb = frame_to_rgb(sample, header.is_rgb, pattern, algo_key, output_depth)
        wb = estimate_wb_gains(sample_rgb)
        if progress:
            progress(0, count, f"白平衡增益 R={wb[0]:.3f} G={wb[1]:.3f} B={wb[2]:.3f}（全片同一系数）")
    if progress:
        progress(0, count, f"{input_format_name(header)} 按帧读取，输出 RGB SER")
    try:
        with open_input(src_path) as source, open(tmp_path, "wb") as out:
            write_rgb_header(out, header, count, output_depth)
            for out_index, src_index in enumerate(range(start, start + count), start=1):
                if should_cancel and should_cancel():
                    raise InterruptedError("已取消")
                frame = source.read_frame(src_index)
                rgb = frame_to_rgb(frame, header.is_rgb, pattern, algo_key, output_depth)
                rgb = apply_wb(rgb, wb)
                if rgb.dtype == np.uint16:
                    rgb = rgb.astype("<u2", copy=False)
                out.write(np.ascontiguousarray(rgb).tobytes())
                if progress:
                    progress(out_index, count, f"已转 {out_index}/{count}")
        os.replace(tmp_path, dst_path)
        if progress:
            progress(count, count, f"完成: {dst_path}")
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def convert_input(
    src_path: str,
    dst_path: str,
    pattern: str,
    algo_key: str,
    output_depth: int,
    start: int,
    count: int,
    workers: int,
    white_balance: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    header = read_input_header(src_path)
    fmt = input_format_name(header)
    if output_depth not in (8, 16):
        raise ValueError(f"输出位深必须是 8 或 16，当前为 {output_depth}")
    if algo_key not in ALG_BY_KEY:
        raise ValueError(f"未知算法: {algo_key}")
    if not header.is_rgb and output_depth > ALG_BY_KEY[algo_key].max_output_depth:
        raise ValueError(f"{ALG_BY_KEY[algo_key].label} 最高支持 {ALG_BY_KEY[algo_key].max_output_depth}bit 输出")
    if header.is_rgb and fmt == "SER":
        raise ValueError(f"输入已经是 RGB SER（当前 {header.color_name}）")
    if not header.is_rgb and not header.can_debayer:
        raise ValueError(f"输入不是单通道 Bayer 数据，无法 Debayer（当前 {header.color_name}）")
    if start < 0 or count < 1 or start + count > header.frames:
        raise ValueError("帧范围无效")

    if fmt != "SER":
        return _convert_generic(
            src_path,
            dst_path,
            header,
            pattern,
            algo_key,
            output_depth,
            start,
            count,
            white_balance,
            progress,
            should_cancel,
        )

    with open(src_path, "rb") as f:
        f.seek(HEADER_SIZE)
        probe = f.read(header.frame_size)
    dtype = detect_pixel_dtype(header, probe)
    timestamps = slice_timestamps(read_timestamps(src_path, header), start, count)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    tmp_path = dst_path + ".partial"
    indices = list(range(start, start + count))
    n_workers = recommended_workers(algo_key, workers)
    rgb_bpp = 3 if output_depth <= 8 else 6
    rgb_frame = header.width * header.height * rgb_bpp
    trailer = timestamps or b""
    total_size = HEADER_SIZE + count * rgb_frame + len(trailer)
    wb = (1.0, 1.0, 1.0)
    if white_balance:
        with open(src_path, "rb") as f:
            sample = read_frame(f, header, start, dtype)
        sample_rgb = demosaic_frame(sample, pattern, algo_key, output_depth)
        wb = estimate_wb_gains(sample_rgb)
        del sample, sample_rgb
        if progress:
            progress(0, count, f"白平衡增益 R={wb[0]:.3f} G={wb[1]:.3f} B={wb[2]:.3f}（全片同一系数）")
    if progress:
        ram = _available_ram_gb()
        if n_workers < workers:
            progress(
                0,
                count,
                f"可用内存约 {ram:.1f} GB，进程数 {workers} → {n_workers}",
            )
        else:
            progress(0, count, f"使用 {n_workers} 个进程，可用内存约 {ram:.1f} GB")

    ctx = get_context("spawn")
    try:
        with open(tmp_path, "wb") as out:
            write_rgb_header(out, header, count, output_depth)
            out.truncate(total_size)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(
                src_path,
                tmp_path,
                header.width,
                header.height,
                header.frame_size,
                rgb_frame,
                dtype.str,
                pattern,
                algo_key,
                output_depth,
                wb[0],
                wb[1],
                wb[2],
            ),
        ) as pool:
            jobs = [pool.submit(process_index, (src_i, out_i)) for out_i, src_i in enumerate(indices)]
            done = 0
            for fut in as_completed(jobs):
                if should_cancel and should_cancel():
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise InterruptedError("已取消")
                fut.result()
                done += 1
                if progress:
                    progress(done, count, f"已转 {done}/{count}")
        if trailer:
            with open(tmp_path, "r+b") as out:
                out.seek(HEADER_SIZE + count * rgb_frame)
                out.write(trailer)
        os.replace(tmp_path, dst_path)
        if progress:
            progress(count, count, f"完成: {dst_path}")
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# 保留旧 API，方便已有调用方继续使用。
convert_ser = convert_input


def default_output_path(src_path: str, depth: int) -> str:
    p = Path(src_path)
    tag = "RGB8" if depth <= 8 else "RGB16"
    return str(p.with_name(f"{p.stem}_{tag}.ser"))


def repair_rgb_ser(
    src_path: str,
    dst_path: str,
    white_balance: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """不重新转色：只修正字节序标志，并可选加上亮部白平衡。"""
    header = read_header(src_path)
    if not header.is_rgb:
        raise ValueError(f"不是 RGB SER（当前 {header.color_name}）")
    with open(src_path, "rb") as f:
        f.seek(HEADER_SIZE)
        probe = f.read(header.frame_size)
    dtype = detect_pixel_dtype(header, probe)
    timestamps = read_timestamps(src_path, header)
    with open(src_path, "rb") as f:
        first = read_frame(f, header, 0, dtype)
    wb = estimate_wb_gains(first) if white_balance else (1.0, 1.0, 1.0)
    if progress:
        progress(0, header.frames, f"修复字节序标志=0，白平衡 R={wb[0]:.3f} G={wb[1]:.3f} B={wb[2]:.3f}")
    tmp_path = dst_path + ".partial"
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    try:
        with open(src_path, "rb") as inf, open(tmp_path, "wb") as out:
            write_rgb_header(out, header, header.frames, header.depth)
            for i in range(header.frames):
                rgb = apply_wb(read_frame(inf, header, i, dtype), wb)
                out.write(np.ascontiguousarray(rgb).tobytes())
                if progress:
                    progress(i + 1, header.frames, f"已修复 {i + 1}/{header.frames}")
            if timestamps:
                out.write(timestamps)
        os.replace(tmp_path, dst_path)
        if progress:
            progress(header.frames, header.frames, f"完成: {dst_path}")
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
