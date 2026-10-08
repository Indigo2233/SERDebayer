"""适合脚本和 agent 调用的命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import freeze_support

from convert import convert_input, default_output_path, estimate_output_bytes, recommended_workers
from engines import ALG_BY_KEY
from input_io import input_format_name, read_input_header


def _header_dict(header) -> dict:
    return {
        "path": os.path.abspath(header.path),
        "format": input_format_name(header),
        "width": header.width,
        "height": header.height,
        "frames": header.frames,
        "depth": header.depth,
        "planes": header.planes,
        "color": header.color_name,
        "bayer_pattern": header.bayer_pattern,
        "file_size": header.file_size,
        "can_debayer": header.can_debayer,
        "is_rgb": header.is_rgb,
    }


def _emit(payload: dict, json_mode: bool, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    if json_mode:
        print(json.dumps(payload, ensure_ascii=True), file=stream, flush=True)
    else:
        event = payload.get("event")
        if event == "progress":
            print(f"[{payload['current']}/{payload['total']}] {payload['message']}", file=stream, flush=True)
        elif event == "complete":
            print(payload["output"], file=stream, flush=True)
        elif event == "error":
            print(payload["message"], file=stream, flush=True)
        else:
            for key, value in payload.items():
                print(f"{key}: {value}", file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="SERDebayerCLI",
        description="将 Bayer SER、8-bit 灰度 AVI、8/16-bit FITS 转换为 RGB SER。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="读取输入文件元数据")
    info.add_argument("input")
    info.add_argument("--json", action="store_true", help="输出 JSON")

    convert = sub.add_parser("convert", help="转换为 RGB SER")
    convert.add_argument("input")
    convert.add_argument("-o", "--output", help="输出路径，默认写入输入文件旁")
    convert.add_argument("--pattern", choices=("auto", "RGGB", "GRBG", "GBRG", "BGGR"), default="auto")
    convert.add_argument("--algorithm", choices=tuple(ALG_BY_KEY), default="ddfapd")
    convert.add_argument("--output-depth", type=int, choices=(8, 16), default=16)
    convert.add_argument("--start", type=int, default=0, help="起始帧，0-based")
    frame_range = convert.add_mutually_exclusive_group()
    frame_range.add_argument("--count", type=int, help="转换帧数")
    frame_range.add_argument("--end", type=int, help="结束帧，0-based 且包含该帧")
    convert.add_argument("--workers", type=int, help="SER 多进程数；AVI/FITS 顺序读取")
    convert.add_argument(
        "--white-balance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或关闭亮部灰世界白平衡；Bayer 默认启用，RGB FITS 默认关闭",
    )
    convert.add_argument("--overwrite", action="store_true", help="覆盖现有输出文件")
    convert.add_argument("--json", action="store_true", help="按 JSON Lines 输出进度和结果")
    convert.add_argument("--quiet", action="store_true", help="只输出最终结果或错误")
    return parser


def _run_info(args) -> int:
    header = read_input_header(args.input)
    payload = _header_dict(header)
    payload["event"] = "info"
    _emit(payload, args.json)
    return 0


def _run_convert(args) -> int:
    header = read_input_header(args.input)
    output = os.path.abspath(args.output or default_output_path(args.input, args.output_depth))
    if os.path.splitext(output)[1].lower() != ".ser":
        raise ValueError(f"输出文件必须使用 .ser 扩展名: {output}")
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"输出文件已经存在；使用 --overwrite 覆盖: {output}")
    start = args.start
    if args.end is not None:
        count = args.end - start + 1
    elif args.count is not None:
        count = args.count
    else:
        count = header.frames - start
    if start < 0 or count < 1 or start + count > header.frames:
        raise ValueError(f"帧范围无效: start={start}, count={count}, frames={header.frames}")
    pattern = args.pattern
    if pattern == "auto":
        pattern = header.bayer_pattern or "RGGB"
    workers = args.workers or recommended_workers(args.algorithm)
    white_balance = (not header.is_rgb) if args.white_balance is None else args.white_balance
    estimated = estimate_output_bytes(header, count, args.output_depth)
    if not args.quiet:
        _emit(
            {
                "event": "start",
                "input": os.path.abspath(args.input),
                "output": output,
                "format": input_format_name(header),
                "frames": count,
                "estimated_bytes": estimated,
                "pattern": None if header.is_rgb else pattern,
                "algorithm": None if header.is_rgb else args.algorithm,
                "output_depth": args.output_depth,
                "white_balance": white_balance,
            },
            args.json,
        )

    def progress(current: int, total: int, message: str) -> None:
        if not args.quiet:
            _emit(
                {"event": "progress", "current": current, "total": total, "message": message},
                args.json,
                stream=sys.stderr,
            )

    convert_input(
        src_path=args.input,
        dst_path=output,
        pattern=pattern,
        algo_key=args.algorithm,
        output_depth=args.output_depth,
        start=start,
        count=count,
        workers=workers,
        white_balance=white_balance,
        progress=progress,
    )
    _emit({"event": "complete", "output": output}, args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_info(args) if args.command == "info" else _run_convert(args)
    except Exception as exc:
        json_mode = bool(getattr(args, "json", False))
        _emit({"event": "error", "type": type(exc).__name__, "message": str(exc)}, json_mode, stream=sys.stderr)
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
