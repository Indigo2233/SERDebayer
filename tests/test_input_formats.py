from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import tempfile
import unittest

import cv2
import numpy as np
from astropy.io import fits

import cli
from convert import convert_input
from input_io import open_input, read_input_header
from ser_io import detect_pixel_dtype, read_frame, read_header


def _write_mono_ser(path: str, frames: list[np.ndarray], depth: int = 8) -> None:
    height, width = frames[0].shape
    header = bytearray(178)
    header[:14] = b"LUCAM-RECORDER"
    struct.pack_into("<iiiiiii", header, 14, 0, 8, 0, width, height, depth, len(frames))
    with open(path, "wb") as file:
        file.write(header)
        for frame in frames:
            file.write(np.ascontiguousarray(frame).tobytes())


class FitsInputTests(unittest.TestCase):
    def test_reads_8_and_16_bit_mono_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            path8 = os.path.join(temp, "mono8.fits")
            path16 = os.path.join(temp, "mono16.fits")
            data8 = np.arange(48, dtype=np.uint8).reshape(6, 8)
            data16 = (np.arange(48, dtype=np.uint16).reshape(6, 8) * 1000)
            fits.writeto(path8, data8)
            fits.writeto(path16, data16)

            header8 = read_input_header(path8)
            header16 = read_input_header(path16)
            self.assertEqual((header8.depth, header8.frames, header8.planes), (8, 1, 1))
            self.assertEqual((header16.depth, header16.frames, header16.planes), (16, 1, 1))
            with open_input(path8) as source:
                np.testing.assert_array_equal(source.read_frame(0), data8)
            with open_input(path16) as source:
                np.testing.assert_array_equal(source.read_frame(0), data16)

    def test_converts_rgb_fits_directly_to_rgb_ser(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "rgb.fits")
            dst = os.path.join(temp, "rgb.ser")
            planar = np.zeros((3, 6, 8), dtype=np.uint8)
            planar[0] = 12
            planar[1] = 34
            planar[2] = 56
            fits.writeto(src, planar)

            convert_input(src, dst, "RGGB", "bilinear", 8, 0, 1, 1, white_balance=False)

            header = read_header(dst)
            self.assertTrue(header.is_rgb)
            with open(dst, "rb") as file:
                file.seek(178)
                dtype = detect_pixel_dtype(header, file.read(header.frame_size))
                frame = read_frame(file, header, 0, dtype)
            np.testing.assert_array_equal(frame[:, :, 0], planar[0])
            np.testing.assert_array_equal(frame[:, :, 1], planar[1])
            np.testing.assert_array_equal(frame[:, :, 2], planar[2])

    def test_converts_8_bit_bayer_fits_to_rgb_ser(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "bayer.fits")
            dst = os.path.join(temp, "bayer_rgb.ser")
            bayer = np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
            fits.writeto(src, bayer, header=fits.Header({"BAYERPAT": "RGGB"}))

            convert_input(src, dst, "RGGB", "bilinear", 8, 0, 1, 1, white_balance=False)

            header = read_header(dst)
            self.assertEqual((header.depth, header.frames, header.color_name), (8, 1, "RGB"))
            self.assertEqual(header.file_size, 178 + 8 * 8 * 3)

    def test_converts_16_bit_bayer_fits_to_rgb_ser(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "bayer16.fits")
            dst = os.path.join(temp, "bayer16_rgb.ser")
            bayer = np.arange(64, dtype=np.uint16).reshape(8, 8) * 800
            fits.writeto(src, bayer, header=fits.Header({"BAYERPAT": "BGGR"}))

            convert_input(src, dst, "BGGR", "bilinear", 16, 0, 1, 1, white_balance=False)

            header = read_header(dst)
            self.assertEqual((header.depth, header.frames, header.color_name), (16, 1, "RGB"))
            self.assertEqual(header.file_size, 178 + 8 * 8 * 6)


class AviInputTests(unittest.TestCase):
    def test_reads_and_converts_8_bit_gray_avi(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "bayer.avi")
            dst = os.path.join(temp, "bayer_rgb.ser")
            writer = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (8, 8), isColor=False)
            if not writer.isOpened():
                self.skipTest("当前 OpenCV 构建没有 FFV1 AVI 编码器")
            try:
                for index in range(3):
                    writer.write(np.full((8, 8), 30 + index * 20, dtype=np.uint8))
            finally:
                writer.release()

            header = read_input_header(src)
            self.assertEqual((header.depth, header.frames, header.planes), (8, 3, 1))
            with open_input(src) as source:
                self.assertEqual(source.read_frame(1).shape, (8, 8))

            convert_input(src, dst, "RGGB", "bilinear", 8, 0, 3, 1, white_balance=False)
            output = read_header(dst)
            self.assertEqual((output.depth, output.frames, output.color_name), (8, 3, "RGB"))


class SerRegressionTests(unittest.TestCase):
    def test_existing_ser_conversion_path(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "bayer.ser")
            dst = os.path.join(temp, "bayer_rgb.ser")
            frames = [np.arange(64, dtype=np.uint8).reshape(8, 8)]
            _write_mono_ser(src, frames)

            convert_input(src, dst, "RGGB", "bilinear", 8, 0, 1, 1, white_balance=False)

            output = read_header(dst)
            self.assertEqual((output.depth, output.frames, output.color_name), (8, 1, "RGB"))


class CliTests(unittest.TestCase):
    def test_info_json(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "mono8.fits")
            fits.writeto(src, np.zeros((6, 8), dtype=np.uint8))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(["info", src, "--json"])
            payload = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["format"], "FITS")
            self.assertEqual(payload["depth"], 8)

    def test_rgb_fits_cli_preserves_channels_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            src = os.path.join(temp, "rgb.fits")
            dst = os.path.join(temp, "rgb.ser")
            planar = np.stack(
                [
                    np.full((6, 8), 20, dtype=np.uint8),
                    np.full((6, 8), 80, dtype=np.uint8),
                    np.full((6, 8), 140, dtype=np.uint8),
                ]
            )
            fits.writeto(src, planar)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(["convert", src, "-o", dst, "--output-depth", "8", "--quiet"])
            self.assertEqual(code, 0)
            header = read_header(dst)
            with open(dst, "rb") as file:
                file.seek(178)
                dtype = detect_pixel_dtype(header, file.read(header.frame_size))
                frame = read_frame(file, header, 0, dtype)
            np.testing.assert_array_equal(frame[:, :, 0], planar[0])
            np.testing.assert_array_equal(frame[:, :, 1], planar[1])
            np.testing.assert_array_equal(frame[:, :, 2], planar[2])


if __name__ == "__main__":
    unittest.main()
