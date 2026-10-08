"""Bayer SER/AVI/FITS 真彩转色 GUI。"""

from __future__ import annotations

import os
import sys
import traceback

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from convert import convert_input, default_output_path, estimate_output_bytes, frame_to_rgb, recommended_workers
from engines import ALGORITHMS, ALG_BY_KEY, apply_wb, estimate_wb_gains
from input_io import SUPPORTED_INPUT_SUFFIXES, input_format_name, open_input, read_input_header

HELP = """这个软件使用现成算法完成 Bayer 转色，并负责 SER/AVI/FITS 读取与 RGB SER 写出：

• DDFAPD / Menon 2007
  OpenCV float32 卷积（menon_fast）
• Malvar 2004
  colour-demosaicing（开发机可选；发行包默认不带，体积小很多）
• VNG、EA、双线性
  来自 OpenCV

Fitswork4.exe 没有命令行，dcrawfw.dll 只是 32 位相机 RAW 加载器，
既不能吃 SER 帧，也写不出 24/48bit RGB SER。
菜单里的 Astro / Kimmel / VNG Color Correction 封在 exe 里，外部调不到。
OpenCV VNG 和 dcraw -q 1、Fitswork 相机 RAW 转色是同一族算法。

本机 Siril 1.4 的 CLI convert 只转换 FITS/TIFF/视频，不会去转现成的 Bayer SER，
所以没有做成一键调用。要 RCD 请直接在 Siril 里打开该 SER。

输入支持 Bayer/MONO SER、8-bit 灰度 Bayer AVI、8/16-bit FITS。
三通道 RGB FITS 会直接转换为 RGB SER。
输出是 SER v3 ColorID=100 的 RGB 视频，可直接丢给 AutoStakkert!4。
进 AS!4 后 Color 选 Auto Detect，不要再 Force Debayer。
头信息写成 MONO 的单通道 SER 也会 Debayer，阵列自己选（默认 RGGB）。

分发：运行 build.bat，把 dist\\SERDebayer 整个文件夹打成 zip 即可。
对方不需要安装 Python。
"""


def auto_stretch(rgb: np.ndarray, sat: float) -> np.ndarray:
    x = rgb.astype(np.float32)
    lum = 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]
    lo, hi = np.percentile(lum, (0.5, 99.8))
    y = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb8 = (y * 255).astype(np.uint8)
    return auto_stretch_sat(rgb8, sat)


def auto_stretch_sat(rgb8: np.ndarray, sat: float) -> np.ndarray:
    if abs(sat - 1.0) < 1e-3:
        return rgb8
    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def to_pixmap(rgb8: np.ndarray) -> QPixmap:
    h, w, _ = rgb8.shape
    qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class Worker(QThread):
    progress = Signal(int, int, str)
    failed = Signal(str)
    done = Signal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            convert_input(
                **self.kwargs,
                progress=lambda i, n, msg: self.progress.emit(i, n, msg),
                should_cancel=lambda: self._cancel,
            )
            self.done.emit(self.kwargs["dst_path"])
        except InterruptedError as exc:
            self.failed.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc())


class PreviewWorker(QThread):
    ready = Signal(object, str, float)
    failed = Signal(str)

    def __init__(self, path, index, pattern, algo, depth, sat, white_balance=True):
        super().__init__()
        self.path = path
        self.index = index
        self.pattern = pattern
        self.algo = algo
        self.depth = depth
        self.sat = sat
        self.white_balance = white_balance

    def run(self) -> None:
        try:
            header = read_input_header(self.path)
            with open_input(self.path) as source:
                frame = source.read_frame(self.index)
            rgb = frame_to_rgb(frame, header.is_rgb, self.pattern, self.algo, self.depth)
            if self.white_balance:
                rgb = apply_wb(rgb, estimate_wb_gains(rgb))
            note = f"第 {self.index} 帧  {rgb.shape[1]}x{rgb.shape[0]}  {rgb.dtype}"
            vis = auto_stretch(rgb, 1.0)
            self.ready.emit(vis, note, float(self.sat))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SER 真彩转色")
        self.resize(1180, 760)
        self.header = None
        self.worker = None
        self.preview_worker = None
        self._preview_base = None
        self._build()

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout = QVBoxLayout(root)
        layout.addWidget(split, 1)

        left = QWidget()
        form = QVBoxLayout(left)
        files = QGroupBox("文件")
        fl = QFormLayout(files)
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()
        btn_in = QPushButton("打开输入…")
        btn_out = QPushButton("另存为…")
        btn_in.clicked.connect(self.pick_input)
        btn_out.clicked.connect(self.pick_output)
        row_in, row_out = QHBoxLayout(), QHBoxLayout()
        row_in.addWidget(self.in_edit, 1)
        row_in.addWidget(btn_in)
        row_out.addWidget(self.out_edit, 1)
        row_out.addWidget(btn_out)
        in_box, out_box = QWidget(), QWidget()
        in_box.setLayout(row_in)
        out_box.setLayout(row_out)
        fl.addRow("输入", in_box)
        fl.addRow("输出", out_box)
        self.meta = QLabel("尚未打开文件")
        self.meta.setWordWrap(True)
        fl.addRow(self.meta)
        form.addWidget(files)

        opt = QGroupBox("转色")
        ol = QFormLayout(opt)
        self.pattern = QComboBox()
        self.pattern.addItems(["自动", "RGGB", "GRBG", "GBRG", "BGGR"])
        self.algo = QComboBox()
        for a in ALGORITHMS:
            mark = " ★" if a.recommend else ""
            self.algo.addItem(f"{a.label}  ·  {a.source}{mark}", a.key)
        self.algo.currentIndexChanged.connect(self._algo_changed)
        self.depth = QComboBox()
        self.depth.addItem("16 bit/通道（48bit RGB SER）", 16)
        self.depth.addItem("8 bit/通道（24bit RGB SER）", 8)
        self.frame0 = QSpinBox()
        self.frame1 = QSpinBox()
        self.frame0.setMaximum(999999)
        self.frame1.setMaximum(999999)
        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, os.cpu_count() or 4))
        self.workers.setValue(recommended_workers("ddfapd"))
        self.algo_note = QLabel()
        self.algo_note.setWordWrap(True)
        ol.addRow("Bayer", self.pattern)
        ol.addRow("算法", self.algo)
        ol.addRow(self.algo_note)
        ol.addRow("输出位深", self.depth)
        ol.addRow("起始帧", self.frame0)
        ol.addRow("结束帧(含)", self.frame1)
        ol.addRow("进程数", self.workers)
        self.wb_check = QCheckBox("亮部灰世界白平衡（月亮/行星会变成正常灰色，全片同一系数）")
        self.wb_check.setChecked(True)
        ol.addRow(self.wb_check)
        form.addWidget(opt)

        btns = QHBoxLayout()
        self.btn_preview = QPushButton("预览当前起始帧")
        self.btn_run = QPushButton("开始转成 RGB SER")
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_preview.clicked.connect(self.preview)
        self.btn_run.clicked.connect(self.start_convert)
        self.btn_cancel.clicked.connect(self.cancel_convert)
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)
        form.addLayout(btns)

        self.bar = QProgressBar()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        form.addWidget(self.bar)
        form.addWidget(self.log, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.preview_info = QLabel("预览（仅显示拉伸，不写入文件）")
        self.view = QLabel("打开 SER 后点预览")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(480, 400)
        self.view.setStyleSheet("background:#111; color:#888;")
        sat_row = QHBoxLayout()
        sat_row.addWidget(QLabel("预览饱和度（拉高可看网格）"))
        self.sat = QSlider(Qt.Orientation.Horizontal)
        self.sat.setRange(10, 50)
        self.sat.setValue(10)
        self.sat_val = QDoubleSpinBox()
        self.sat_val.setRange(1.0, 5.0)
        self.sat_val.setSingleStep(0.1)
        self.sat_val.setValue(1.0)
        self.sat.valueChanged.connect(lambda v: self.sat_val.setValue(v / 10))
        self.sat_val.valueChanged.connect(self._sat_changed)
        sat_row.addWidget(self.sat, 1)
        sat_row.addWidget(self.sat_val)
        rl.addWidget(self.preview_info)
        rl.addWidget(self.view, 1)
        rl.addLayout(sat_row)
        split.addWidget(right)
        split.setSizes([420, 760])

        help_act = QAction("关于 / 算法来源", self)
        help_act.triggered.connect(lambda: QMessageBox.information(self, "关于", HELP))
        self.menuBar().addAction(help_act)
        self.setAcceptDrops(True)
        self._algo_changed()
        self.log.appendPlainText(
            "转色引擎: DDFAPD(menon_fast) + OpenCV(VNG/EA/双线性)"
            + (" + Malvar(colour-demosaicing)" if "malvar" in ALG_BY_KEY else "（发行包不含 Malvar）")
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() in SUPPORTED_INPUT_SUFFIXES:
                self.in_edit.setText(path)
                self.load_input(path)
                break

    def _algo_changed(self) -> None:
        algo = ALG_BY_KEY[self.algo.currentData()]
        extra = ""
        if algo.max_output_depth == 8 and not (self.header and self.header.is_rgb):
            self.depth.setCurrentIndex(1)
            extra += "  该算法只能出 8bit。"
        self.workers.setValue(recommended_workers(algo.key))
        extra += f"  建议进程数 {self.workers.value()}。"
        self.algo_note.setText(algo.note + extra)

    def pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 Bayer 图像/视频",
            "",
            "支持的输入 (*.ser *.avi *.fit *.fits *.fts);;SER (*.ser);;AVI (*.avi);;FITS (*.fit *.fits *.fts)",
        )
        if path:
            self.in_edit.setText(path)
            self.load_input(path)

    def pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "输出 RGB SER", self.out_edit.text(), "SER (*.ser)")
        if path:
            self.out_edit.setText(path)

    def load_input(self, path: str) -> None:
        try:
            self.header = read_input_header(path)
        except Exception as exc:
            QMessageBox.critical(self, "无法读取", str(exc))
            return
        h = self.header
        gb = h.file_size / 1024**3
        self.meta.setText(
            f"{input_format_name(h)}  {h.width}×{h.height}  {h.frames} 帧  {h.depth}bit  {h.color_name}\n"
            f"{h.instrument or '(无相机名)'}  {gb:.2f} GB"
        )
        self.frame0.setMaximum(max(0, h.frames - 1))
        self.frame1.setMaximum(max(0, h.frames - 1))
        self.frame0.setValue(0)
        self.frame1.setValue(h.frames - 1)
        self.out_edit.setText(default_output_path(path, int(self.depth.currentData())))
        if h.bayer_pattern:
            self.pattern.setCurrentText("自动")
        elif h.can_debayer:
            self.pattern.setCurrentText("RGGB")
        if h.is_rgb and input_format_name(h) == "SER":
            self.log.appendPlainText("这个文件已经是 RGB/BGR SER。")
        elif h.is_rgb:
            self.wb_check.setChecked(False)
            self.log.appendPlainText(f"已打开 {os.path.basename(path)}  RGB FITS，将直接写入 RGB SER。")
        elif h.bayer_pattern:
            self.log.appendPlainText(f"已打开 {os.path.basename(path)}  Bayer={h.bayer_pattern}")
        else:
            self.log.appendPlainText(
                f"已打开 {os.path.basename(path)}  头信息是 {h.color_name}，"
                "仍按 Bayer 转色。阵列请自己核对（默认 RGGB）。"
            )

    def current_pattern(self) -> str:
        text = self.pattern.currentText()
        if text != "自动":
            return text
        if self.header and self.header.bayer_pattern:
            return self.header.bayer_pattern
        return "RGGB"

    def preview(self) -> None:
        path = self.in_edit.text().strip()
        if not path:
            return
        if self.preview_worker and self.preview_worker.isRunning():
            return
        self.view.setText("转色预览中…")
        self.preview_worker = PreviewWorker(
            path,
            self.frame0.value(),
            self.current_pattern(),
            self.algo.currentData(),
            int(self.depth.currentData()),
            float(self.sat_val.value()),
            self.wb_check.isChecked(),
        )
        self.preview_worker.ready.connect(self._show_preview)
        self.preview_worker.failed.connect(lambda m: QMessageBox.critical(self, "预览失败", m))
        self.preview_worker.start()

    def _sat_changed(self, value: float) -> None:
        if abs(self.sat.value() / 10 - value) > 0.05:
            self.sat.setValue(int(round(value * 10)))
        self._apply_preview_sat()

    def _apply_preview_sat(self) -> None:
        if self._preview_base is None:
            return
        sat = float(self.sat_val.value())
        vis = self._preview_base if abs(sat - 1.0) < 1e-3 else auto_stretch_sat(self._preview_base, sat)
        pix = to_pixmap(vis)
        self.view.setPixmap(
            pix.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _show_preview(self, vis, note: str, sat: float) -> None:
        self._preview_base = vis
        self.sat_val.setValue(sat)
        self.preview_info.setText(note + "  （显示已拉伸，文件仍是线性数据）")
        self._apply_preview_sat()

    def start_convert(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        path = self.in_edit.text().strip()
        out = self.out_edit.text().strip()
        if not path or not out:
            QMessageBox.warning(self, "缺少路径", "请先选择输入和输出 SER。")
            return
        try:
            header = read_input_header(path)
        except Exception as exc:
            QMessageBox.critical(self, "无法读取", str(exc))
            return
        if header.is_rgb and input_format_name(header) == "SER":
            QMessageBox.warning(self, "已经是 RGB", f"当前是 {header.color_name}，不用转色。")
            return
        if not header.is_rgb and not header.can_debayer:
            QMessageBox.warning(self, "无法转色", f"当前是 {header.color_name}，不是单通道 Bayer 数据。")
            return
        start = self.frame0.value()
        end = self.frame1.value()
        if end < start:
            QMessageBox.warning(self, "帧范围", "结束帧必须大于等于起始帧。")
            return
        count = end - start + 1
        depth = int(self.depth.currentData())
        algo = ALG_BY_KEY[self.algo.currentData()]
        if not header.is_rgb and algo.max_output_depth < depth:
            depth = algo.max_output_depth
            self.depth.setCurrentIndex(1)
        est = estimate_output_bytes(header, count, depth)
        disk = shutil_free(out)
        if disk is not None and est > disk:
            QMessageBox.critical(self, "磁盘空间不足", f"预计输出 {est/1024**3:.2f} GB，剩余 {disk/1024**3:.2f} GB")
            return
        self.log.appendPlainText(
            f"开始: {'RGB 直通' if header.is_rgb else algo.label}  {self.current_pattern()}  "
            f"{depth}bit  帧 {start}-{end}  约 {est/1024**3:.2f} GB"
        )
        self.bar.setValue(0)
        self.bar.setMaximum(count)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.worker = Worker(
            dict(
                src_path=path,
                dst_path=out,
                pattern=self.current_pattern(),
                algo_key=algo.key,
                output_depth=depth,
                start=start,
                count=count,
                workers=self.workers.value(),
                white_balance=self.wb_check.isChecked(),
            )
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(self._on_fail)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_progress(self, i: int, n: int, msg: str) -> None:
        if n:
            self.bar.setMaximum(n)
            self.bar.setValue(i)
        self.log.appendPlainText(msg)

    def _on_fail(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log.appendPlainText(msg)
        QMessageBox.critical(self, "转换失败", msg[-1500:])

    def _on_done(self, path: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.bar.setValue(self.bar.maximum() or 1)
        QMessageBox.information(self, "完成", f"已写出:\n{path}\n\nAS!4 打开后不要再 Debayer。")

    def cancel_convert(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.log.appendPlainText("正在取消…")


def shutil_free(path: str) -> int | None:
    import shutil

    try:
        drive = os.path.splitdrive(os.path.abspath(path))[0] or os.path.abspath(path)
        return shutil.disk_usage(drive).free
    except OSError:
        return None


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.splitext(sys.argv[1])[1].lower() in SUPPORTED_INPUT_SUFFIXES:
        win.in_edit.setText(sys.argv[1])
        win.load_input(sys.argv[1])
    sys.exit(app.exec())
