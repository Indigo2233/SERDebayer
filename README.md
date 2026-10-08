# SERDebayer

把行星 / 月亮的 Bayer SER、8-bit 灰度 Bayer AVI、8/16-bit FITS 转成真彩 RGB SER，给 [AutoStakkert!4](https://www.autostakkert.com/) 叠加。头信息写成 MONO 的单通道文件同样会 Debayer，阵列在界面里选（默认 RGGB）。

AS!4 默认双线性 Debayer 在高饱和时容易出彩色网格。本工具在叠加前用更好的 demosaic 写出 24/48bit RGB SER；进 AS!4 后选 Auto Detect，**不要再 Force Debayer**。

## 运行

需要 Python 3.11+（开发时在 3.13 测过）：

```bat
python -m pip install -r requirements.txt
run.bat
```

也可以把 `.ser`、`.avi`、`.fit`、`.fits` 或 `.fts` 拖到窗口里，或：

```bat
python app.py 你的文件.ser
```

输入范围：

- SER：8/16-bit 单通道 Bayer/MONO。
- AVI：8-bit 灰度 Bayer 视频。普通 RGB/YUV 彩色视频会被拒绝。
- FITS：8/16-bit 二维 Bayer/MONO、三维单通道帧序列，以及形状为 `3×H×W` 或 `H×W×3` 的 RGB 图像。
- FITS 中的 `BAYERPAT`、`BAYERPATN` 或 `BAYER` 会用于自动识别阵列。

三通道 FITS 会跳过 Debayer，直接写入 RGB SER。形状为 `3×H×W` 的三帧灰度 FITS 存在布局歧义，当前按 RGB 解释。

## 命令行

CLI 不加载 Qt，适合脚本和 agent 调用：

```bat
python cli.py info input.fits --json
python cli.py convert input.avi -o output.ser --pattern RGGB --algorithm ddfapd --output-depth 16
python cli.py convert input.fits --pattern auto --output-depth 8 --no-white-balance --json
```

也可以通过统一入口调用：

```bat
python app.py info input.fits --json
python app.py convert input.fits --pattern RGGB --overwrite
```

`convert` 默认转换全部帧。Bayer 输入默认启用亮部灰世界白平衡，RGB FITS 默认保持原通道。可用 `--start`、`--count` 或 `--end` 选择帧范围；现有输出需要 `--overwrite` 才会覆盖。运行 `python cli.py convert --help` 查看完整参数。

可选算法 Malvar 2004 需要另装 `colour-demosaicing`。默认的 DDFAPD / OpenCV 不需要。

## 算法

| 菜单 | 来源 | 说明 |
| --- | --- | --- |
| DDFAPD / Menon 2007 ★ | OpenCV float32（`menon_fast.py`） | 默认。抗网格，16bit |
| DDFAPD + 细化 | 同上 | 更慢，边缘更干净 |
| EA / VNG / 双线性 | OpenCV | VNG 只有 8bit |
| Malvar 2004 | colour-demosaicing（可选） | 未打进发行包 |

输出是 SER v3，`ColorID=100`（RGB）。可以选择 16bit/通道（48bit RGB）或 8bit/通道（24bit RGB）。白平衡用亮部灰世界，全片同一组增益。

## 分发

不换 GUI 框架。Windows 上打目录包：

```bat
build.bat
```

把 `dist\SERDebayer` **整个文件夹**打成 zip 发出去。对方解压后运行 `SERDebayer.exe` 使用 GUI，运行 `SERDebayerCLI.exe` 使用命令行。

单文件（onefile）不适合这个程序：转色会起多个进程，再解压临时目录又慢又容易和大文件 mmap 打架。

## 仓库结构

```
app.py              入口（打包后先 freeze_support，子进程不加载 Qt）
gui.py              PySide6 界面
cli.py              info / convert 命令行入口
input_io.py         SER / AVI / FITS 统一读取
convert.py          SER 多进程转换、AVI/FITS 顺序转换
engines.py          调用 OpenCV / 可选 Malvar
menon_fast.py       Menon 2007 DDFAPD 的 OpenCV 加速
ser_io.py           SER v3 读写（ColorID、LittleEndian=0）
ser_debayer.spec    PyInstaller onedir
run.bat / build.bat 开发启动 / 打包
```

Fitswork / Siril CLI 不能作为本工具的 SER 转色引擎：Fitswork 没有命令行 SER 写出；Siril 1.4 的 `convert` 不会吃现成的 Bayer SER。

## 许可

源码按仓库内文件使用。OpenCV、PySide6、numpy 等仍遵循各自许可证。
