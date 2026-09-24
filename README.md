# SERDebayer

把行星 / 月亮的 Bayer SER 转成真彩 RGB SER，给 [AutoStakkert!4](https://www.autostakkert.com/) 叠加。

AS!4 默认双线性 Debayer 在高饱和时容易出彩色网格。本工具在叠加前用更好的 demosaic 写出 24/48bit RGB SER；进 AS!4 后选 Auto Detect，**不要再 Force Debayer**。

## 运行

需要 Python 3.11+（开发时在 3.13 测过）：

```bat
python -m pip install -r requirements.txt
run.bat
```

也可以把 `.ser` 拖到窗口里，或：

```bat
python app.py 你的文件.ser
```

可选算法 Malvar 2004 需要另装 `colour-demosaicing`。默认的 DDFAPD / OpenCV 不需要。

## 算法

| 菜单 | 来源 | 说明 |
| --- | --- | --- |
| DDFAPD / Menon 2007 ★ | OpenCV float32（`menon_fast.py`） | 默认。抗网格，16bit |
| DDFAPD + 细化 | 同上 | 更慢，边缘更干净 |
| EA / VNG / 双线性 | OpenCV | VNG 只有 8bit |
| Malvar 2004 | colour-demosaicing（可选） | 未打进发行包 |

输出是 SER v3，`ColorID=100`（RGB）。16bit 源会写成 48bit RGB；也可以选 8bit/通道（24bit）。白平衡用亮部灰世界，全片同一组增益。

## 分发

不换 GUI 框架。Windows 上打目录包：

```bat
build.bat
```

把 `dist\SERDebayer` **整个文件夹**打成 zip 发出去。对方解压后运行 `SERDebayer.exe`，不需要安装 Python。不要只发那一个 exe。

单文件（onefile）不适合这个程序：转色会起多个进程，再解压临时目录又慢又容易和大文件 mmap 打架。

## 仓库结构

```
app.py              入口（打包后先 freeze_support，子进程不加载 Qt）
gui.py              PySide6 界面
convert.py          多进程整段转换 + mmap 写出
engines.py          调用 OpenCV / 可选 Malvar
menon_fast.py       Menon 2007 DDFAPD 的 OpenCV 加速
ser_io.py           SER v3 读写（ColorID、LittleEndian=0）
ser_debayer.spec    PyInstaller onedir
run.bat / build.bat 开发启动 / 打包
```

Fitswork / Siril CLI 不能作为本工具的 SER 转色引擎：Fitswork 没有命令行 SER 写出；Siril 1.4 的 `convert` 不会吃现成的 Bayer SER。

## 许可

源码按仓库内文件使用。OpenCV、PySide6、numpy 等仍遵循各自许可证。
