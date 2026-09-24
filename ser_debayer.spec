# -*- mode: python ; coding: utf-8 -*-
# 目录包（onedir），不要 onefile：
# 多进程转色会再启动本 exe；单文件解压到临时目录又慢又容易和 mmap 打架。

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "gui",
        "convert",
        "engines",
        "ser_io",
        "menon_fast",
        "multiprocessing",
        "multiprocessing.spawn",
        "multiprocessing.freeze_support",
        "cv2",
        "numpy",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "shiboken6",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "colour",
        "colour_demosaicing",
        "scipy",
        "matplotlib",
        "PIL",
        "IPython",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "pytest",
        "setuptools",
        "pip",
        "unittest",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SERDebayer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SERDebayer",
)
