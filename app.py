"""入口。必须先 freeze_support，打包后的转色子进程才不会再加载 Qt。"""

from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    from gui import main

    main()
