"""入口。必须先 freeze_support，打包后的转色子进程才不会再加载 Qt。"""

import os
import sys
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    cli_mode = (len(sys.argv) > 1 and sys.argv[1] in {"info", "convert"}) or os.path.basename(
        sys.executable
    ).lower().startswith("serdebayercli")
    if cli_mode:
        from cli import main

        raise SystemExit(main())
    from gui import main as gui_main

    gui_main()
