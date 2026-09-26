"""PyInstaller entry; worker subprocesses re-enter the same executable."""
import multiprocessing
from facemap_desktop.__main__ import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
