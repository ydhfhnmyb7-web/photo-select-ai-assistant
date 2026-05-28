from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.core.app_logging import get_logger, setup_logging
from app.core.config import load_config
from app.ui.main_window import MainWindow


def main() -> int:
    setup_logging()
    load_config()

    def log_unhandled_exception(exc_type, exc, tb):
        get_logger().exception("程序异常退出", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = log_unhandled_exception

    app = QApplication(sys.argv)
    app.setApplicationName("PhotoSelect AI Assistant")

    window = MainWindow()
    window.resize(1280, 820)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
