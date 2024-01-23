# /main.py

import sys
import logging

import matplotlib
from PyQt6.QtWidgets import QApplication
from views.main_window import MainWindow
matplotlib.use("QtAgg")

def main():
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

    app = QApplication(sys.argv)
    main_view = MainWindow()
    main_view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
