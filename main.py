# /main.py

import sys
from PyQt6.QtWidgets import QApplication
from views.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    main_view = MainWindow()
    main_view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
