# utils/progress_window.py


from PyQt6.QtWidgets import *
from PyQt6.QtCore import *


class ProgressWindow(QDialog):
    """
    A custom QDialog class that displays a progress window.

    Attributes:
        progress_bar (QProgressBar): The progress bar widget.
        progress_label (QLabel): The label widget displaying the progress message.

    Methods:
        update_progress(value, message): Updates the progress bar value and message.

    """

    def __init__(self, parent=None):
        """
        Initialize the Progress class.

        Args:
        parent: Optional; the parent widget.

        Raises:
        None.

        Example:
        ```
        progress = Progress(parent_widget)
        ```

        """
        super().__init__(parent)
        self.setWindowTitle("Progress")
        self.setGeometry(300, 300, 400, 100)
        self.layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar(self)
        self.layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Starting...", self)
        self.layout.addWidget(self.progress_label)
        self.time_label = QLabel("Estimated time remaining: Calculating...", self)
        self.layout.addWidget(self.time_label)
        self.start_time = QDateTime.currentDateTime()

    def update_progress(
        self, value: int, message: str, index: int = None, total: int = None
    ):
        """
        Update the progress bar and label with the given value and message.

        Args:
        - value (int): The value to set the progress bar to.
        - message (str): The message to display in the progress label.
        - index (int, optional): The current index of the progress. Defaults to None.
        - total (int, optional): The total number of items in the progress. Defaults to None.

        Raises:
        - No specific exceptions are raised by this function.

        Example:
        update_progress(50, "Processing item 5 of 10", 5, 10)
        """
    

        self.progress_bar.setValue(value)
        if message: 
            self.progress_label.setText(message)

        # Calculate elapsed time and estimate total time
        if index and total:
            elapsed = self.start_time.msecsTo(QDateTime.currentDateTime())
            if index > 0:
                estimated_total = elapsed / index * total
                remaining = estimated_total - elapsed
                remaining_time = (
                    QTime(0, 0).addMSecs(int(remaining)).toString("hh:mm:ss")
                )
                self.time_label.setText(f"Estimated time remaining: {remaining_time}")
            else:
                self.time_label.setText("Estimated time remaining: Calculating...")

        self.app.processEvents()
