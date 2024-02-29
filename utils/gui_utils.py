from typing import List, Tuple

import matplotlib.pyplot as plt
import zarr
from matplotlib.widgets import RectangleSelector, Slider
from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


def virtual_sequence_bbox(zarr_array: zarr.Array) -> Tuple[int, List[int]]:
    """
    Processes a Zarr array, displaying each slice and allowing the user to select a
    bounding box on a specific slice. Captures the details via a dialog box.

    Args:
    zarr_array (zarr.Array): Zarr array representing image slices.

    Returns:
    Tuple[int, List[int]]: Tuple containing the index of the selected slice and the bounding box coordinates.
    """
    global bbox, current_slice, bbox_label
    bbox = None
    current_slice = 0
    bbox_label = None

    def onselect(eclick, erelease):
        """Callback for the rectangle selector to update the bounding box."""
        global bbox
        bbox = [
            int(eclick.xdata),
            int(eclick.ydata),
            int(erelease.xdata),
            int(erelease.ydata),
        ]
        update_dialog()

    def update_slice(val, img):
        global current_slice
        current_slice = int(slider.val)
        try:
            # Accessing the slice from the Zarr array
            slice_data = zarr_array[current_slice, :, :]
            img.set_data(slice_data)
            fig.canvas.toolbar.set_message(f"Slice: {current_slice}")
            fig.canvas.draw()
            update_dialog()
        except Exception as e:
            print(f"Error in update_slice: {e}")

    def change_slice(offset: int):
        """Change the current slice to be displayed."""
        global current_slice
        current_slice = (current_slice + offset) % len(zarr_array)
        update_slice()

    def update_dialog():
        global bbox_label
        if bbox_label is not None:
            bbox_text = (
                f"Slice: {current_slice}, Bounding Box: {bbox}"
                if bbox
                else f"Slice: {current_slice}"
            )
            bbox_label.setText(bbox_text)

    def capture_bbox_and_return() -> Tuple[int, List[int]]:
        """Capture the bounding box and return the results."""
        global bbox_label
        dlg = QDialog()
        dlg.setWindowTitle("Bounding Box Values")

        layout = QVBoxLayout()
        bbox_label = QLabel(
            f"Slice: {current_slice}, Bounding Box: {bbox}"
            if bbox
            else f"Slice: {current_slice}"
        )
        update_dialog()
        layout.addWidget(bbox_label)

        ok_button = QPushButton("OK", dlg)
        ok_button.clicked.connect(lambda: (dlg.accept(), plt.close(fig)))
        layout.addWidget(ok_button)

        dlg.setLayout(layout)
        dlg.show()

        plt.show(block=False)
        dlg.exec()

        return current_slice, bbox

    # Setting up Matplotlib figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img = ax.imshow(zarr_array[current_slice, :, :], cmap="gray")

    # Rectangle selector
    toggle_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider = Slider(
        ax_slider, "Slice", 0, len(zarr_array) - 1, valinit=0, valfmt="%0.0f"
    )

    slider.on_changed(lambda val: update_slice(val, img))

    def key_press(event):
        """Handle key press events to navigate through slices."""
        global current_slice
        if event.key == "left":
            current_slice = max(
                current_slice - 1, 0
            )  # Ensure slice index doesn't go below 0
        elif event.key == "right":
            current_slice = min(
                current_slice + 1, len(zarr_array) - 1
            )  # Ensure slice index doesn't exceed the maximum
        slider.set_val(current_slice)  # Update the slider
        update_slice(current_slice, img)  # Update the displayed slice

    # Connect the key press event handler
    fig.canvas.mpl_connect("key_press_event", key_press)

    return capture_bbox_and_return()


def virtual_sequence_slice(zarr_array: zarr.Array) -> int:
    """
    Processes a Zarr array, displaying each slice and allowing the user to select a slice.
    A dialog is shown to confirm the selected slice.

    Args:
    zarr_array (zarr.Array): Zarr array representing image slices.

    Returns:
    int: Index of the selected slice.
    """
    global current_slice, slice_label
    current_slice = 0
    slice_label = None

    def update_slice(val, img):
        global current_slice, slice_label
        current_slice = int(val)
        try:
            # Accessing the slice from the Zarr array
            slice_data = zarr_array[current_slice, :, :]
            img.set_data(slice_data)
            fig.canvas.toolbar.set_message(f"Slice: {current_slice}")
            fig.canvas.draw()
            if slice_label is not None:
                slice_label.setText(f"Slice: {current_slice}")
        except Exception as e:
            print(f"Error in update_slice: {e}")

    def key_press(event):
        """Handle key press events to navigate through slices."""
        global current_slice
        if event.key == "left":
            current_slice = max(
                current_slice - 1, 0
            )  # Ensure slice index doesn't go below 0
        elif event.key == "right":
            current_slice = min(
                current_slice + 1, len(zarr_array) - 1
            )  # Ensure slice index doesn't exceed the maximum
        slider.set_val(current_slice)  # Update the slider
        update_slice(current_slice, img)  # Update the displayed slice

    # Setting up Matplotlib figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img = ax.imshow(zarr_array[current_slice, :, :], cmap="gray")

    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider = Slider(
        ax_slider, "Slice", 0, len(zarr_array) - 1, valinit=0, valfmt="%0.0f"
    )

    slider.on_changed(lambda val: update_slice(val, img))

    # Connect the key press event handler
    fig.canvas.mpl_connect("key_press_event", key_press)

    # Dialog for confirming the selected slice
    dlg = QDialog()
    dlg.setWindowTitle("Confirm Slice")

    layout = QVBoxLayout()
    slice_label = QLabel(f"Slice: {current_slice}")
    layout.addWidget(slice_label)

    ok_button = QPushButton("OK", dlg)
    ok_button.clicked.connect(lambda: (dlg.accept(), plt.close(fig)))
    layout.addWidget(ok_button)

    dlg.setLayout(layout)
    dlg.show()

    plt.show(block=False)
    dlg.exec()

    return current_slice
