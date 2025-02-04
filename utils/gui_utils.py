from typing import List, Tuple

import matplotlib.pyplot as plt
import zarr
from matplotlib.widgets import RectangleSelector, Slider
from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.colors import Normalize


def virtual_sequence_bbox(zarr_array: zarr.Array):
    global bbox, current_slice, bbox_label
    bbox = None
    current_slice = 0
    bbox_label = None

    # Initialize dictionaries to store brightness and contrast for each slice
    brightness_settings = {}
    contrast_settings = {}

    # -- Callbacks --
    def onselect(eclick, erelease):
        global bbox
        bbox = [
            int(eclick.xdata),
            int(eclick.ydata),
            int(erelease.xdata),
            int(erelease.ydata),
        ]
        update_dialog()

    def update_dialog():
        global bbox_label
        if bbox_label is not None:
            bbox_text = (
                f"Slice: {current_slice}, Bounding Box: {bbox}"
                if bbox
                else f"Slice: {current_slice}"
            )
            bbox_label.setText(bbox_text)

    def update_slice(val):
        """Called when the slice slider is changed."""
        global current_slice
        current_slice = int(slice_slider.val)
        slice_data = zarr_array[current_slice, :, :]
        img.set_data(slice_data)

        # Update brightness and contrast sliders to the current slice's settings
        brightness_slider.set_val(brightness_settings.get(current_slice, 0.0))
        contrast_slider.set_val(contrast_settings.get(current_slice, 1.0))

        # Re-apply brightness/contrast to the display range
        apply_brightness_contrast()

        fig.canvas.toolbar.set_message(f"Slice: {current_slice}")
        fig.canvas.draw_idle()
        update_dialog()

    def apply_brightness_contrast():
        """
        Adjust brightness and contrast using PIL ImageEnhance.
        """
        # Get the current slice data
        slice_data = zarr_array[current_slice, :, :]

        # Normalize slice data to 0-255 for PIL processing
        slice_min, slice_max = slice_data.min(), slice_data.max()
        if slice_max > slice_min:
            normalized = (slice_data - slice_min) / (slice_max - slice_min) * 255.0
        else:
            normalized = slice_data * 0.0  # Handle uniform slices

        # Convert to uint8
        image_uint8 = normalized.astype(np.uint8)

        # Convert to PIL Image
        
        pil_image = Image.fromarray(image_uint8)

        # Retrieve brightness and contrast settings for the current slice
        brightness_val = brightness_settings.get(current_slice, 0.0)  # Percentage
        contrast_val = contrast_settings.get(current_slice, 1.0)      # Factor

        # Map brightness_val from -100 to +100 to 0.0 to 2.0
        brightness_factor = 1.0 + (brightness_val / 100.0)
        brightness_factor = max(brightness_factor, 0.0)  # Prevent negative factors

        # Apply brightness enhancement
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)

        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_val)

        # Convert back to numpy array
        adjusted = np.array(pil_image)

        # Update the displayed image
        img.set_data(adjusted)
        img.set_clim(0, 255)  # Since data is now in 0-255 range

    def key_press(event):
        """Handle key press to move left/right in the stack."""
        global current_slice
        if event.key == "left":
            current_slice = max(current_slice - 1, 0)
        elif event.key == "right":
            current_slice = min(current_slice + 1, len(zarr_array) - 1)
        slice_slider.set_val(current_slice)
        update_slice(current_slice)

    def update_brightness_contrast(val):
        """
        Called when brightness or contrast sliders change.
        Saves the current settings for the slice and re-applies the transform.
        """
        # Save the current brightness and contrast for the slice
        brightness_settings[current_slice] = brightness_slider.val
        contrast_settings[current_slice] = contrast_slider.val

        apply_brightness_contrast()
        fig.canvas.draw_idle()

    def capture_bbox_and_return():
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

    # -- Matplotlib figure setup --
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Initial slice
    slice_data = zarr_array[current_slice, :, :]
    img = ax.imshow(slice_data, cmap="gray")
    apply_brightness_contrast()  # Apply initial brightness and contrast

    # Rectangle selector
    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    # Slice slider
    ax_slice_slider = plt.axes([0.1, 0.22, 0.65, 0.03])
    slice_slider = Slider(
        ax_slice_slider,
        "Slice",
        0,
        len(zarr_array) - 1,
        valinit=0,
        valfmt="%0.0f",
    )
    slice_slider.on_changed(update_slice)

    # Brightness slider (representing -100% to +100% shifts)
    ax_brightness = plt.axes([0.1, 0.12, 0.65, 0.03])
    brightness_slider = Slider(
        ax_brightness,
        "Brightness (%)",
        -100.0,
        100.0,   # Represents -100% to +100% shift
        valinit=brightness_settings.get(current_slice, 0.0),
        valfmt="%1.0f%%",
    )
    brightness_slider.on_changed(update_brightness_contrast)

    # Contrast slider
    ax_contrast = plt.axes([0.1, 0.07, 0.65, 0.03])
    contrast_slider = Slider(
        ax_contrast,
        "Contrast",
        0.1,
        3.0,     # Adjust based on data range
        valinit=contrast_settings.get(current_slice, 1.0),
        valfmt="%1.2f",
    )
    contrast_slider.on_changed(update_brightness_contrast)

    # Key press for left/right arrows
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


def get_bbox(zarr_array) -> Tuple[List[int]]:
    """
    Processes a single Zarr array slice, displaying the slice and allowing the user to select a
    bounding box on it. Captures the details via a dialog box.

    Args:
        zarr_array (zarr.Array): Zarr array representing image slices.

    Returns:
        Tuple[List[int]]: Bounding box coordinates on the given slice.
    """
    global bbox, bbox_label
    bbox = None
    bbox_label = None

    def onselect(eclick, erelease):
        """Callback for the rectangle selector to update the bounding box."""
        global bbox
        bbox = [int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)]
        update_dialog()

    def update_dialog():
        """Update the dialog to show the current bounding box."""
        global bbox_label
        if bbox_label is not None:
            bbox_text = f"Bounding Box: {bbox}" if bbox else "No Bounding Box Selected"
            bbox_label.setText(bbox_text)

    def capture_bbox_and_return() -> Tuple[List[int]]:
        """Capture the bounding box and return the results."""
        global bbox_label
        dlg = QDialog()
        dlg.setWindowTitle("Bounding Box Values")

        layout = QVBoxLayout()
        bbox_label = QLabel("Bounding Box: {}".format(bbox) if bbox else "No Bounding Box Selected")
        update_dialog()
        layout.addWidget(bbox_label)

        ok_button = QPushButton("OK", dlg)
        ok_button.clicked.connect(lambda: (dlg.accept(), plt.close(fig)))
        layout.addWidget(ok_button)

        dlg.setLayout(layout)
        dlg.show()

        plt.show(block=False)
        dlg.exec()

        return bbox

    # Setting up Matplotlib figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # leave space for sliders

    orig_min, orig_max = zarr_array.min(), zarr_array.max()
    midpoint = (orig_min + orig_max) / 2.0

    # Initial normalization and display
    norm = Normalize(vmin=orig_min, vmax=orig_max)
    img = ax.imshow(zarr_array, cmap="gray", norm=norm)

    # Rectangle selector for bounding box selection
    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True
    )

    # Add sliders for brightness and contrast
    ax_brightness = fig.add_axes([0.25, 0.1, 0.50, 0.03])
    ax_contrast = fig.add_axes([0.25, 0.05, 0.50, 0.03])

    brightness_slider = Slider(ax_brightness, 'Brightness', -1.0, 1.0, valinit=0.0)
    contrast_slider = Slider(ax_contrast, 'Contrast', 0.1, 3.0, valinit=1.0)

    def update_image(val):
        # Get current slider values
        brightness = brightness_slider.val
        contrast = contrast_slider.val

        # Apply contrast by scaling around the midpoint
        adjusted = (zarr_array - midpoint) * contrast + midpoint

        # Apply brightness: shift by a fraction of the full range
        adjusted = adjusted + brightness * (orig_max - orig_min)

        # Clip to the valid range
        adjusted = np.clip(adjusted, orig_min, orig_max)

        # Update the displayed image
        img.set_data(adjusted)
        fig.canvas.draw_idle()

    # Connect sliders to the update callback
    brightness_slider.on_changed(update_image)
    contrast_slider.on_changed(update_image)

    return capture_bbox_and_return()