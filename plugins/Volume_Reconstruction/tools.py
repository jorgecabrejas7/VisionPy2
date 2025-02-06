# Using pyqt6 create a dialog with 5 check boxes and a OK button
from typing import Any, Dict, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QCheckBox,
    QPushButton,
    QLineEdit,
)


class Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=None)

        layout = QVBoxLayout()

        self.checkbox_json = QCheckBox(
            "Save Binary Data as JSON? (May take a lot of space)"
        )
        layout.addWidget(self.checkbox_json)

        self.checkbox_raw = QCheckBox("Save RAW Image")
        layout.addWidget(self.checkbox_raw)

        self.checkbox_dsp = QCheckBox("Save DSP Image (RAW + TCG Curve)")
        layout.addWidget(self.checkbox_dsp)

        self.checkbox_cf = QCheckBox(
            "Store Coherence Factor Image ([0-1] Value for each pixel)"
        )
        layout.addWidget(self.checkbox_cf)

        self.checkbox_pci = QCheckBox(
            "Store PCI Image (DSP Image multiplied by Coherence Factor)"
        )
        layout.addWidget(self.checkbox_pci)

        # Add a field for the user to set a default name for the files
        self.line_edit = QLineEdit("Set a name for the files")
        layout.addWidget(self.line_edit)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

        self.setLayout(layout)

    # Add a method to retrieve the checkboxes values
    def get_checkboxes(self):
        return {
            "json": self.checkbox_json.isChecked(),
            "raw": self.checkbox_raw.isChecked(),
            "dsp": self.checkbox_dsp.isChecked(),
            "cf": self.checkbox_cf.isChecked(),
            "pci": self.checkbox_pci.isChecked(),
            "name": self.line_edit.text(),
        }


def get_dialog(parent=None) -> Tuple[bool, bool, bool, bool, str]:
    dialog = Dialog(parent=parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_checkboxes()

    return None


def get_bscans(
    data: Dict[str, Any],
    compute_raw: bool = True,
    compute_dsp: bool = True,
    compute_pci: bool = True,
    compute_pci_image: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extracts and processes different types of B-scans from given data, based on boolean flags.

    Parameters:
        data (dict): The input data containing imaging information and parameters.
        compute_raw (bool): If True, compute raw B-scans.
        compute_dsp (bool): If True, compute DSP B-scans.
        compute_pci (bool): If True, compute PCI B-scans.
        compute_pci_image (bool): If True, compute PCI image B-scans.

    Returns:
        Dict of numpy arrays: Returns a dictionary containing arrays for each B-scan type requested.
    """
    vch = data["virtual_channel_1"]
    n_ascans = vch["n_ascan"]
    n_samples = vch["n_samples"]
    depth = data["auxiliar_lines_number"] * data["trigger_lines_number"]
    total_pixels_bscan = n_ascans * n_samples

    bscans = {}

    for i in range(0, depth * total_pixels_bscan, total_pixels_bscan):
        if compute_raw:
            raw_segment = np.array(vch["scan_raw"][i : i + total_pixels_bscan]).reshape(
                n_ascans, n_samples
            )
            bscans.setdefault("raw", []).append(np.rot90(raw_segment))

        if compute_dsp:
            dsp_segment = np.array(vch["scan_dsp"][i : i + total_pixels_bscan]).reshape(
                n_ascans, n_samples
            )
            bscans.setdefault("dsp", []).append(np.rot90(dsp_segment))

        if compute_pci:
            pci_segment = np.array(vch["scan_pci"][i : i + total_pixels_bscan]).reshape(
                n_ascans, n_samples
            )
            bscans.setdefault("pci", []).append(np.rot90(pci_segment))

        if compute_pci_image:
            pci_img_segment = np.array(
                vch["scan_pci_img"][i : i + total_pixels_bscan]
            ).reshape(n_ascans, n_samples)
            bscans.setdefault("pci_image", []).append(np.rot90(pci_img_segment))

    # Convert lists to numpy arrays and normalize each type of B-scan to the 16-bit range
    def normalize(bscans_list):
        bscans_array = np.array(bscans_list)
        return (
            (bscans_array - bscans_array.min())
            / (bscans_array.max() - bscans_array.min())
            * 65535
        ).astype(np.uint16)

    for key in bscans.keys():
        bscans[key] = normalize(bscans[key])

    return bscans
