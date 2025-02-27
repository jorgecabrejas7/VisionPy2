# plugins/Equalizer%204.3/tools.py

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
)
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtCore import QLocale
from typing import Union


def calculate_min(mat, bkg, ref_mat, ref_bkg):
    # Computing intermediate value E3
    E3 = (mat - bkg) / (ref_mat - ref_bkg)
    # Calculating X
    X = bkg - ref_bkg * E3
    # Rounding off the result
    X = round(X)

    return X


def calculate_max(mat, bkg, ref_mat, ref_bkg):
    # Computing intermediate value E3
    E3 = (mat - bkg) / (ref_mat - ref_bkg)
    # Calculating Y
    Y = bkg + E3 * (255 - ref_bkg)
    # Rounding off the result
    Y = round(Y)

    return Y


def create_equalization_settings_dialog(parent: QMainWindow) -> callable:
    def get_equalization_settings() -> dict[str, Union[float, int]]:
        class CustomDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Equalization Settings")
                self.create_layout()

            def create_layout(self):
                layout = QVBoxLayout(self)
                form_layout = QFormLayout()

                double_validator = QDoubleValidator(
                    notation=QDoubleValidator.Notation.StandardNotation
                )
                double_validator.setLocale(
                    QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
                )
                self.target_mat_edit = QLineEdit("210")
                self.target_mat_edit.setValidator(double_validator)
                form_layout.addRow("Target material value:", self.target_mat_edit)

                self.target_bkg_edit = QLineEdit("50")
                self.target_bkg_edit.setValidator(double_validator)
                form_layout.addRow("Target background value:", self.target_bkg_edit)

                self.t_mat_edit = QLineEdit("0.3")
                self.t_mat_edit.setValidator(double_validator)
                form_layout.addRow("Threshold material:", self.t_mat_edit)

                self.t_bkg_edit = QLineEdit("0.55")
                self.t_bkg_edit.setValidator(double_validator)
                form_layout.addRow("Threshold background:", self.t_bkg_edit)

                self.average_error_edit = QLineEdit("0.7")
                self.average_error_edit.setValidator(double_validator)
                form_layout.addRow("Tolerance:", self.average_error_edit)

                self.max_it = QLineEdit("10")
                self.max_it.setValidator(QIntValidator())
                form_layout.addRow("Max iterations/slice:", self.max_it)

                self.start_slice_edit = QLineEdit("1")
                self.start_slice_edit.setValidator(QIntValidator())
                form_layout.addRow("Start slice:", self.start_slice_edit)

                self.end_slice_edit = QLineEdit("4425")
                self.end_slice_edit.setValidator(QIntValidator())
                form_layout.addRow("End slice::", self.end_slice_edit)

                self.select_bkg_roi_check = QCheckBox("Manually Select Background ROI")
                self.select_bkg_roi_check.setChecked(True)
                form_layout.addRow(self.select_bkg_roi_check)

                self.select_mat_roi_check = QCheckBox("Manually Select Material ROI")
                self.select_mat_roi_check.setChecked(True)
                form_layout.addRow(self.select_mat_roi_check)

                self.histogram_matching_check = QCheckBox("Histogram Matching")
                self.histogram_matching_check.setChecked(False)
                form_layout.addRow(self.histogram_matching_check)

                layout.addLayout(form_layout)

                info_button = QPushButton("Info")
                info_button.clicked.connect(self.show_info)
                layout.addWidget(info_button)

                ok_button = QPushButton("OK")
                ok_button.clicked.connect(self.validate_and_accept)
                layout.addWidget(ok_button)

            def validate_and_accept(self):
                if self.validate_inputs():
                    self.accept()

            def validate_inputs(self):
                if not self.validate_field(
                    self.target_mat_edit, "Reference material", float
                ):
                    return False
                if not self.validate_field(
                    self.target_bkg_edit, "Reference background", float
                ):
                    return False
                if not self.validate_field(
                    self.t_mat_edit, "Threshold material", float
                ):
                    return False
                if not self.validate_field(
                    self.t_bkg_edit, "Threshold background", float
                ):
                    return False
                if not self.validate_field(self.average_error_edit, "Tolerance", float):
                    return False
                if not self.validate_field(self.max_it, "Max iterations/slice", int):
                    return False
                if not self.validate_field(self.start_slice_edit, "Start slice", int):
                    return False
                if not self.validate_field(self.end_slice_edit, "End slice", int):
                    return False
                return True

            def validate_field(self, field, name, data_type):
                try:
                    data_type(field.text())
                    return True
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error", f"Invalid input for {name}."
                    )
                    return False

            def show_info(self):
                QMessageBox.information(
                    self, "Information", "Explanation about each part of the dialog."
                )

            def get_values(self):
                return {
                    "target_material": float(self.target_mat_edit.text()),
                    "target_background": float(self.target_bkg_edit.text()),
                    "threshold_mat": float(self.t_mat_edit.text()),
                    "threshold_bkg": float(self.t_bkg_edit.text()),
                    "error_threshold": float(self.average_error_edit.text()),
                    "max_it": int(self.max_it.text()),
                    "start_slice": int(self.start_slice_edit.text()),
                    "end_slice": int(self.end_slice_edit.text()),
                    "select_roi_mat": self.select_bkg_roi_check.isChecked(),
                    "select_roi_bkg": self.select_mat_roi_check.isChecked(),
                    "histogram_matching": self.histogram_matching_check.isChecked(),
                }

        dialog = CustomDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_values()
        else:
            return None

    return get_equalization_settings


class VolumeModificationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Modification")
        self.user_choice = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        label = QLabel(
            "Do you want to duplicate the volume or apply changes directly to the volume on disk?"
        )
        layout.addWidget(label)

        duplicate_button = QPushButton("Duplicate Volume")
        duplicate_button.clicked.connect(self.duplicate_choice)
        layout.addWidget(duplicate_button)

        apply_button = QPushButton("Apply Directly")
        apply_button.clicked.connect(self.apply_choice)
        layout.addWidget(apply_button)

    def duplicate_choice(self):
        self.user_choice = "duplicate"
        self.accept()

    def apply_choice(self):
        self.user_choice = "apply"
        self.accept()

    def exec(self):
        super().exec()
        return self.user_choice


def create_file_settings_dialog(parent: QMainWindow):
    def get_file_settings():
        dialog = VolumeModificationDialog(parent)
        return dialog.exec()

    return get_file_settings


# create a dialog that ask the user for a star and end slice
def create_slice_range_dialog(parent: QMainWindow):
    def get_slice_range():
        class SliceRangeDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Slice Range for Z-Projection")
                self.create_layout()

            def create_layout(self):
                layout = QVBoxLayout(self)
                form_layout = QFormLayout()

                self.start_slice_edit = QLineEdit("1")
                self.start_slice_edit.setValidator(QIntValidator())
                form_layout.addRow("Start slice:", self.start_slice_edit)

                self.end_slice_edit = QLineEdit("4425")
                self.end_slice_edit.setValidator(QIntValidator())
                form_layout.addRow("End slice:", self.end_slice_edit)

                layout.addLayout(form_layout)

                ok_button = QPushButton("OK")
                ok_button.clicked.connect(self.validate_and_accept)
                layout.addWidget(ok_button)

            def validate_and_accept(self):
                if self.validate_inputs():
                    self.accept()

            def validate_inputs(self):
                if not self.validate_field(self.start_slice_edit, "Start slice", int):
                    return False
                if not self.validate_field(self.end_slice_edit, "End slice", int):
                    return False
                return True

            def validate_field(self, field, name, data_type):
                try:
                    data_type(field.text())
                    return True
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error", f"Invalid input for {name}."
                    )
                    return False

            def get_values(self):
                return {
                    "start_slice": int(self.start_slice_edit.text()),
                    "end_slice": int(self.end_slice_edit.text()),
                }

        dialog = SliceRangeDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.get_values()
        else:
            return None

    return get_slice_range
