# plugins/Equalizer%204.3/tools.py

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QInputDialog,
    QFormLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
)
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QValidator
from PyQt6.QtCore import QLocale
from typing import Union


def equalX(mat, bkg, ref_mat, ref_bkg):
    E3 = (mat - bkg) / (ref_mat - ref_bkg)
    X = bkg - ref_bkg * E3
    X = round(X)
    return X


def equalY(mat, bkg, ref_mat, ref_bkg, X):
    E3 = (mat - bkg) / (ref_mat - ref_bkg)
    Y = bkg + E3 * (255 - ref_bkg)
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
                self.ref_mat_edit = QLineEdit()
                self.ref_mat_edit.setValidator(double_validator)
                form_layout.addRow("Reference material:", self.ref_mat_edit)

                self.ref_bkg_edit = QLineEdit()
                self.ref_bkg_edit.setValidator(double_validator)
                form_layout.addRow("Reference background:", self.ref_bkg_edit)

                self.t_mat_edit = QLineEdit()
                self.t_mat_edit.setValidator(double_validator)
                form_layout.addRow("Threshold material:", self.t_mat_edit)

                self.t_bkg_edit = QLineEdit()
                self.t_bkg_edit.setValidator(double_validator)
                form_layout.addRow("Threshold background:", self.t_bkg_edit)

                self.delta_edit = QLineEdit()
                self.delta_edit.setValidator(double_validator)
                form_layout.addRow("Tolerance:", self.delta_edit)

                self.start_slice_edit = QLineEdit()
                self.start_slice_edit.setValidator(QIntValidator())
                form_layout.addRow("Start slice:", self.start_slice_edit)

                self.end_slice_edit = QLineEdit()
                self.end_slice_edit.setValidator(QIntValidator())
                form_layout.addRow("End slice::", self.end_slice_edit)

                self.fix_bkg_ROI_check = QCheckBox("Fixed Background ROI")
                form_layout.addRow(self.fix_bkg_ROI_check)

                self.fix_mat_ROI_check = QCheckBox("Fixed Material ROI")
                form_layout.addRow(self.fix_mat_ROI_check)

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
                    self.ref_mat_edit, "Reference material", float
                ):
                    return False
                if not self.validate_field(
                    self.ref_bkg_edit, "Reference background", float
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
                if not self.validate_field(self.delta_edit, "Tolerance", float):
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
                    "ref_mat_original": float(self.ref_mat_edit.text()),
                    "ref_bkg_original": float(self.ref_bkg_edit.text()),
                    "t_mat": float(self.t_mat_edit.text()),
                    "t_bkg": float(self.t_bkg_edit.text()),
                    "delta": float(self.delta_edit.text()),
                    "start_slice": int(self.start_slice_edit.text()),
                    "end_slice": int(self.end_slice_edit.text()),
                    "fix_bkg_ROI": self.fix_bkg_ROI_check.isChecked(),
                    "fix_mat_ROI": self.fix_mat_ROI_check.isChecked(),
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
