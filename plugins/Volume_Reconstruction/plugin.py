from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from base_plugin import BasePlugin
from plugins.Volume_Reconstruction.tools import get_dialog, get_bscans
from plugins.Volume_Reconstruction.load_bin_file import load_bin_file
import logging


class Plugin(BasePlugin):
    """
    json": self.checkbox_json.isChecked(),
    "raw": self.checkbox_raw.isChecked(),
    "dsp": self.checkbox_dsp.isChecked(),
    "cf": self.checkbox_cf.isChecked(),
    "pci": self.checkbox_pci.isChecked(),
    "name": self.line_edit.text(),
    """

    def execute(self):
        try:
            try:
                (
                    save_json,
                    compute_raw,
                    compute_dsp,
                    compute_pci,
                    compute_pci_image,
                    default_name,
                ) = self.request_gui(get_dialog, self.main_window)

            # Except the exception of receiving a None Object
            except TypeError:
                self.prompt_error("Error: Could not get the dialog.")
                logging.exception("Error: Could not get the dialog.")
                return

            file_path = self.select_file("Select Binary File")
            data, error = load_bin_file(file_path)

            if error < 0:
                self.prompt_error("Error: Could not read bin file succesfully.")

            # Get the B-Scans from the data
            bscans = get_bscans(
                data=data,
                compute_raw=compute_raw,
                compute_dsp=compute_dsp,
                compute_pci=compute_pci,
                compute_pci_image=compute_pci_image,
            )

            save_path = self.select_folder("Select Folder to Save Files")
            name = default_name if default_name else "bscans"
            json_name = default_name if default_name else "json_data"

            self.update_progress(0, "Saving Files...", 0, 100)
            if compute_raw:
                tifffile.imwrite(f"{save_path}/{name}_raw.tif", bscans["raw"])

            self.update_progress(20, "RAW Image Finished", 20, 100)

            if compute_dsp:
                tifffile.imwrite(f"{save_path}/{name}_dsp.tif", bscans["dsp"])

            self.update_progress(40, "DSP Image Finished", 40, 100)

            if compute_pci:
                tifffile.imwrite(f"{save_path}/{name}_pci.tif", bscans["pci"])

            self.update_progress(60, "PCI Image Finished", 60, 100)

            if compute_pci_image:
                tifffile.imwrite(
                    f"{save_path}/{name}_pci_image.tif", bscans["pci_image"]
                )

            self.update_progress(80, "PCI Image Finished", 80, 100)

            if save_json:
                import json

                with open(f"{save_path}/{json_name}.json", "w") as file:
                    file.write(json.dumps(data, indent=4))

            self.update_progress(100, "JSON File Finished", 100, 100)

            return  # No return value is expected

        except Exception as e:
            self.prompt_error(f"Error: {e}")
            logging.exception(f"Error: {e}")
            return
