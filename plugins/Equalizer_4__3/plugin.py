import numpy as np

from base_plugin import BasePlugin
from plugins.Equalizer_4__3.tools import (
    create_equalization_settings_dialog,
    create_file_settings_dialog,
    equalX,
    equalY,
)
from utils.image_sequence import read_virtual_sequence


class Plugin(BasePlugin):
    def __init__(self, main_window, plugin_name):
        super().__init__(main_window, plugin_name)

    def execute(self):
        folder_path = self.select_folder("Select folder to load volume from")
        volume = read_virtual_sequence(folder_path)
        result = self.request_gui(create_equalization_settings_dialog(self.main_window))
        print(f"{result = }")
        file_settings = self.request_gui(create_file_settings_dialog(self.main_window))
        print(f"{file_settings = }")

        if result["start_slice"] < 1 or result["end_slice"] > volume.shape[0]:
            self.prompt_error("Slice out of bounds")
            self.finished.emit(self)
            return

        if not result["fix_mat_ROI"]:
            mat_roi_path = self.select_file("Select MAT ROI")
            if not mat_roi_path:
                self.prompt_error("No MAT ROI selected")
                self.finished.emit(self)
                return
            mat_roi = read_virtual_sequence(mat_roi_path)

        else:
            slice_mat, bbox_mat = self.get_volume_bbox(volume)
            mat_roi = volume[0].copy()
            mat_roi[:, :] = 0
            mat_roi[bbox_mat[0] : bbox_mat[1], bbox_mat[2] : bbox_mat[3]] = 1
            mat_roi = mat_roi.astype(bool)

        if not result["fix_bkg_ROI"]:
            bkg_roi_path = self.select_file("Select BKG ROI")
            if not bkg_roi_path:
                self.prompt_error("No BKG ROI selected")
                self.finished.emit(self)
                return
            bkg_roi = read_virtual_sequence(mat_roi_path)

        else:
            slice_bkg, bbox_bkg = self.get_volume_bbox(volume)
            bkg_roi = volume[0].copy()
            bkg_roi[:, :] = 0
            bkg_roi[bbox_bkg[0] : bbox_bkg[1], bbox_bkg[2] : bbox_bkg[3]] = 1
            bkg_roi = bkg_roi.astype(bool)

        for slice_index in range(result["start_slice"], result["end_slice"] + 1):
            slice_mat = (
                bool(mat_roi[slice_index])
                if not result["fix_mat_ROI"]
                else mat_roi.copy()
            )
            slice_bkg = (
                bool(bkg_roi[slice_index])
                if not result["fix_bkg_ROI"]
                else bkg_roi.copy()
            )

            current_slice = volume[slice_index]

            # Material values
            mat_max = np.max(current_slice[slice_mat])
            mat_min = np.min(current_slice[slice_mat])

            # Get material histogram
            mat_hist, mat_bins = np.histogram(
                current_slice[slice_mat], bins=256, range=(mat_min, mat_max)
            )

            max_mat = np.max(mat_hist)
            threshold_mat = max_mat * result["t_mat"]
            bin_centers = (mat_bins[:-1] + mat_bins[1:]) / 2
            mask = bin_centers > threshold_mat

            weighted_sum = np.sum(mat_hist[mask] * bin_centers[mask])
            total_count = np.sum(mat_hist[mask])

            mat_val = weighted_sum / total_count if total_count > 0 else 0

            # Repeat process for background
            bkg_max = np.max(current_slice[slice_bkg])
            bkg_min = np.min(current_slice[slice_bkg]) 

            bkg_hist, bkg_bins = np.histogram(
                current_slice[slice_bkg], bins=256, range=(bkg_min, bkg_max)
            )

            max_bkg = np.max(bkg_hist)
            threshold_bkg = max_bkg * result["t_bkg"]
            bin_centers = (bkg_bins[:-1] + bkg_bins[1:]) / 2
            mask = bin_centers > threshold_bkg

            weighted_sum = np.sum(bkg_hist[mask] * bin_centers[mask])
            total_count = np.sum(bkg_hist[mask])

            bkg_val = weighted_sum / total_count if total_count > 0 else 0

            # Equalization loop

            delta_mat = 2  # Difference between ref_mat_original and ref_mat calculated
            delta_bkg = 2  # Difference between ref_bkg_original and ref_bkg calculated
            delta_avg = 2  # Difference between delta average and and ref_bkg and ref_mat calculated
            n_it = 0
