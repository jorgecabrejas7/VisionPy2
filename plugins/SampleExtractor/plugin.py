from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from utils import image_sequence
from plugins.SampleExtractor.sampleextractor import process_volume


class Plugin(BasePlugin):
    def run(self):
        # Prompt the user to select a folder containing the volume
        file_path = self.select_folder("Select Folder Containing Volume")

        # Prompt the user to enter the number of samples
        n_samples, ok = self.ask_number("Enter Number of Samples", True)
        if not ok:
            self.prompt_error("Invalid number of samples.")
            return

        # Prompt the user to select folders to save each sample
        save_paths = []
        for i in range(n_samples):
            save_path = self.select_folder(f"Select Folder to Save Sample {i + 1}")
            if not save_path:
                self.prompt_error(f"Invalid folder for sample {i + 1}.")
                return
            save_paths.append(save_path)

        # Load the volume
        volume = image_sequence.read_sequence2(file_path, progress_window=self)
        self.update_progress(100, "Volume loaded successfully.")

        # Process the volume to extract samples
        self.update_progress(0, "Processing volume to extract samples.")
        samples = process_volume(volume, n_samples)
        self.update_progress(100, "Samples extracted successfully.")

        # Save the extracted samples
        for i, sample in enumerate(samples):
            sample_save_path = save_paths[i]
            image_sequence.write_sequence2(
                sample_save_path, "volume_eq", sample, progress_window=self
            )

        self.update_progress(100, "Samples saved successfully.")
        self.prompt_message("Samples saved successfully.")
