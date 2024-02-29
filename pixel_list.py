import pickle
import gzip
import tifffile
import numpy as np
from skimage import measure
import pandas as pd
from PyQt6.QtWidgets import QFileDialog, QApplication
import argparse
import zarr
from pathlib import Path

app = QApplication([])
# Add argparse logic to pass the file path and -g to prompt a GUI to select the file
parser = argparse.ArgumentParser(description="Save binary images as a list of pixels.")
# Add option -s to select an image sequence folder
parser.add_argument("-s", action="store_true", help="Select an image sequence folder")
# Add -i if image needs to be inverted
parser.add_argument("-i", action="store_true", help="Invert the image")
# Add a argument to specify if the process should be done in 3D, telling the user it may increase time and memory usage a lot
parser.add_argument(
    "-3d",
    action="store_true",
    help="Process the image in 3D\n Warning: This may highly increase time and memory usage.",
)

# -c to compress the image and -x to extract the image
parser.add_argument("-c", action="store_true", help="Compress the image")
parser.add_argument("-x", action="store_true", help="Extract the image")
args = parser.parse_args()


# If the -g option is used, prompt a GUI to select the file
def compress():
    if args.s:
        file_path = QFileDialog.getExistingDirectory(None, "Select a folder")
        images = zarr.open(tifffile.TiffSequence(file_path).aszarr())
        print(f"Reading images from {file_path}")
        print(f"Images shape: {images.shape}")
    else:
        file_path = QFileDialog.getOpenFileName(
            None, "Select a file", "", "TIFF Files (*.tif)"
        )[0]
        images = zarr.open(tifffile.TiffFile(file_path).aszarr())
        print(f"Reading images from {file_path}")
        print(f"Images shape: {images.shape}")

        labeled_image = measure.label(images)
        properties = measure.regionprops_table(labeled_image, properties=("coords",))
        df = pd.DataFrame(properties)

        save_path = Path(
            QFileDialog.getSaveFileName(
                None, "Where to save the file", "", "GZIP Files (*pkl.gz)"
            )[0]
        )

        data = {"df": df, "shape": images.shape}
        with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
            pickle.dump(data, f)


def extract():
    file_path = QFileDialog.getOpenFileName(
        None, "Select a file", "", "GZIP Files (*.gz)"
    )[0]
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
        print(data)
        df = data["df"]
        shape = data["shape"]
    reconstructed_image = np.zeros(shape, dtype=np.uint8)
    for index, row in df.iterrows():
        # Get the coordinates for the current region
        coords = row["coords"]
        for coord in coords:
            x, y, z = coord
            reconstructed_image[x, y, z] = 255

    save_path = Path(
        QFileDialog.getSaveFileName(
            None, "Where to save the file", "", "TIFF Files (*.tif)"
        )[0]
    )
    tifffile.imwrite(
        f"{save_path}", reconstructed_image, imagej=True, metadata={"axes": "ZYX"}
    )


if args.c:
    compress()

if args.x:
    extract()
