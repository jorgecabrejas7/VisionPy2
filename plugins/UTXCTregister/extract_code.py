import json
import sys
from pathlib import Path


def extract_code_from_notebook(notebook_path, output_py_path):
    """
    Extracts Python code from a Jupyter Notebook and writes it to a .py file.

    Parameters:
        notebook_path (str): Path to the .ipynb file.
        output_py_path (str): Path to the .py file to create.
    """
    try:
        # Read the notebook file
        with open(notebook_path, "r", encoding="utf-8") as nb_file:
            notebook_data = json.load(nb_file)

        # Extract code cells
        code_cells = [
            cell["source"]
            for cell in notebook_data.get("cells", [])
            if cell["cell_type"] == "code"
        ]

        # Flatten the list of source lines and concatenate them
        code = "\n\n".join(["".join(cell) for cell in code_cells])

        # Write the extracted code to the .py file
        with open(output_py_path, "w", encoding="utf-8") as py_file:
            py_file.write("# Extracted Python code from notebook\n")
            py_file.write(code)

        print(f"Code extracted and written to {output_py_path}")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_code.py <notebook_path> <output_py_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_py_path = sys.argv[2]

    # Validate paths
    if not Path(notebook_path).exists():
        print(f"Notebook file '{notebook_path}' does not exist.")
        sys.exit(1)

    extract_code_from_notebook(notebook_path, output_py_path)
