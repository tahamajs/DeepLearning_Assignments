#!/usr/bin/env python3
"""Extract images from Jupyter notebook and save them."""

import json
import base64
import os
from pathlib import Path


def extract_images_from_notebook(notebook_path, output_dir):
    """Extract all PNG images from notebook outputs."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    image_count = 0
    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        if "outputs" not in cell:
            continue

        for output_idx, output in enumerate(cell["outputs"]):
            if "data" in output and "image/png" in output["data"]:
                image_data = output["data"]["image/png"]
                image_bytes = base64.b64decode(image_data)

                image_filename = f"image_cell{cell_idx}_output{output_idx}.png"
                image_path = os.path.join(output_dir, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                image_count += 1
                print(f"Extracted: {image_filename}")

    print(f"\nTotal images extracted: {image_count}")
    return image_count


if __name__ == "__main__":
    notebook_path = "code/NNDL_CAe_2.ipynb"
    output_dir = "images"

    if os.path.exists(notebook_path):
        extract_images_from_notebook(notebook_path, output_dir)
    else:
        print(f"Notebook not found: {notebook_path}")
