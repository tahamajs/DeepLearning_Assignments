#!/usr/bin/env python3
"""
Script to extract images from Jupyter notebook outputs
"""
import json
import base64
import os
from pathlib import Path


def extract_images_from_notebook(notebook_path, output_dir):
    """Extract all images from notebook outputs"""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_count = 0

    for cell_idx, cell in enumerate(notebook["cells"]):
        if "outputs" in cell:
            for output_idx, output in enumerate(cell["outputs"]):
                if output.get("output_type") == "display_data":
                    data = output.get("data", {})
                    # Check for PNG images
                    if "image/png" in data:
                        image_data = data["image/png"]
                        if isinstance(image_data, str):
                            # Base64 encoded image
                            image_bytes = base64.b64decode(image_data)
                            image_filename = (
                                output_dir
                                / f"image_cell{cell_idx}_output{output_idx}.png"
                            )
                            with open(image_filename, "wb") as img_file:
                                img_file.write(image_bytes)
                            image_count += 1
                            print(f"Extracted: {image_filename}")

    print(f"\nTotal images extracted: {image_count}")
    return image_count


if __name__ == "__main__":
    notebook_path = "code/NNDL_CA6_1.ipynb"
    output_dir = "images"
    extract_images_from_notebook(notebook_path, output_dir)
