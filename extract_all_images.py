#!/usr/bin/env python3
"""Extract images from all Jupyter notebooks and save them in organized folders."""

import json
import base64
import os
from pathlib import Path
import glob


def extract_images_from_notebook(notebook_path, output_dir):
    """Extract all PNG images from notebook outputs."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    image_count = 0
    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        if "outputs" not in cell:
            continue

        for output_idx, output in enumerate(cell["outputs"]):
            if "data" in output and "image/png" in output["data"]:
                image_data = output["data"]["image/png"]
                image_bytes = base64.b64decode(image_data)

                image_filename = f"image_cell{cell_idx:03d}_output{output_idx:03d}.png"
                image_path = os.path.join(output_dir, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                image_count += 1
                print(f"  Extracted: {image_filename}")

    return image_count


def main():
    """Extract images from all notebooks in the repository."""
    base_dir = Path(__file__).parent

    # Find all notebooks
    notebooks = glob.glob(str(base_dir / "**" / "*.ipynb"), recursive=True)

    # Filter out checkpoint notebooks
    notebooks = [n for n in notebooks if ".ipynb_checkpoints" not in n]

    print(f"Found {len(notebooks)} notebooks\n")

    total_images = 0
    for notebook_path in notebooks:
        rel_path = Path(notebook_path).relative_to(base_dir)
        print(f"Processing: {rel_path}")

        # Create output directory structure matching notebook location
        notebook_dir = Path(notebook_path).parent
        output_dir = notebook_dir / "notebook_images"

        count = extract_images_from_notebook(notebook_path, output_dir)
        total_images += count
        print(f"  Total: {count} images\n")

    print(f"\n{'='*50}")
    print(f"Extraction complete! Total images: {total_images}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
