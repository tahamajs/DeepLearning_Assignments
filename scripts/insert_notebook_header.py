#!/usr/bin/env python3
"""
Insert a top markdown cell into multiple Jupyter notebooks with course/author info.
Creates a .bak copy of each notebook before modifying.
"""
import json
from pathlib import Path

NOTEBOOKS = [
    "CA3_Object_Detection/Oriented_RCNN/code/NNDL_CA3_2.ipynb",
    "CA3_Object_Detection/Fast_SCNN/code/NNDL_CA3_1.ipynb",
    "CA4_Sequence_Modeling/Image_Captioning/code/nndl-ca4-1.ipynb",
    "CA7_Advanced_Topics/Image_Captioning/code/NNDL_CAe_2.ipynb",
    "CA7_Advanced_Topics/CNN_VIT_Adversarial_Attack/NNDL_CAe_1.ipynb",
    "CA4_Sequence_Modeling/Time_Series_Prediction/code/NNDL_CA4_2_1.ipynb",
    "CA1_Neural_Networks_Basics/code/NNDL_CA1_Q1.ipynb",
    "CA7_Advanced_Topics/CNN_VIT_Adversarial_Attack/code/NNDL_CAe_1.ipynb",
    "CA5_Vision_Transformers/VIT_Classification/code/NNDL_CA5_1.ipynb",
    "CA6_Generative_Models/Unsupervised_Domain_Adaptation_GAN/code/NNDL_CA6_1.ipynb",
    "CA6_Generative_Models/VAE/code/NNDL_CA6_2.ipynb",
    "CA2_CNN_Applications/Vehicle_Classification/code/NNDL_CA2_2.ipynb",
    "CA2_CNN_Applications/Vehicle_Classification/code/NNDL_CA2_2_normalized.ipynb",
    "CA5_Vision_Transformers/CLIP_Adversarial_Attack/code/NNDL_CA5_2.ipynb",
    "CA2_CNN_Applications/Covid_Detection/code/NNDL_CA2_1.ipynb",
]

ROOT = Path(__file__).resolve().parents[1]

# Assumption: use this exact course name. If you'd like a different phrasing, update COURSE_NAME.
COURSE_NAME = "Neural Networks and Deep Learning"
UNIVERSITY = "University of Tehran"
AUTHOR = "Mohammad Taha Majlesi"
STUDENT_ID = "810101504"

# HTML header block provided by the user. This will be inserted as a markdown cell so the HTML renders in notebooks.
HEADER_HTML = (
    '<div style="display:block;width:100%;margin:auto;" direction=rtl align=center><br><br>'
    '    <div  style="width:100%;margin:100;display:block;background-color:#fff0;"  display=block align=center>'
    '        <table style="border-style:hidden;border-collapse:collapse;">             <tr>'
    '                <td  style="border: none!important;">'
    '                    <img width=130 align=right src="https://i.ibb.co/yXKQmtZ/logo1.png" style="margin:0;" />'
    '                </td>'
    '                <td style="text-align:center;border: none!important;">'
    '                    <h1 align=center><font size=5 color="#025F5F"> <b>Deep Generative Models</b><br><br>Project 3 DDPM and DDIM implementation </i></font></h1>'
    '                </td>'
    '                <td style="border: none!important;">'
    '                    <img width=170 align=left  src="https://i.ibb.co/wLjqFkw/logo2.png" style="margin:0;" />'
    '                </td>'
    '           </tr>'
    '</div>'
    '        </table>'
    '    </div>'
)

# A small Topics placeholder cell inserted after the HTML header.
TOPICS_MARKDOWN = "## Topics\n\n- Add topic 1\n- Add topic 2\n- Add topic 3\n"

html_cell = {"cell_type": "markdown", "metadata": {}, "source": [HEADER_HTML]}
topics_cell = {"cell_type": "markdown", "metadata": {}, "source": [TOPICS_MARKDOWN]}


def insert_header(nb_path: Path):
    if not nb_path.exists():
        return (False, f"missing: {nb_path}")
    try:
        text = nb_path.read_text(encoding="utf-8")
        nb = json.loads(text)
    except Exception as e:
        return (False, f"json-error: {e}")

    cells = nb.get("cells", [])
    # If first cell already contains the HTML header marker or university/student id, skip.
    if cells:
        first_src = "".join(cells[0].get("source", [])).lower()
        if (
            "deep generative models" in first_src
            or "project 3 ddpm" in first_src
            or UNIVERSITY.lower() in first_src
            or AUTHOR.lower() in first_src
            or STUDENT_ID in first_src
        ):
            return (False, "already-has-header")

    # Backup
    bak = nb_path.with_suffix(nb_path.suffix + ".bak")
    bak.write_text(text, encoding="utf-8")

    # Insert at top: first the HTML header, then the Topics placeholder, then existing cells
    nb["cells"] = [html_cell, topics_cell] + cells

    try:
        nb_path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except Exception as e:
        # restore backup
        bak.rename(nb_path)
        return (False, f"write-error: {e}")

    return (True, "updated")


if __name__ == "__main__":
    results = {}
    for rel in NOTEBOOKS:
        p = ROOT / rel
        ok, msg = insert_header(p)
        results[str(rel)] = (ok, msg)
    # Print summary
    for k, (ok, msg) in results.items():
        status = "OK" if ok else "SKIP"
        print(f"{status}: {k} -> {msg}")
    # Exit code non-zero if any failed (excluding skips)
    any_failed = any(
        not ok and msg not in ("already-has-header", "missing: " + str(ROOT / ""))
        for ok, msg in results.values()
    )
    if any_failed:
        raise SystemExit(2)
