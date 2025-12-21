Report build instructions

To build the IEEE-style report PDF:

1. Ensure you have a LaTeX distribution installed (TeX Live or MikTeX) and `pdflatex` and `bibtex` available in PATH.
2. From the `report/` folder run:

```bash
make all
```

3. The generated `report.pdf` will include figures referenced from the project `codes/*/images/ieee_assets/` directory. Ensure the `.pdf` figure files exist there before building.

Notes
- If figures are only available as PNGs, either convert them to PDF or change the `\includegraphics{}` references in `report.tex` to the PNG filenames. The `codes/*/images` folders contain the generated high-resolution PNGs and the `ieee_assets/` subfolders hold copied PDFs and per-figure `.tex` snippets produced by the asset helper scripts.
- For IEEE double-column figures, the default snippets use 0.48\textwidth width; tweak where appropriate.
