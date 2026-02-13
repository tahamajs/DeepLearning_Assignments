# CAe Report Completion Notes

## Scope

This note completes report evidence for CAe by attaching generated metric plots from existing result CSVs.

## Report Sources

- Main LaTeX report source: `report.tex`
- Build helper: `Makefile`
- Bibliography: `refs.bib`

## Added Plot Assets (Generated from existing metrics)

- `images/q1_captioning_metrics.svg`
- `images/q2_urban_sound_metrics.svg`
- `images/q4_adversarial_robustness.svg`

## Metrics Used

### Q1 Image Captioning

- Run1: BLEU1=56.7, BLEU4=12.0, METEOR=18.5, CIDEr=21.4
- Run2: BLEU1=58.2, BLEU4=13.1, METEOR=19.0, CIDEr=23.0

### Q2 Urban Sound

- Run1: accuracy=0.62, macro_f1=0.60
- Run2: accuracy=0.71, macro_f1=0.69

### Q4 Adversarial

- eps=0.03, clean=0.80, FGSM=0.45, PGD5=0.32, PGD10=0.28

## Data Provenance

All figures were generated directly from:

- `../codes/q1_image_captioning/results/run1_metrics.csv`
- `../codes/q1_image_captioning/results/run2_metrics.csv`
- `../codes/q2_urban_sound/results/run1_metrics.csv`
- `../codes/q2_urban_sound/results/run2_metrics.csv`
- `../codes/q4_adversarial/results/run1_metrics.csv`

## Remaining Optional Improvements

- Embed the new SVG plots into `report.tex` figure sections.
- Add per-question error analysis paragraphs after each metrics table.
