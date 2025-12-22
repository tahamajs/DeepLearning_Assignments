# Makefile for building CA1 Neural Networks Basics Report

.PHONY: all clean view help

# Default target
all: report.pdf

# Build the PDF
report.pdf: report.tex refs.bib
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls

# Clean all generated files including PDF
cleanall: clean
	rm -f report.pdf

# View the PDF (macOS)
view: report.pdf
	open report.pdf

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the PDF report (default)"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove auxiliary files and PDF"
	@echo "  view     - Open the PDF report (macOS)"
	@echo "  help     - Show this help message"

# Rebuild from scratch
rebuild: cleanall all