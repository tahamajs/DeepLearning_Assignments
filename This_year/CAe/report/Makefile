TEX=pdflatex
BIB=bibtex

all: report.pdf

report.pdf: report.tex refs.bib
	$(TEX) -interaction=nonstopmode report.tex
	$(BIB) report
	$(TEX) -interaction=nonstopmode report.tex
	$(TEX) -interaction=nonstopmode report.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out report.pdf
