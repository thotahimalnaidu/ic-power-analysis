PY := ./venv/bin/python

all:
	@t0=$$(date +%s); \
	$(PY) src/ic.py && $(PY) src/simulate.py && $(PY) src/empirical.py \
	  && $(PY) src/figures.py && $(PY) tools/make_numbers.py \
	  && echo "\n=== make all: $$(( $$(date +%s) - t0 ))s ==="

venv:
	python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

# download writes ic-data/; panel and audit read it (audit also needs osap_fundamentals.parquet)
download:
	$(PY) src/download.py

panel:
	$(PY) src/panel.py

audit:
	$(PY) src/audit.py

# tectonic if installed, else pdflatex
paper:
	@cd paper && if command -v tectonic >/dev/null; then \
	  tectonic -X compile paper.tex; \
	else \
	  pdflatex -interaction=nonstopmode paper.tex >/dev/null && bibtex paper >/dev/null \
	  && pdflatex -interaction=nonstopmode paper.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode paper.tex >/dev/null; \
	fi && echo "paper/paper.pdf"

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg

.PHONY: all venv download panel audit paper clean
