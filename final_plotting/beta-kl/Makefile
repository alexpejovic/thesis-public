RUN_DIR=../..
PY=../.venv/bin/python
CUR_DIR=final_plotting/beta-kl

.PHONY: all clean

all: final.pdf

final.pdf: final.svg
	inkscape final.svg -o final.pdf

final.svg: assemble.py panels/panela.svg panels/panelb.svg panels/panelc.svg \
	panels/paneld.svg panels/panele.svg
	../../$(PY) assemble.py

panels/panel%.svg: panel%.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<

clean:
	rm -rf panels
