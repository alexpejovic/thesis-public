RUN_DIR=../..
PY=../.venv/bin/python
CUR_DIR=final_plotting/baselines

.PHONY: all clean

all: final.pdf

final.pdf: final.svg
	inkscape final.svg -o final.pdf

final.svg: assemble.py panels/panela.svg panels/panelb.svg panels/panelc.svg
	../../$(PY) assemble.py

panels/panel%.svg: panel%.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<

clean:
	rm -rf panels
