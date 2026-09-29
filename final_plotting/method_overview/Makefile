RUN_DIR=../..
PY=../.venv/bin/python
CUR_DIR=final_plotting/method_overview

all: svg/simple_trace.svg svg/simple_reconstruction.svg svg/ideal_trace.svg \
	svg/gen_trace.svg svg/gaussian_blue.svg svg/gaussian_black.svg \
	svg/gaussian_orange.svg

svg/gaussian_blue.svg svg/gaussian_black.svg svg/gaussian_orange.svg: gaussian.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<

svg/simple_trace.svg: simple_trace.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<

svg/ideal_trace.svg: ideal_trace.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<

svg/gen_trace.svg: gen_trace.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<

svg/simple_reconstruction.svg: simple_reconstruction.py
	cd $(RUN_DIR); $(PY) $(CUR_DIR)/$<
