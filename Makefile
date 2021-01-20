default: notebooks

PYTHON=ipython --gui=qt
PYTHON=ipython

# RUNNING
notebooks:
	cd notebooks; make
	
%.ipynb:
	$(JN) $@

%.html:
	$(J) --to html  $@

weigth:
	du -h . --max-depth=1

count_file:
	ls -Al | wc -l

test_version:
	python3 -V
	python3 -c 'import torch; print(torch.__version__)'

# CODING
pep8:
	autopep8 *.py -r -i --max-line-length 120 --ignore E402

## INSTALL

clean:
	rm -fr /tmp/tensor*
