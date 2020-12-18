default: update

PYTHON=ipython --gui=qt
PYTHON=ipython

# RUNNING

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
activate:
	conda activate hots

install:
	conda env create -f environment.yml
	#python3 -m pip install -r requirements.txt
	python -m ipykernel install --user --name=hots

update:
	conda env update -f environment.yml

clean:
	rm -fr /tmp/tensor*
