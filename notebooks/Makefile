default: A B C D E

J=jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A:
	$(JN) 2020-11-06_NEW_HOTS_05_LogisticRegression-A.ipynb
	git commit -m' LR results : A ' 2020-11-06_NEW_HOTS_05_LogisticRegression-A.ipynb
	git push
B:
	$(JN) 2020-11-06_NEW_HOTS_05_LogisticRegression-B.ipynb
	git commit -m' LR results : B ' 2020-11-06_NEW_HOTS_05_LogisticRegression-B.ipynb
	git push
C:
	$(JN) 2020-11-06_NEW_HOTS_05_LogisticRegression-C.ipynb
	git commit -m' LR results : C ' 2020-11-06_NEW_HOTS_05_LogisticRegression-C.ipynb
	git push
D:
	$(JN) 2020-11-06_NEW_HOTS_05_LogisticRegression-D.ipynb
	git commit -m' LR results : D ' 2020-11-06_NEW_HOTS_05_LogisticRegression-D.ipynb
	git push
E:
	$(JN) 2020-11-06_NEW_HOTS_05_LogisticRegression-E.ipynb
	git commit -m' LR results : E ' 2020-11-06_NEW_HOTS_05_LogisticRegression-E.ipynb
	git push


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
	#conda install -c conda-forge opencv --override-channels
	#python3 -m pip install -r requirements.txt
	python -m ipykernel install --user --name=hots

update:
	conda env update -f environment.yml

clean:
	rm -fr /tmp/tensor*
