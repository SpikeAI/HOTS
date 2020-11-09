default: D
A:
	jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace 2020-11-06_NEW_HOTS_05_LogisticRegression-A.ipynb
	git commit -m' LR results : A ' -a
	git push
B:
	jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace 2020-11-06_NEW_HOTS_05_LogisticRegression-B.ipynb
	git commit -m' LR results : B ' -a
	git push
C:
	jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace 2020-11-06_NEW_HOTS_05_LogisticRegression-C.ipynb
	git commit -m' LR results : C ' -a
	git push
D:
	jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace 2020-11-06_NEW_HOTS_05_LogisticRegression-D.ipynb
	git commit -m' LR results : D ' -a
	git push


PYTHON=ipython --gui=qt
PYTHON=ipython

# RUNNING

%.ipynb:
	jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace  $@

%.html:
	jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to html  $@

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
install:
	python3 -m pip install -r requirements.txt

clean:
	rm -fr /tmp/tensor*
