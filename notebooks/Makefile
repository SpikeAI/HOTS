default: A B C D E

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=hots --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A:
	$(JN) 2020-11-19_NEW_HOTS_05_LogisticRegression-A.ipynb
	git commit -m' LR results : A ' 2020-11-19_NEW_HOTS_05_LogisticRegression-A.ipynb
	git push
B:
	$(JN) 2020-11-19_NEW_HOTS_05_LogisticRegression-B.ipynb
	git commit -m' LR results : B ' 2020-11-19_NEW_HOTS_05_LogisticRegression-B.ipynb
	git push
C:
	$(JN) 2020-11-19_NEW_HOTS_05_LogisticRegression-C.ipynb
	git commit -m' LR results : C ' 2020-11-19_NEW_HOTS_05_LogisticRegression-C.ipynb
	git push
D:
	$(JN) 2020-11-19_NEW_HOTS_05_LogisticRegression-D.ipynb
	git commit -m' LR results : D ' 2020-11-19_NEW_HOTS_05_LogisticRegression-D.ipynb
	git push
E:
	$(JN) 2020-11-19_NEW_HOTS_05_LogisticRegression-E.ipynb
	git commit -m' LR results : E ' 2020-11-19_NEW_HOTS_05_LogisticRegression-E.ipynb
	git push


F:
	$(JN) HOTS_EXP_04_LR-large.ipynb
	git commit -m' LR results : HOTS_EXP_04_LR-large ' HOTS_EXP_04_LR-large.ipynb
	git push

