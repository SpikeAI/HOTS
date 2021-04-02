default: G

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A:    
	$(JN) HOTS_EXP_02_NMNIST.ipynb
	git commit -m'results : HOTS_EXP_02 ' HOTS_EXP_02_NMNIST.ipynb
	git push
B:    
	$(JN) figures.ipynb
	git commit -m'results : figures ' figures.ipynb
	git push
C:    
	$(JN) BASICS_04_Hacks.ipynb
	git commit -m'results : basics ' BASICS_04_Hacks.ipynb
	git push
D:
	$(JN) HOTS_EXP_03_Pytorch_LR.ipynb
	git commit -m 'EXP 03 results' HOTS_EXP_03_Pytorch_LR.ipynb
	git push
E:
	$(JN) HOTS_EXP_05_architectures.ipynb
	git commit -m 'EXP 05 results' HOTS_EXP_05_architectures.ipynb
	git push

F1:
	python3 homhots_full.py
	python3 homhots_nmnist_Kfold.py
F2:
	python3 homhots_ncars.py
F3:
	python3 homhots_archiz.py

G:
	$(JN) HOTS_EXP_06_PokerDVS.ipynb
	git commit -m 'EXP 06 results' HOTS_EXP_06_PokerDVS.ipynb
	git push