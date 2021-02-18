default: G

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

laurent:
	$(JN) HOTS_EXP_09_LR_PyTorch-Laurent-full.ipynb

A:
	$(JN) HOTS_EXP_05_NCARS.ipynb
	git commit -m 'EXP 05 results' HOTS_EXP_05_NCARS.ipynb
	git push
B:
	$(JN) HOTS_EXP_07_Online.ipynb
	git commit -m'EXP 07 results' HOTS_EXP_07_Online.ipynb
	git push
C:
	$(JN) HOTS_EXP_04_LR-large.ipynb
	git commit -m' EXP large results ' HOTS_EXP_04_LR-large.ipynb
	git push
D:
	$(JN) HOTS_EXP_04_LR-XXL.ipynb
	git commit -m' EXP XXL results ' HOTS_EXP_04_LR-XXL.ipynb
	git push
E:
	$(JN) HOTS_EXP_04_XXL-8clust.ipynb
	git commit -m' EXP XXL 8 clusters results ' HOTS_EXP_04_XXL-8clust.ipynb
	git push
F:
	$(JN) HOTS_EXP_04_LR-large.ipynb
	git commit -m' LR results : HOTS_EXP_04_LR-large ' HOTS_EXP_04_LR-large.ipynb
	git push
G:    
	$(JN) ../HOTS_EXP_08_spatial_jitter.ipynb
	git commit -m'results : HOTS_EXP_08_jitter ' ../HOTS_EXP_08_spatial_jitter.ipynb
	git push 
H:    
	$(JN) HOTS_EXP_02_NMNIST.ipynb
	git commit -m'results : HOTS_EXP_02 ' HOTS_EXP_02_NMNIST.ipynb
	git push
I:    
	$(JN) HOTS_EXP_09_LR_PyTorch-Laurent-full.ipynb
	git commit -m'results : HOTS_EXP_09 ' HOTS_EXP_09_LR_PyTorch-Laurent-full.ipynb
	git push
J:    
	$(JN) HOTS_EXP_09_LR_PyTorch-Laurent-max.ipynb
	git commit -m'results : HOTS_EXP_09 ' HOTS_EXP_09_LR_PyTorch-Laurent-max.ipynb
	git push
