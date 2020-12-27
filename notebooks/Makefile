default: A B C D E

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A:
	$(JN) HOTS_EXP_03_Network_Architectures.ipynb
	git commit -m 'EXP 03 results' HOTS_EXP_03_Network_Architectures.ipynb
	git push
B:
	$(JN) HOTS_EXP_04_LR.ipynb
	git commit -m'EXP 04 results' HOTS_EXP_04_LR.ipynb
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
