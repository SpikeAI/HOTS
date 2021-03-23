default: A

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A:    
	$(JN) ../HOTS_EXP_03_Pytorch_LR-XL.ipynb
	git commit -m'results : HOTS_EXP_03 ' ../HOTS_EXP_03_Pytorch_LR-XL.ipynb
	git push
