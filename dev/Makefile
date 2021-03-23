default: B

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

    
A:
	$(JN) 2021-02-16-HOTS_EXP_08_spatial_jitter.ipynb
	git commit -m 'EXP 08 results' 2021-02-16-HOTS_EXP_08_spatial_jitter.ipynb
	git push
B:
	$(JN) ../notebooks/HOTS_EXP_04_jitter.ipynb
	git commit -m 'EXP 04 results' ../notebooks/HOTS_EXP_04_jitter.ipynb
	git push
