default: C

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
C:
	python3 2021-04-30-homhots_nmnist_jitter_0.py
C1:
	python3 2021-04-30-homhots_nmnist_jitter_1.py
C2:
	python3 2021-04-30-homhots_nmnist_jitter_2.py
C3:
	python3 2021-04-30-homhots_nmnist_jitter_3.py
C4:
	python3 2021-04-30-homhots_nmnist_jitter_4.py
C5:
	python3 2021-04-30-homhots_nmnist_jitter_5.py
C6:
	python3 2021-04-30-homhots_nmnist_jitter_6.py
C7:
	python3 2021-04-30-homhots_nmnist_jitter_7.py
C8:
	python3 2021-04-30-homhots_nmnist_jitter_8.py
C9:
	python3 2021-04-30-homhots_nmnist_jitter_9.py
D:
	python3 2021-05-22-homhots_LR_jitter.py
D1:
	python3 2021-05-22-homhots_LR_jitter_runonly.py
D2:
	python3 2021-05-22-homhots_LR_jitter_runonly2.py
