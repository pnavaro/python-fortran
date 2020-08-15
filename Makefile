all: 01.f2py.ipynb 02.f2py.ipynb 03.julia-set.ipynb 04.vlasov-poisson.ipynb 05.co-rotating-vortex.ipynb 06.gray-scott-model.ipynb 07.maxwell-fdtd-2d.ipynb 08.swig.ipynb

.SUFFIXES: .py .ipynb

.py.ipynb:
	jupytext --set-formats py,notebooks//ipynb $<

01.f2py.ipynb: 01.f2py.py
02.f2py.ipynb: 02.f2py.py
03.julia-set.ipynb: 03.julia-set.py
04.vlasov-poisson.ipynb: 04.vlasov-poisson.py
05.co-rotating-vortex.ipynb: 05.co-rotating-vortex.py
06.gray-scott-model.ipynb: 06.gray-scott-model.py
07.maxwell-fdtd-2d.ipynb: 07.maxwell-fdtd-2d.py
08.swig.ipynb: 08.swig.py
