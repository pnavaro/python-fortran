[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pnavaro/python-fortran/master)

```bash
git clone https://github.com/pnavaro/python-fortran.git
cd python-fortran
conda env create environment.yml
conda activate python-fortran
jupyter notebook
```

## Call Fortran from Python with f2py
  - [Introduction to F2PY](01.f2py.html)
  - [F2PY and Fortran 90 modules](02.f2py.html)

## Examples
  - [Comparison with Cython, Numba, Pythran and PyJulia : The Julia set example](03.julia-set.html)
  - [The 1D1V Vlasov-Poisson solved with Semi-Lagrangian method](04.vlasov-poisson.html)
  - [Co-rotating vortices solved with discrete vortex method](05.co-rotating-vortex.html)
  - [Finite differences and the Gray-Scott model](06.gray-scott-model.html)
  - [Finite differences on staggered grid to solve Maxwell equations in 2D](07.maxwell-fdtd-2d.html)

## Other way to call Fortran
  - [Short example with SWIG and CTYPES](08.swig.html)
  - [How to: call Fortran from Python](http://www.legi.grenoble-inp.fr/people/Pierre.Augier/how-to-call-fortran-from-python.html)
