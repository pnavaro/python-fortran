# Python-Fortran bindings examples

[![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pnavaro/python-fortran/master)
[![jupyterbook](https://github.com/pnavaro/python-fortran/workflows/jupyter-book/badge.svg)](https://pnavaro.github.io/python-fortran)

```bash
git clone https://github.com/pnavaro/python-fortran.git
cd python-fortran
conda env create -f environment.yml
conda activate python-fortran
jupyter notebook
```

You can open py files or ipynb files in notebooks directory. They are synchronized with
jupytext.


  - [Introduction to F2PY](https://pnavaro.github.io/python-fortran/01.f2py.html)
  - [F2PY and Fortran 90 modules](https://pnavaro.github.io/python-fortran/02.f2py.html)
  - [Comparison with Cython, Numba, Pythran and PyJulia : The Julia set example](https://pnavaro.github.io/python-fortran/03.julia-set.html)
  - [The 1D1V Vlasov-Poisson solved with Semi-Lagrangian method](https://pnavaro.github.io/python-fortran/04.vlasov-poisson.html)
  - [Co-rotating vortices solved with discrete vortex method](https://pnavaro.github.io/python-fortran/05.co-rotating-vortex.html)
  - [Finite differences and the Gray-Scott model](https://pnavaro.github.io/python-fortran/06.gray-scott-model.html)
  - [Finite differences on staggered grid to solve Maxwell equations in 2D](https://pnavaro.github.io/python-fortran/07.maxwell-fdtd-2d.html)
  - [Short example with SWIG and CTYPES](https://pnavaro.github.io/python-fortran/08.swig.html)

Reminder: Build the Jupyter book

```bash
make
jupyter-book build notebooks
```
and eventually push the book to gh-pages
```bash
ghp-import -n -p -f notebooks/_build/html
```

Pierre Navaro IRMAR CNRS
