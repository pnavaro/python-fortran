# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3.7
#     language: python
#     name: python3
# ---

# %% [markdown] {"internals": {"slide_helper": "subslide_end", "slide_type": "subslide"}, "slide_helper": "slide_end", "slideshow": {"slide_type": "slide"}}
# # Julia Set
#
# Modified version from [Loic Gouarin](https://github.com/gouarin/GTSage2014/)
#
# [Julia set on wikipedia](https://en.wikipedia.org/wiki/Julia_set)

# %% {"internals": {"slide_helper": "subslide_end", "slide_type": "subslide"}, "slide_helper": "slide_end", "slideshow": {"slide_type": "skip"}}
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# %% {"internals": {"slide_helper": "subslide_end", "slide_type": "subslide"}, "slide_helper": "slide_end", "slideshow": {"slide_type": "skip"}}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams['figure.figsize'] = (6,6)

# %% {"internals": {"slide_helper": "subslide_end"}, "slide_helper": "slide_end", "slideshow": {"slide_type": "-"}}
nx, ny = 512, 512 # mesh
lim, maxit = 400, 2000 # limits 
vmin, vmax = 0, 200 

x = np.linspace(-1.6, 1.6, nx)
y = np.linspace(-1.6, 1.6, ny)
c = -0.772691322542185 + 0.124281466072787j


# %% [markdown] {"internals": {"slide_type": "subslide"}, "slideshow": {"slide_type": "slide"}}
# # Pure Python

# %% {"internals": {}, "slideshow": {"slide_type": "-"}}
def juliaset_python(x, y, c, lim, maxit):
    """ 
    returns Julia set
    """
    julia = np.zeros((x.size, y.size))

    for i in range(x.size):
        for j in range(y.size):
            z = x[i] + 1j*y[j]
            ite = 0
            while abs(z) < lim and ite < maxit:
                z = z**2 + c
                ite += 1
            julia[j, i] = ite

    return julia


# %% {"internals": {}, "slideshow": {"slide_type": "-"}}
def plot_julia_set(julia):
    plt.figure(figsize=(6,6))
    plt.imshow(julia, cmap = cm.Greys, vmin=vmin, vmax=vmax)


# %% {"internals": {"slide_helper": "subslide_end"}, "slide_helper": "slide_end", "slideshow": {"slide_type": "-"}}
plot_julia_set(juliaset_python(x, y, c, lim, maxit))

# %% [markdown]
# # Fortran

# %%
%load_ext fortranmagic


# %%
%env FC=gfortran
%env CC=gcc-9

# %%
%%fortran --f90flags "-fopenmp" --opt="-O3" --extra "-L/usr/local/lib -lgomp"
subroutine juliaset_fortran(x, y, c, lim, maxit, julia)

    real(8),    intent(in)  :: x(:)
    real(8),    intent(in)  :: y(:)
    complex(8), intent(in)  :: c
    real(8),    intent(in)  :: lim
    integer,    intent(in)  :: maxit
    integer,    intent(out) :: julia(size(x),size(y))

    real(8)    :: zr, zi, limsq, cr, ci, tmp
    integer    :: ite, nx, ny

    nx = size(x)
    ny = size(y)
    limsq = lim * lim
    cr = real(c)
    ci = imag(c)

    !$OMP PARALLEL DEFAULT(NONE) &
    !$OMP FIRSTPRIVATE(nx,ny,x,y,c,limsq,maxit,cr,ci) &
    !$OMP PRIVATE(i,j,ite,zr,zi, tmp) &
    !$OMP SHARED(julia)
    !$OMP DO SCHEDULE(DYNAMIC)
    do i = 1, nx
       do j = 1, ny   
            zr = x(i)
            zi = y(j)
            ite = 0
            do while (zr*zr+zi*zi < limsq .and. ite < maxit)
                tmp = zr*zr - zi*zi 
                zi = 2*zr*zi + ci
                zr = tmp + cr
                ite = ite + 1
            end do
            julia(j, i) = ite
        end do
    end do
    
    !$OMP END PARALLEL


end subroutine juliaset_fortran

# %%
plot_julia_set(juliaset_fortran(x, y, c, lim, maxit))

# %% [markdown] {"internals": {"slide_type": "subslide"}, "slideshow": {"slide_type": "slide"}}
# # Numpy

# %% {"internals": {"slide_helper": "subslide_end"}, "slide_helper": "subslide_end", "slideshow": {"slide_type": "-"}}
import itertools

def juliaset_numpy(x, y, c, lim, maxit):
    julia = np.zeros((x.size, y.size), dtype=np.int32)

    zx = x[np.newaxis, :]
    zy = y[:, np.newaxis]
    
    z = zx + zy*1j
    
    for ite in itertools.count():
        
        z = z**2 + c 
        mask = np.logical_not(julia) & (np.abs(z) >= lim)
        julia[mask] = ite
        if np.all(julia) or ite > maxit:
            return julia
            

    

# %%
plot_julia_set(juliaset_numpy(x, y, c, lim, maxit))

# %% [markdown] {"internals": {"slide_type": "subslide"}, "slideshow": {"slide_type": "slide"}}
# # Cython

# %% {"internals": {}, "slideshow": {"slide_type": "-"}}
import os, sys

if sys.platform == 'darwin':
    os.environ['CC'] = 'gcc-9'
    os.environ['CXX'] = 'g++-9'
else:
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'


# %% {"internals": {}, "slideshow": {"slide_type": "-"}}
%load_ext cython

# %% {"internals": {"slide_helper": "subslide_end", "slide_type": "subslide"}, "slide_helper": "subslide_end", "slideshow": {"slide_type": "subslide"}}
%%cython
import numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def juliaset_cython(double [:] x, double [:] y, double complex c, double lim, int maxit):
    cdef:
        int [:, ::1] julia = np.zeros((x.size, y.size), dtype = np.int32)
        double tmp, zr, zi, lim2 = lim*lim
        double cr = c.real, ci = c.imag
        int ite, i, j, nx=x.size, ny=y.size

    for i in range(nx):
        for j in range(ny):
            zr = x[i] 
            zi = y[j]
            ite = 0
            while (zr*zr + zi*zi) < lim2 and ite < maxit:
                zr, zi = zr*zr - zi*zi + cr, 2*zr*zi + ci
                ite += 1
            julia[j, i] = ite

    return julia


# %%
plot_julia_set(juliaset_cython(x, y, c, lim, maxit))

# %% {"internals": {"slide_helper": "subslide_end"}, "slide_helper": "subslide_end", "slideshow": {"slide_type": "-"}}
%%cython --v -f -c-fopenmp --link-args=-fopenmp
import numpy as np
import cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free 

@cython.boundscheck(False)
@cython.wraparound(False)
def juliaset_cython_omp(double [:] x, double [:] y, double complex c, double lim, int maxit):
    cdef:
        int [:, ::1] julia = np.zeros((x.size, y.size), dtype = np.int32)
        double tmp, zr, zi, lim2 = lim*lim
        double cr = c.real, ci = c.imag
        int  i, j, nx=x.size, ny=y.size
        int *ite

    for j in prange(ny, nogil=True, schedule='dynamic'):
        ite = <int *> malloc(sizeof(int))
        for i in range(nx):
            zr = x[i] 
            zi = y[j]
            ite[0] = 0
            while (zr*zr + zi*zi) < lim2 and ite[0] < maxit:
                zr, zi = zr*zr - zi*zi + cr, 2*zr*zi + ci
                ite[0] += 1
            julia[j, i] = ite[0]
        free(ite)
        
    return julia


# %%
plot_julia_set(juliaset_cython_omp(x, y, c, lim, maxit))

# %% [markdown] {"internals": {"slide_type": "subslide"}, "slideshow": {"slide_type": "slide"}}
# # numba

# %% {"internals": {"slide_helper": "subslide_end"}, "slide_helper": "subslide_end", "slideshow": {"slide_type": "slide"}}
from numba import autojit

@autojit
def juliaset_numba(x, y, c, lim, maxit):
    julia = np.zeros((x.size, y.size))
    lim2 = lim*lim
    
    c = complex(c)  # needed for numba
    for j in range(y.size):
        for i in range(x.size):

            z = complex(x[i], y[j])
            ite = 0
            while (z.real*z.real + z.imag*z.imag) < lim2 and ite < maxit:
                z = z*z + c
                ite += 1
            julia[j, i] = ite

    return julia


# %% {"slideshow": {"slide_type": "slide"}}
plot_julia_set(juliaset_numba(x, y, c, lim, maxit))

# %% {"slideshow": {"slide_type": "slide"}}
%reload_ext pythran.magic

# %% {"slideshow": {"slide_type": "fragment"}}
%%pythran

import numpy as np

#pythran export juliaset_pythran(float64[], float64[],complex, int, int)
def juliaset_pythran(x, y, c, lim, maxit):
    """ 
    returns Julia set
    """
    julia = np.zeros((x.size, y.size), dtype=np.int32)

    for j in range(y.size):
        for i in range(x.size):
            z = x[i] + 1j*y[j]
            ite = 0
            while abs(z) < lim and ite < maxit:
                z = z**2 + c
                ite += 1
            julia[j, i] = ite

    return julia


# %%
plot_julia_set(juliaset_pythran(x, y, c, lim, maxit))

# %% {"internals": {"slide_type": "subslide"}, "slideshow": {"slide_type": "subslide"}}
import pandas as pd
from collections import defaultdict
results = defaultdict(list)

functions = [juliaset_python,
             juliaset_fortran,
             juliaset_numpy,
             juliaset_cython,
             juliaset_cython_omp,
             juliaset_numba,
             juliaset_pythran]

for f in functions:

    _ = %timeit -oq -n 1 f(x, y, c, lim, maxit)
    results['etime'] += [_.best]

# %% {"slideshow": {"slide_type": "slide"}}
results = pd.DataFrame(results, index=list(map(lambda f:f.__name__[9:],functions)))


# %% {"slideshow": {"slide_type": "fragment"}}
results["speed_up"] = [results.etime[0]/t for t in results.etime]

# %% {"slideshow": {"slide_type": "fragment"}}
results.sort_values(by="speed_up",axis=0)
