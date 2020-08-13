# ---
# jupyter:
#   jupytext:
#     formats: py:light,docs//ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # SWIG and Ctypes
#  
# Interfacing Fortran code through C interface

#
# I found [this reference](https://www.fortran90.org/src/best-practices.html#interfacing-with-python) with good examples.

# ## Problem: Collatz conjecture
#
#  -  Choose $u_0$ 
#  -  If $u_k$ even, $u_{k+1} \rightarrow \frac{u_k}{2}$ ;
#  -  If $u_k$ odd,  $u_{k+1}  \rightarrow 3 u_k+1$
#  
#  - The Collatz conjecture is: For all  $u_0>0$ , the process will eventually reach $u_k=1$.
#  - The programs below compute number of steps (named flight) to reach $f(u_0)$ for $1\leq u_0 \leq N$, $N$ given.
#  
# [The Collatz conjecture on Wikipedia](https://en.wikipedia.org/wiki/Collatz_conjecture)

# ## C program

# +
# %%file syracuse.c
#include <stdlib.h> 
#include <stdio.h>
long syracuse(long n) { 
   long count = 0L ; 
   while (n > 1) {
      if ((n&1)==0) 
          n /= 2; 
      else 
          n = 3*n+1; 
      count++;   
   }
   return count ; 
}

int main() {
   const long N = 1000000; 
   double t1, t2;
   long i , *flights ;
   flights = (long*)malloc(N*sizeof(long));
   for (i = 0; i <N; i++) flights[i] = syracuse(i+1); 
   return EXIT_SUCCESS;
}

# + language="bash"
# gcc -O3 syracuse.c 
# time ./a.out
# -

# ## Python program

# +
# %%time

from itertools import count

def syracuse(n):
    x = n
    for steps in count() :
        if x & 1 : 
            x = 3*x+1
        else:
            x = x // 2
            
        if x == 1:
            return steps

N = 1000000
flights = [syracuse(i) for i in range(1,N+1)]
# -

# ## Performances
#
# - The python syntax is simpler.
# - 100 times slower
# - Solution : call the C function from python.

# ## Ctypes
#
# This is the C function we will call from python

# +
# %%file syrac.c

long syracuse(long n)
{ 
   long count = 0L ; 
   while (n > 1)
   {
      if ((n&1)==0) 
         n /= 2; 
      else 
         n = 3*n+1; 
      count++;   
   }
   return count ; 
}
# -

# Build the shared library

# + language="bash"
# gcc -fPIC -shared -O3 \
#     -o syrac.so syrac.c

# +
# %%time

import time
from ctypes import *

syracDLL = CDLL("./syrac.so")
syracuse = syracDLL.syracuse

flights = [syracuse(i) for i in range(1,N+1)]
# -

# ## Ctypes with Fortran module
#
# If you change the fortran file you have to restart the kernel

# +
# %%file syrac.F90

module syrac_f90
  use iso_c_binding
  implicit none

contains

  function f_syrac(n) bind(c, name='c_syrac') result(f)
    
    integer(c_long) :: f
    integer(c_long), intent(in), value :: n
    integer(c_long) :: x
    x = n
    f = 0_8
    do while(x>1)
       if (iand(x,1_8) == 0) then
          x = x / 2
       else
          x = 3*x+1
       end if
       f = f + 1_8
    end do

  end function f_syrac

end module syrac_f90

# + language="bash"
# rm -f *.o *.so *.dylib
# gfortran -fPIC -shared -O3 -o syrac.dylib syrac.F90

# +
from ctypes import *

syrac_f90 = CDLL('./syrac.dylib')

syrac_f90.c_syrac.restype = c_long

syrac_f90.c_syrac(1000)
# -

# %%time
N = 1000000
flights = [syrac_f90.c_syrac(i) for i in range(1,N+1)]

# - Faster than pure Python
# - We can call function from DLL windows libraries.
# - Unfortunately you need to adapt the syntax to the operating system.
#
# http://docs.python.org/library/ctypes.html}

# ## SWIG

# Interface file syrac.i for C function in syrac.c

# +
# %%file syrac.i

# %module syracuseC
%{
   extern long syracuse(long n);
%}
extern long syracuse(long n);


# + language="bash"
# swig -python syrac.i
# -

# ### Build the python module 
#
# - Using command line
#
# ```bash
# swig -python syrac.i
#
# gcc `python3-config --cflags` -fPIC \
#   -shared -O3 -o _syracuseC.so syrac_wrap.c syrac.c `python3-config --ldflags`
#  ```

# - With distutils

# +
# %%file setup.py
from numpy.distutils.core import Extension, setup


module_swig = Extension('_syracuseC', sources=['syrac_wrap.c', 'syrac.c'])

setup( name='Syracuse',
       version = '0.1.0',
       author      = "Pierre Navaro",
       description = """Simple C Fortran interface example """,
       ext_modules = [module_swig],
)

# +
import sys, os

if sys.platform == "darwin":
    os.environ["CC"] = "gcc-10"
    
!{sys.executable} setup.py build_ext --inplace --quiet

# +
import _syracuseC

syracuse = _syracuseC.syracuse
syracuse(1000)

# +
# %%time
N=1000000

flights = [syracuse(i) for i in range(1,N+1)]
# -

# ## References
#
#  - [Interfacage C-Python par Xavier Juvigny](http://calcul.math.cnrs.fr/Documents/Ecoles/2010/InterfacagePython.pdf)
#  - [Optimizing and interfacing with Cython par Konrad Hinsen](http://calcul.math.cnrs.fr/Documents/Ecoles/2010/cours_cython.pdf)
#  - Python Scripting for Computational Science de Hans Petter Langtangen chez Springer
