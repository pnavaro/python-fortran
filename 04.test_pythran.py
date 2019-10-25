# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3.7
#     language: python
#     name: python3
# ---

# # Julia Set
#
#
# In the notebook I present different way to accelerate python code. 
# This is a modified version from [Loic Gouarin](https://github.com/gouarin/GTSage2014/)
#
# The test case is the computation of the Julia set [wikipedia](https://en.wikipedia.org/wiki/Julia_set)
#
# # Pythran
#
# [Pythran](https://pythran.readthedocs.io/en/latest/) is a Python-to-C++ translator
#
# Add a comment line before your python function and it runs much faster.
#
# ### Configuration
#
# `~/.pythranrc` file on my mac (gcc is installed with hombrew and pythran with pip)
#
# ```
# [compiler]
# include_dirs=/usr/local/opt/openblas/include
# library_dirs=/usr/local/opt/openblas/lib
# blas=openblas
# CXX=g++-9
# CC=gcc-9
# ```

# + {"internals": {"slide_helper": "subslide_end", "slide_type": "subslide"}, "slide_helper": "slide_end", "slideshow": {"slide_type": "skip"}}
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#import matplotlib.pyplot as plt
# -

%load_ext pythran.magic

# + {"slideshow": {"slide_type": "fragment"}}
%%pythran

import numpy as np

#pythran export test_pythran(float64[], float64[])
def test_pythran(x, y):
    """ 
    returns Julia set
    """
    z = np.zeros((x.size, y.size), dtype=np.float64)

    #omp parallel for private(z)
    for j in range(y.size):
        for i in range(x.size):
            z[j,i] = x[i] + y[j]

    return z
# -


