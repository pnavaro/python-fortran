# ---
# jupyter:
#   jupytext:
#     formats: py:light,docs//ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Python class linked with fortran module

# %load_ext fortranmagic

# +
# %%fortran
module particles_f90

    implicit none
    
    real(8), dimension(:), allocatable :: positions
    real(8), dimension(:), allocatable :: velocities
            
contains
    subroutine init_particles( n )
    
        integer, intent(in) :: n
                
        integer :: i
        
        if (.not. allocated(positions)) then
            allocate(positions(n))
        end if
        positions = [(i, i = 1, n, 1)]
        if (.not. allocated(velocities)) then
            allocate(velocities(n))
        end if
        velocities = 1.0

    end subroutine init_particles
 
    subroutine push_particles( dt )
        
        real(8), intent(in) :: dt
    
        positions = positions + dt * velocities
        
    end subroutine push_particles
end module particles_f90


# -

# # The Python class

class Particles(object):
    
    def __init__(self, n):
        self.index       = 0
        self.numberof    = n
        particles_f90.init_particles( n)
        self.positions  = particles_f90.positions
        self.velocities = particles_f90.velocities
        
    @property 
    def position(self):      
        return self.positions[self.index]
    
    @property 
    def velocity(self):      
        return self.velocities[self.index]


# # Access to Fortran data from Python

particles = Particles(10)
particles.velocities 

particles.positions

particles.index = 0
particles.position

particles.index = 1
particles.position


# # Create an Iterator class

class ParticleArray(object):
    
    def __init__(self, particles):
        self.particles = particles
        self.numberof = particles.numberof
        
    def __getitem__(self, index): 
        self.particles.index = index 
        return self.particles
    
    def __len__(self): 
        return self.numberof
    
    def __iter__(self): 
        for i in range(self.numberof):
            self.particles.index = i
            yield self.particles



particle_array = ParticleArray(particles)
particle_array[0].position

for p in particle_array:
    print(p.position)


# # Fortran derived type

# +
# %%fortran
module mesh

implicit none
type :: geometry
    real(8) :: xmin, xmax, dx            ! coordinates of origin and grid size
    integer :: nx                        ! number of grid points
    real(8), dimension(:), pointer :: x  ! coordinates of points
end type geometry

contains

subroutine create(geom, xmin, xmax, nx)

#     !f2py integer(8), intent(out) :: geom
    type(geometry), pointer :: geom
    real(8), intent(in) :: xmin, xmax
    integer, intent(in) :: nx
            
    real(8) :: dx
            
    integer :: i
            
    allocate(geom)
    geom%xmin = xmin
    geom%xmax = xmax
    geom%dx = ( xmax - xmin ) / (nx-1) 
    geom%nx = nx
    allocate(geom%x(nx))
    do i=1,nx
        geom%x(i)=geom%xmin+(i-1)*geom%dx
    end do

end subroutine create

subroutine view(geom)
#     !f2py integer(8), intent(in) :: geom
    type(geometry), pointer :: geom
    print*, 'nx = ', geom%nx
    print*, geom%xmin, geom%xmax
    print*, geom%x(:)
end subroutine view

subroutine get_size(geom, nx)

#     !f2py integer(8), intent(in) :: geom
    type(geometry), pointer :: geom
    integer, intent(out) :: nx
    
    nx = geom%nx
    
end subroutine get_size


end module mesh


# -

geom = mesh.create(0.0, 1.0, 10)

mesh.get_size(geom)

type(geom)

# # f2py with C code
#
# - Signature file is mandatory
# - `intent(c)` must be used for all variables and can be set globally.
# - Function name is declared with `intent(c)`

# %rm -rf cfuncts*

# +
# %%file cfuncts.c

void push_particles(double* positions, double* velocities, double dt, int n){
    for (int i=0; i<n; i++){
       positions[i] = positions[i] + dt * velocities[i];
        
    }
} 


# +
# %%file cfuncts.pyf

python module cfuncts 
    interface
        subroutine push_particles(positions, velocities, dt, n) 
            intent(c):: push_particles
            intent(c)
            integer, optional, depend(velocities) :: n = len(velocities)
            real(8), dimension(n),  intent(inplace)  :: positions 
            real(8), dimension(n),  intent(in) :: velocities
            real(8), intent(in) :: dt
        end subroutine push_particles
    end interface
end python module cfuncts
# -

# %env FC=gfortran
# %env CC=gcc-9
# !f2py -c cfuncts.c cfuncts.pyf -m cfuncts

import numpy as np
import cfuncts

print(cfuncts.push_particles.__doc__)

n = 10
dt = 0.1
x = np.arange(n, dtype="d")
v = np.ones(n, dtype="d")
cfuncts.push_particles( x, v, dt)

x

# # References
#  
# - f2py documentation https://docs.scipy.org/doc/numpy/f2py/
# - Transparents E. Sonnendrucker http://calcul.math.cnrs.fr/Documents/Journees/dec2006/python-fortran.pdf
# - Documentation Sagemath http://doc.sagemath.org/html/en/thematic_tutorials/numerical_sage/f2py.html
# - Hans Petter Langtangen : Python Scripting for Computational Science.


