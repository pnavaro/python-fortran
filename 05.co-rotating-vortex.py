# ---
# jupyter:
#   jupytext:
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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.figsize'] = (10,6)

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# %env FC=/opt/miniconda3/envs/python-fortran/bin/gfortran


# %load_ext fortranmagic

# +
# %%fortran
subroutine biot ( n, xp, yp, op, up, vp )
  implicit none
  integer, intent(in)  :: n
  real(8), intent(in)  :: xp( n ), yp( n ), op( n )
  real(8), intent(out) :: up( n ), vp( n )
  !f2py optional , depend(a) :: n=len(xp)
        
  integer :: k, j
  real(8) :: dpi, a1, a12, a122, r2, r22, r2a1, r2a13
  real(8) :: f_ik, usum, vsum, dx, dy

  dpi   = 8d0 * atan( 1d0 )

  a1    = .05
  a12   = a1*a1
  a122  = a12*a12
    

  do  k = 1, n

    usum = 0
    vsum = 0
    do j = 1, n
        dx    = xp( j ) - xp( k )
        dy    = yp( j ) - yp( k )
        r2    = dx * dx + dy * dy
        if ( r2 > 1e-12 ) then
            r22   = r2 * r2
            r2a1  = r2 + a12
            r2a13 = r2a1 * r2a1 * r2a1
            f_ik  = ( r22 + 3d0*a12*r2 + 4d0*a122 ) / r2a13 
            usum = usum + dy * op(j) * f_ik
            vsum = vsum - dx * op(j) * f_ik 
        end if
    end do

    up(k) = usum / dpi 
    vp(k) = vsum / dpi 
            
  end do

end subroutine biot

# -

def distrib(ray, nray = 10,nsec = 6, gauss=True, gam0 = 1.0, eps= 0):
    
    """
        Particles distributed evenly spaced into a disc.
         rf,zf : particles positions
         ds    : particle surface size 
         cir   : particle circulation
         dr    : size step along radial axis
         nray  : nb of points along adial axis
         dray  : particle ray in the center
         ray   : disc ray
         gam0  : total strength of the vortex 
         surf  : disc surface
         nsec  : number of particles around the center particle
         eps   : ellipsis parameter
         
    """

    pi = np.pi
    
    dr = ray / (nray + 0.5)
    dray = 0.5 * dr  
    surf = pi * ray * ray
    dtheta = 2.0 * pi / nsec

    k = 0
    rf = [0.0]
    zf = [0.0]
    ds = [pi * dray * dray]

    if gauss:
        gamt = gam0 / (1. - np.exp(-1.0))
        cf = [gamt * (1. - np.exp(-(dray / ray) ** 2))]  # gaussian
    else:
        cf = [gam0 * ds[k] / surf]  # uniform

    r1 = dray
    s1 = pi * r1 ** 2
    nsec0 = nsec
    nsec = 0

    for i in range(nray):

        nsec = nsec + nsec0
        dtheta = 2.0 * pi / nsec
        r = (i + 1) * dr

        r2  = r + 0.5 * dr
        s2  = pi * r2 ** 2
        dss = s2 - s1
        s1  = s2

        for j in range(nsec):

            k = k + 1
            theta = (j + 1) * dtheta
            sigma = r * (1.0 + eps * np.cos(2.0 * theta))
            rf.append(sigma * np.cos(theta))
            zf.append(sigma * np.sin(theta))
            ds.append(dss / nsec)
            if gauss:  # gaussian
                q = 0.5 * (np.exp(-(r1 / ray) ** 2) - np.exp(-(r2 / ray) ** 2))
                strength = gamt * dtheta / pi * q
            else:      # uniform
                strength = gam0 * ds[k] / surf
            
            cf.append(strength)
                
        r1 = r2
        kd = k - nsec + 1

    nr = k
    print (len(rf),len(zf),len(cf))

    ssurf = np.sum(ds)
    sgam  = np.sum(cf)

    print('toal number of particles :', nr)
    print('check surface :', (surf), ' - ', (ssurf))
    if (gauss):
        print('check vortex strength :', (gam0), ' ; ', (sgam))
    else:
        print('check vortex strength :', (gam0), ' ; ', (sgam))
        
    

    return np.array(rf), np.array(zf), np.array(ds), np.array(cf)


import seaborn as sns
sns.set()
r, z, s, g = distrib(1.0,nray = 40,nsec = 10)
plt.scatter(r, z, 1e3*s, g)
plt.axis('scaled')
plt.grid(True)
plt.colorbar();

# +
nstep = 1
amach = 0.1
nproc = 1
dt    = 0.1

pi = np.pi
r0 = 0.6
u0 = amach

#gam0   = u0 * 2.0 * pi / 0.7 * r0	!gaussienne
#gam0   = 2. * pi * r0 * u0		!constant
gam0   = 2. * pi / 10.0

aom    = gam0 / ( pi * r0**2 )	#Amplitude du vortex
tau    = 8.0 * pi**2 / gam0	#Periode de co-rotation
gomeg  = gam0/ (4.0*pi)		#Vitesse angulaire

print( " --------------------------------------------- ")
print( " steps : ", nstep)
print( " time step : ", dt)
print( " aom = ", aom)
print( " r0 = ", r0)
print( " strength = ", gam0)
print( " rotation speed gomeg = ", gomeg)
print( " corotation period = ", tau)
print( " --------------------------------------------- ")

rf, zf , ds, gam =  distrib( r0 )
n = 2 * rf.size
X = np.zeros((n,2),dtype=np.float64)
X[:,0] = np.concatenate([rf, rf])
X[:,1] = np.concatenate([zf + 1., zf - 1])
op = np.concatenate([gam,gam])
# -

n = X.shape[0]
plt.scatter(X[:,0], X[:,1], 10*np.ones(n),  op)
plt.axis('scaled')
plt.grid(True)
plt.colorbar();

# +
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2));

# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=2);

def init():
    particles.set_data([], [])
    return particles,

def animate(i):
    global dt, ax, fig, X
    up, vp = biot(X[:,0], X[:,1], op)
    up, vp = biot(X[:,0] + 0.5 * dt * up, 
                  X[:,1] + 0.5 * dt * vp, op)
    X[:,0] += dt * up
    X[:,1] += dt * vp

    particles.set_data(X[:, 0], X[:, 1])
    return particles,

ani = animation.FuncAnimation(fig, animate, frames=400,
                              interval=10, blit=True, init_func=init);


# +
from IPython.display import HTML

HTML(ani.to_jshtml())
# -

n = xp.size
plt.scatter(X[:,0], X[:,1], 10*np.ones(n),  op)
plt.axis('scaled')
plt.grid(True)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.colorbar();

rand = np.random.RandomState(42)
X = rand.rand(5, 2)
plt.scatter(X[:, 0], X[:, 1], s=100);


def velocities (X, op, delta = 1e-6):

    dpi   = 2 * np.pi
    n = X.shape[0]

    dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    rsq = np.sum(dX**2, axis=-1)

    rsq += rsq < delta
     
    V[:,0] =  np.sum(dX[:,1] / rsq,axis=1) * op / dpi 
    V[:,1] =  np.sum(-dX[:,0] / rsq,axis=1) * op / dpi 
    return V


dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
dX.shape

#square the coordinate differences
sq_dX = dX ** 2
sq_dX.shape

rsq = np.sum(dX**2, axis=-1)
rsq


np.cross(dX,op)

# +
"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ParticleBox:
    """Orbits class
    
    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-2, 2, -2, 2],
                 size = 0.04,
                 M = 0.05,
                 G = 9.8):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2) 

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1

        # add gravity
        self.state[:, 3] -= self.M * self.G * dt


#------------------------------------------------------------
# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((50, 4))
init_state[:, :2] *= 3.9

box = ParticleBox(init_state, size=0.04)
dt = 1. / 30 # 30fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=6)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


# +
from IPython.display import HTML

HTML(ani.to_jshtml())
# -


