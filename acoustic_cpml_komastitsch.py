#! /usr/bin/env python3
# Desenvolvimento do simulador de onda acústica com CPML
# Modificação do código de Komastitsch fortran 90
# Giovanni Alfredo Guarneri


from PyQt6.QtWidgets import *
import numpy as np
from scipy import signal
from time import perf_counter
import math
import matplotlib.pyplot as plt
from matplotlib import use
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from PyQt6.QtGui import *
from PyQt6.QtCore import *


# use('TkAgg')

# Image View class
class ImageView(pg.ImageView):
    # constructor which inherit original
    # ImageView
    def __init__(self, *args, **kwargs):
        pg.ImageView.__init__(self, *args, **kwargs)


# Window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle(f"{ny}x{nx} Grid x {nstep} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")

        # setting geometry
        # self.setGeometry(200, 50, 1600, 800)
        self.setGeometry(200, 50, 500, 500)

        # setting animation
        self.isAnimated()

        # setting image
        self.image = np.random.normal(size=(500, 500))

        # showing all the widgets
        self.show()

        # creating a widget object
        self.widget = QWidget()

        # setting configuration options
        pg.setConfigOptions(antialias=True)

        # creating image view view object
        self.imv = RawImageWidget()

        # setting image to image view
        self.imv.setImage(self.image, levels=[-0.1, 0.1])

        # Creating a grid layout
        self.layout = QGridLayout()

        # setting this layout to the widget
        self.widget.setLayout(self.layout)

        # plot window goes on right side, spanning 3 rows
        self.layout.addWidget(self.imv, 0, 0, 4, 1)

        # setting this widget as central widget of the main window
        self.setCentralWidget(self.widget)


## Simulation Parameters
# number of points
nx = 801  # colunas
ny = 801  # linhas

# size of grid cell
dx = 1.5  # [m? km?]
dy = dx

# Thickness of the PML layer in grid points
npoints_pml = 10

# P-velocity and density
cp_unrelaxed = 2000.0  # [m/s] ??
density = 2000.0  # [kg / m ** 3] ??

# Total number of time steps
nstep = 1500

# Time step in seconds
dt = 5.2e-4  # [s]

# Parameters for the source
f0 = 35.0  # frequency? [Hz]
t0 = 1.20 / f0  # delay? [?]
factor = 1.0

# Source (in pressure)
xsource = 600.0
ysource = 600.0
isource = int(xsource / dx) + 1
jsource = int(ysource / dy) + 1

# Receivers
nrec = 1
xdeb = 561.0  # First receiver x in meters
ydeb = 561.0  # First receiver y in meters
xfin = 561.0  # Last receiver x in meters
yfin = 561.0  # Last receiver y in meters

# Large value for maximum
HUGEVAL = 1.0e30

# Threshold above which we consider that the code became unstable
STABILITY_THRESHOLD = 1.0e25

# Main arrays
p_2 = np.zeros((ny, nx))  # Pressão passada
p_1 = np.zeros((ny, nx))  # Pressão presente
p_0 = np.zeros((ny, nx))  # Pressão futura
dp_x = np.zeros((ny, nx))
dp_y = np.zeros((ny, nx))
dp = np.zeros((ny, nx))  # Derivada primeira da pressão
v_x = np.zeros((ny, nx))
v_y = np.zeros((ny, nx))
v = np.zeros((ny, nx))  # Velocidade
kappa_unrelaxed = np.zeros((ny, nx))
rho = np.zeros((ny, nx))
Kronecker_source = np.zeros((ny, nx))

# To interpolate material parameters or velocity at the right location in the staggered grid cell
rho_half_x = np.zeros((ny, nx))
rho_half_y = np.zeros((ny, nx))

# Power to compute d0 profile
NPOWER = 2.0

# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = 1.0
ALPHA_MAX_PML = 2.0 * math.pi * (f0 / 2)  # from Festa and Vilotte

# auxiliar
mdp_x = np.zeros((ny, nx))
mdp_y = np.zeros((ny, nx))
mdp = np.zeros((ny, nx))
dmdp_x = np.zeros((ny, nx))
dmdp_y = np.zeros((ny, nx))
dmdp = np.zeros((ny, nx))

vdp_x = np.zeros((ny, nx))
vdp_y = np.zeros((ny, nx))
vdp_xx = np.zeros((ny, nx))
vdp_yy = np.zeros((ny, nx))
vdp_new = np.zeros((ny, nx))

# Arrays for the damping profiles
d_x = np.zeros(nx)
d_x_half = np.zeros(nx)
K_x = np.zeros(nx)
K_x_half = np.zeros(nx)
alpha_x = np.zeros(nx)
alpha_x_half = np.zeros(nx)
a_x = np.zeros(nx)
a_x_half = np.zeros(nx)
b_x = np.zeros(nx)
b_x_half = np.zeros(nx)

d_y = np.zeros(ny)
d_y_half = np.zeros(ny)
K_y = np.zeros(ny)
K_y_half = np.zeros(ny)
alpha_y = np.zeros(ny)
alpha_y_half = np.zeros(ny)
a_y = np.zeros(ny)
a_y_half = np.zeros(ny)
b_y = np.zeros(ny)
b_y_half = np.zeros(ny)

thickness_PML_x = 0.0
thickness_PML_y = 0.0
xoriginleft = 0.0
xoriginright = 0.0
xoriginbottom = 0.0
xorigintop = 0.0
Rcoef = 0.0
d0_x = 0.0
d0_y = 0.0
xval = 0.0
yval = 0.0

# for source
a = 0.0
t = 0.0
source_term = 0.0

# for receivers
xspacerec = 0.0
yspacerec = 0.0
distval = 0.0
dist = 0.0
ix_rec = np.zeros(nrec, dtype='int')
iy_rec = np.zeros(nrec, dtype='int')
xrec = np.zeros(nrec)
yrec = np.zeros(nrec)
myNREC = nrec

i = 0
j = 0
it = 0
irec = 0

Courant_number = 0.0
pressurenorm = 0.0

# program starts here
print("2D acoustic finite-difference code in pressure formulation with C-PML\n")
print(f"NX = {nx}\nNY = {ny}\n\n")
print(f"size of the model along X = {(nx - 1) * dx}")
print(f"size of the model along Y = {(ny - 1) * dy}\n")
print(f"Total number of grid points = {nx * ny}\n")

# define profile of absorption in PML region
# thickness of PML layers in meters
thickness_PML_x = npoints_pml * dx
thickness_PML_y = npoints_pml * dy

# reflection coefficient (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
Rcoef = 0.001

# check that power is okay
if NPOWER < 1:
    raise ValueError('NPOWER must be greater than 1')

# compute d0 from INRIA report section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
d0_x = -(NPOWER + 1) * cp_unrelaxed * math.log(Rcoef) / (2.0 * thickness_PML_x)
d0_y = -(NPOWER + 1) * cp_unrelaxed * math.log(Rcoef) / (2.0 * thickness_PML_y)

## damping in the X direction
# origin of the PML layer (position of right edge minus thickness, in meters)
xoriginleft = thickness_PML_x
xoriginright = (nx - 1) * dx - thickness_PML_x

# dampening profile in X direction at the grid points
i = np.arange(nx)
xval = dx * i
abscissa_in_PML_left = xoriginleft - xval
abscissa_in_PML_right = xval - xoriginright
abscissa_in_PML_mask_left = np.where(abscissa_in_PML_left < 0.0, False, True)
abscissa_in_PML_mask_right = np.where(abscissa_in_PML_right < 0.0, False, True)
absc_mask_x = np.logical_or(abscissa_in_PML_mask_left, abscissa_in_PML_mask_right)
abscissa_in_PML = np.zeros(nx)
abscissa_in_PML[abscissa_in_PML_mask_left] = abscissa_in_PML_left[abscissa_in_PML_mask_left]
abscissa_in_PML[abscissa_in_PML_mask_right] = abscissa_in_PML_right[abscissa_in_PML_mask_right]
abscissa_normalized = abscissa_in_PML / thickness_PML_x
d_x = d0_x * abscissa_normalized ** NPOWER
K_x = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_x = ALPHA_MAX_PML * (1.0 - np.where(absc_mask_x, abscissa_normalized, 1.0))

# dampening profile in X direction at half the grid points
abscissa_in_PML_left = xoriginleft - (xval + dx / 2.0)
abscissa_in_PML_right = (xval + dx / 2.0) - xoriginright
abscissa_in_PML_mask_left = np.where(abscissa_in_PML_left < 0.0, False, True)
abscissa_in_PML_mask_right = np.where(abscissa_in_PML_right < 0.0, False, True)
absc_mask_x_half = np.logical_or(abscissa_in_PML_mask_left, abscissa_in_PML_mask_right)
abscissa_in_PML = np.zeros(nx)
abscissa_in_PML[abscissa_in_PML_mask_left] = abscissa_in_PML_left[abscissa_in_PML_mask_left]
abscissa_in_PML[abscissa_in_PML_mask_right] = abscissa_in_PML_right[abscissa_in_PML_mask_right]
abscissa_normalized = abscissa_in_PML / thickness_PML_x
d_x_half = d0_x * abscissa_normalized ** NPOWER
K_x_half = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_x_half = ALPHA_MAX_PML * (1.0 - np.where(absc_mask_x_half, abscissa_normalized, 1.0))
b_x = np.exp(- (d_x / K_x + alpha_x) * dt)
b_x_half = np.exp(- (d_x_half / K_x_half + alpha_x_half) * dt)

# to avoid division by zero outside tha PML
i = np.where(d_x > 1e-6)
a_x[i] = d_x[i] * (b_x[i] - 1.0) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i]))
i = np.where(d_x_half > 1e-6)
a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i]))

## damping in the Y direction
# origin of the PML layer (position of right edge minus thickness, in meters)
yoriginbottom = thickness_PML_y
yorigintop = (ny - 1) * dy - thickness_PML_y

# dampening profile in Y direction at the grid points
j = np.arange(ny)
yval = dy * j
abscissa_in_PML_bottom = yoriginbottom - yval
abscissa_in_PML_top = yval - yorigintop
abscissa_in_PML_mask_bottom = np.where(abscissa_in_PML_bottom < 0.0, False, True)
abscissa_in_PML_mask_top = np.where(abscissa_in_PML_top < 0.0, False, True)
absc_mask_y = np.logical_or(abscissa_in_PML_mask_bottom, abscissa_in_PML_mask_top)
abscissa_in_PML = np.zeros(ny)
abscissa_in_PML[abscissa_in_PML_mask_bottom] = abscissa_in_PML_bottom[abscissa_in_PML_mask_bottom]
abscissa_in_PML[abscissa_in_PML_mask_top] = abscissa_in_PML_top[abscissa_in_PML_mask_top]
abscissa_normalized = abscissa_in_PML / thickness_PML_y
d_y = d0_y * abscissa_normalized ** NPOWER
K_y = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_y = ALPHA_MAX_PML * (1.0 - np.where(absc_mask_y, abscissa_normalized, 1.0))

# dampening profile in X direction at half the grid points
abscissa_in_PML_bottom = yoriginbottom - (yval + dy / 2.0)
abscissa_in_PML_top = (yval + dx / 2.0) - yorigintop
abscissa_in_PML_mask_bottom = np.where(abscissa_in_PML_bottom < 0.0, False, True)
abscissa_in_PML_mask_top = np.where(abscissa_in_PML_top < 0.0, False, True)
absc_mask_y_half = np.logical_or(abscissa_in_PML_mask_bottom, abscissa_in_PML_mask_top)
abscissa_in_PML = np.zeros(ny)
abscissa_in_PML[abscissa_in_PML_mask_bottom] = abscissa_in_PML_bottom[abscissa_in_PML_mask_bottom]
abscissa_in_PML[abscissa_in_PML_mask_top] = abscissa_in_PML_top[abscissa_in_PML_mask_top]
abscissa_normalized = abscissa_in_PML / thickness_PML_y
d_y_half = d0_y * abscissa_normalized ** NPOWER
K_y_half = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized ** NPOWER
alpha_y_half = ALPHA_MAX_PML * (1.0 - np.where(absc_mask_y_half, abscissa_normalized, 1.0))
b_y = np.exp(- (d_y / K_y + alpha_y) * dt)
b_y_half = np.exp(- (d_y_half / K_y_half + alpha_y_half) * dt)

# to avoid division by zero outside tha PML
j = np.where(d_y > 1e-6)
a_y[j] = d_y[j] * (b_y[j] - 1.0) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
j = np.where(d_y_half > 1e-6)
a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))

# # Compute the stiffness (Lame parameter) and density
kappa_unrelaxed = density * cp_unrelaxed ** 2 * np.ones((ny, nx))
rho = density * np.ones((ny, nx))
rho_half_x[:, :-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
rho_half_x[:, nx - 1] = rho_half_x[:, nx - 2]
rho_half_y[:-1, :] = 0.5 * (rho[1:, :] + rho[:-1, :])
rho_half_y[ny - 1, :] = rho_half_y[ny - 2, :]

# Acerta as dimensões dos vetores no sentido "y"
absc_mask_y = absc_mask_y[:, np.newaxis]
absc_mask_y_half = absc_mask_y_half[:, np.newaxis]
a_y = a_y[:, np.newaxis]
a_y_half = a_y_half[:, np.newaxis]
b_y = b_y[:, np.newaxis]
b_y_half = b_y_half[:, np.newaxis]
K_y = K_y[:, np.newaxis]
K_y_half = K_y_half[:, np.newaxis]

# source position
print(f"Position of the source: ")
print(f"x = {xsource}")
print(f"y = {ysource}\n")

# define source location
# Kronecker_source = np.zeros((NY, NX))
Kronecker_source[jsource, isource] = 1

# define location of receivers
print(f"There are {nrec} receivers")

if nrec > 1:
    # this is to avoid a warning with GNU gfortran at compile time about division by zero when NREC = 1
    myNREC = nrec
    xspacerec = (xfin - xdeb) / (myNREC - 1)
    yspacerec = (yfin - ydeb) / (myNREC - 1)
else:
    xspacerec = 0
    yspacerec = 0

for irec in range(0, nrec):
    xrec[irec] = xdeb + (irec) * xspacerec
    yrec[irec] = ydeb + (irec) * yspacerec

# find closest grid point for each receiver
for irec in range(0, nrec):
    dist = HUGEVAL
    for j in range(0, ny):
        for i in range(0, nx):
            distval = np.sqrt((dx * i - xrec[irec]) ** 2 + (dy * j - yrec[irec]) ** 2)
            if distval < dist:
                dist = distval
                ix_rec[irec] = i
                iy_rec[irec] = j

    print(f"Receiver {irec}:")
    print(f"x_target, y_target = {xrec[irec]}, {yrec[irec]}")
    print(f"Closest grid point at distance: {dist} in")
    print(f"i, j = {ix_rec}, {iy_rec}")

# Check the Courant stability condition for the explicit time scheme
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
Courant_number = cp_unrelaxed * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2)
print(f"Courant number is {Courant_number}")
if Courant_number > 1:
    print("time step is too large, simulation will be unstable")
    exit(1)

## Exhiibition Setup
App = pg.QtWidgets.QApplication([])

# create the instance of our Window
window = Window()
# window_dp = Window()
# window_vdp_new = Window()
# window_vdp_sum = Window()
# window_vdp_xx = Window()
# window_vdp_yy = Window()
# window_dp.setWindowTitle(f"{ny}x{nx} DP {nstep} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
# window_vdp_new.setWindowTitle(f"{ny}x{nx} VDP_NEW {nstep} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
# window_vdp_xx.setWindowTitle(f"{ny}x{nx} VDP_XX {nstep} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
# window_vdp_yy.setWindowTitle(f"{ny}x{nx} VDP_YY {nstep} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
# window_vdp_sum.setWindowTitle(f"{ny}x{nx} VDP_SUM {nstep} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")

# Start timer for simulation
start_time = perf_counter()

# beginning of time loop
# Main loop
for it in range(1, nstep):
    # Compute the first spatial derivatives divided by density
    vdp_x[:, :-1] = (p_1[:, 1:] - p_1[:, :-1]) / dx  # p_1[ny, nx] deve ser guardado
    mdp_x = b_x_half * mdp_x + a_x_half * vdp_x  # mdp_x[ny, nx] deve ser guardado, b_x_half[nx] e a_x_half[nx] cte
    vdp_y[:-1, :] = (p_1[1:, :] - p_1[:-1, :]) / dy
    mdp_y = b_y_half * mdp_y + a_y_half * vdp_y  # mdp_y[ny, nx] deve ser guardado, b_y_half[ny] e a_y_half[ny] cte
    dp_x = (vdp_x/K_x_half + mdp_x)/rho_half_x  # dp_x[ny, nx] deve ser guardado, K_x_half[nx] e rho_half_x[ny, nx] cte
    dp_y = (vdp_y/K_y_half + mdp_y)/rho_half_y  # dp_y[ny, nx] deve ser guardado, K_y_half[ny] e rho_half_y[ny, nx] cte

    # Compute the second spatial derivatives
    vdp_xx[:, 1:] = (dp_x[:, 1:] - dp_x[:, :-1]) / dx
    dmdp_x = b_x * dmdp_x + a_x * vdp_xx  # dmdp_x[ny, nx] deve ser guardado, b_x[nx] e a_x[nx] cte
    vdp_yy[1:, :] = (dp_y[1:, :] - dp_y[:-1, :]) / dy
    dmdp_y = b_y * dmdp_y + a_y * vdp_yy  # dmdp_y[ny, nx] deve ser guardado, b_y[ny] e a_y[ny] cte
    v_x = vdp_xx / K_x + dmdp_x  # v_x[ny, nx] deve ser guardado
    v_y = vdp_yy / K_y + dmdp_y  # v_y[ny, nx] deve ser guardado

    # add the source (pressure located at a given grid point)
    a = math.pi ** 2 * f0 ** 2
    t = (it - 1) * dt

    # Gaussian
    # source_term = - factor * np.exp(-a * (t-t0) ** 2) / (2.0 * a)

    # first derivative of a Gaussian
    # source_term = factor * (t - t0) * np.exp(-a * (t-t0) ** 2)

    # Ricker source time function (second derivative of a Gaussian)
    source_term = factor * (1.0 - 2.0 * a * (t - t0) ** 2) * np.exp(-a * (t - t0) ** 2)

    # apply the time evolution scheme
    # we apply it everywhere, including at some points on the edges of the domain that have not be calculated above,
    # which is of course wrong (or more precisely undefined), but this does not matter because these values
    # will be erased by the Dirichlet conditions set on these edges below
    # p_0[ny, nx], p_2[ny, nx] devem ser guardados, kappa_unrelaxed[ny, nx], cp_unrelaxed e Kronecker_source[ny, nx] cte
    p_0 = 2.0 * p_1 - p_2 + \
          dt ** 2 * ((v_x + v_y) * kappa_unrelaxed + 4.0 * math.pi * cp_unrelaxed ** 2 * source_term * Kronecker_source)

    ## apply Dirichlet conditions at the bottom of the C-PML layers
    ## which is the right condition to implement in order for C-PML to remain stable at long times
    # Dirichlet condition for pressure on the left boundary
    p_0[:, 0] = 0

    # Dirichlet condition for pressure on the right boundary
    p_0[:, nx - 1] = 0

    # Dirichlet condition for pressure on the bottom boundary
    p_0[0, :] = 0

    # Dirichlet condition for pressure on the top boundary
    p_0[ny - 1, :] = 0

    # print maximum of pressure and of norm of velocity
    pressurenorm = np.max(np.abs(p_0))
    print(f"Time step {it} out of {nstep}")
    print(f"Time: {(it - 1) * dt} seconds")
    print(f"Max absolute value of pressure = {pressurenorm}")

    # check stability of the code, exit if unstable
    if pressurenorm > STABILITY_THRESHOLD:
        print("code became unstable and blew up")
        exit(2)

    window.imv.setImage(p_0.T, levels=[-1.0, 1.0])
    # window_dp.imv.setImage(dp.T * 500_000, levels=[-1.0, 1.0])
    # window_vdp_new.imv.setImage(vdp_new.T * 500_000, levels=[-1.0, 1.0])
    # window_vdp_xx.imv.setImage(vdp_xx.T * 500_000, levels=[-1.0, 1.0])
    # window_vdp_yy.imv.setImage(vdp_yy.T * 500_000, levels=[-1.0, 1.0])
    # window_vdp_sum.imv.setImage((vdp_xx + vdp_yy).T * 500_000, levels=[-1.0, 1.0])
    # window.imv.setImage(np.log(p_0.T), levels=[-0.5, 0.5])
    App.processEvents()

    # move new values to old values (the present becomes the past, the future becomes the present)
    p_2 = p_1
    p_1 = p_0

App.exit()
end_time = perf_counter()
total_time = end_time - start_time

# End of the main loop
print("Simulation finished.")
print(f"\n\nTotal Time: {total_time} s")
print(f"Total Time: {total_time / 60} min")

print("END")
