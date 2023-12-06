import math

import matplotlib.pyplot as plt
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
import numpy as np
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
from time import time
# from datetime import datetime
from PyQt6.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget


# ==========================================================
# Esse arquivo contém as simulações realizadas dentro da GPU.
# ==========================================================

# Código para visualização da janela de simulação
# Image View class
class ImageView(pg.ImageView):
    # constructor which inherit original
    # ImageView
    def __init__(self, *args, **kwargs):
        pg.ImageView.__init__(self, *args, **kwargs)


# RawImageWidget class
class RawImageWidget(pg.widgets.RawImageWidget.RawImageGLWidget):
    # constructor which inherit original
    # RawImageWidget
    def __init__(self):
        pg.widgets.RawImageWidget.RawImageGLWidget.__init__(self)


# Window class
class Window(QMainWindow):
    def __init__(self, title=None):
        super().__init__()

        # setting title
        if title is None:
            self.setWindowTitle(f"{ny}x{nx} Grid x {NSTEP} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
        else:
            self.setWindowTitle(title)

        # setting geometry
        # self.setGeometry(200, 50, 1600, 800)
        self.setGeometry(200, 50, 300, 300)

        # setting animation
        self.isAnimated()

        # setting image
        self.image = np.random.normal(size=(300, 300))

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


# Parametros dos ensaios
flt32 = np.float32
n_iter_gpu = 1
n_iter_cpu = 1
do_sim_gpu = True
do_sim_cpu = True
do_comp_fig_cpu_gpu = True
use_refletors = False
plot_results = True
show_results = True
save_results = True
gpu_type = "NVIDIA"

# Parametros da simulacao
nx = 301  # colunas
ny = 301  # linhas

# Tamanho do grid (aparentemente em metros)
dx = 10.0
dy = dx
one_dx = one_dy = 1.0 / dx

# Constantes
PI = np.pi
DEGREES_TO_RADIANS = PI / 180.0
ZERO = 0.0
HUGEVAL = 1.0e30  # Valor enorme para o maximo da pressao
STABILITY_THRESHOLD = 1.0e25  # Limite para considerar que a simulacao esta instavel

# flags to add PML layers to the edges of the grid
USE_PML_XMIN = True
USE_PML_XMAX = True
USE_PML_YMIN = True
USE_PML_YMAX = True

# Espessura da PML in pixels
npoints_pml = 10

# Velocidades do som e densidade do meio
cp = 3300.0  # [m/s]
cs = 1000.0  # [m/s]
rho = 2800.0
mu = rho * cs * cs
lambda_ = rho * (cp * cp - 2.0 * cs * cs)
lambdaplus2mu = rho * cp * cp

# Numero total de passos de tempo
NSTEP = 2000

# Passo de tempo em segundos
dt = 1.0e-3

# Numero de iteracoes de tempo para apresentar e armazenar informacoes
IT_DISPLAY = 10

# Parametros da fonte
f0 = 7.0  # frequencia
t0 = 1.20 / f0  # delay
factor = 1.0e7
a = PI ** 2 * f0 ** 2
t = np.arange(NSTEP) * dt
ANGLE_FORCE = 0.0

# First derivative of a Gaussian
source_term = -(factor * 2.0 * a * (t - t0) * np.exp(-a * (t - t0) ** 2)).astype(flt32)

# Funcao de Ricker (segunda derivada de uma gaussiana)
# source_term = (factor * (1.0 - 2.0 * a * (t - t0) ** 2) * np.exp(-a * (t - t0) ** 2)).astype(flt32)

force_x = np.sin(ANGLE_FORCE * DEGREES_TO_RADIANS) * source_term
force_y = np.cos(ANGLE_FORCE * DEGREES_TO_RADIANS) * source_term

# Posicao da fonte
isource = int(nx / 2)
jsource = int(ny / 2)
xsource = isource * dx
ysource = jsource * dy

# Receptores
NREC = 2
# xdeb = xsource - 100.0  # em unidade de distancia
# ydeb = 2300.0  # em unidade de distancia
# xfin = xsource
# yfin = 300.0
xrec = xsource + np.array([-10, 0]) * dx
yrec = ysource + np.array([10, 30]) * dy
# sens_x = int(xdeb / dx) + 1
# sens_y = int(ydeb / dy) + 1
# sensor = np.zeros(NSTEP, dtype=flt32)  # buffer para sinal do sensor
sisvx = np.zeros((NSTEP, NREC), dtype=flt32)
sisvy = np.zeros((NSTEP, NREC), dtype=flt32)

# for evolution of total energy in the medium
vx_vy_2 = epsilon_xx = epsilon_yy = epsilon_xy = np.zeros((nx + 2, ny + 2), dtype=flt32)
total_energy = np.zeros(NSTEP, dtype=flt32)
total_energy_kinetic = np.zeros(NSTEP, dtype=flt32)
total_energy_potential = np.zeros(NSTEP, dtype=flt32)
v_solid_norm = np.zeros(NSTEP, dtype=flt32)

# Valor da potencia para calcular "d0"
NPOWER = 2.0

# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = 1.0
ALPHA_MAX_PML = 2.0 * PI * (f0 / 2.0)  # from Festa and Vilotte

# Escolha do valor de wsx (GPU)
wsx = 1
for n in range(15, 0, -1):
    if ((nx + 2) % n) == 0:
        wsx = n  # workgroup x size
        break

# Escolha do valor de wsy (GPU)
wsy = 1
for n in range(15, 0, -1):
    if ((ny + 2) % n) == 0:
        wsy = n  # workgroup x size
        break

# Arrays para as variaveis de memoria do calculo
memory_dvx_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dvx_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dvy_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dvy_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dsigmaxx_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dsigmayy_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dsigmaxy_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
memory_dsigmaxy_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)

vx = np.zeros((nx + 2, ny + 2), dtype=flt32)
vy = np.zeros((nx + 2, ny + 2), dtype=flt32)
sigmaxx = np.zeros((nx + 2, ny + 2), dtype=flt32)
sigmayy = np.zeros((nx + 2, ny + 2), dtype=flt32)
sigmaxy = np.zeros((nx + 2, ny + 2), dtype=flt32)

# Total de arrays
N_ARRAYS = 5 + 2 * 5

value_dvx_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dvx_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dvy_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dvy_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dsigmaxx_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dsigmayy_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dsigmaxy_dx = np.zeros((nx + 2, ny + 2), dtype=flt32)
value_dsigmaxy_dy = np.zeros((nx + 2, ny + 2), dtype=flt32)

# Inicializacao dos parametros da PML (definicao dos perfis de absorcao na regiao da PML)
thickness_pml_x = npoints_pml * dx
thickness_pml_y = npoints_pml * dy

# Coeficiente de reflexao (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
rcoef = 0.001

print(f'2D elastic finite-difference code in velocity and stress formulation with C-PML')
print(f'NX = {nx}')
print(f'NY = {ny}')
print(f'Tamanho do modelo ao longo do eixo X = {(nx + 1) * dx}')
print(f'Tamanho do modelo ao longo do eixo Y = {(ny + 1) * dy}')
print(f'Total de pontos no grid = {(nx + 2) * (ny + 2)}')
print(f'Number of points of all the arrays = {(nx + 2) * (ny + 2) * N_ARRAYS}')
print(f'Size in GB of all the arrays = {(nx + 2) * (ny + 2) * N_ARRAYS * 4 / (1024 * 1024 * 1024)}\n')

if NPOWER < 1:
    raise ValueError('NPOWER deve ser maior que 1')

# Calculo de d0 do relatorio da INRIA section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
d0_x = -(NPOWER + 1) * cp * np.log(rcoef) / (2.0 * thickness_pml_x)
d0_y = -(NPOWER + 1) * cp * np.log(rcoef) / (2.0 * thickness_pml_y)

print(f'd0_x = {d0_x}')
print(f'd0_y = {d0_y}')

# Amortecimento na direcao "x" (horizontal)
# Origem da PML (posicao das bordas direita e esquerda menos a espessura, em unidades de distancia)
x_orig_left = thickness_pml_x
x_orig_right = (nx - 1) * dx - thickness_pml_x

# Perfil de amortecimento na direcao "x" dentro do grid de pressao
i = np.arange(nx)
xval = dx * i
xval_pml_left = x_orig_left - xval
xval_pml_right = xval - x_orig_right
x_pml_mask_left = np.where(xval_pml_left < 0.0, False, True)
x_pml_mask_right = np.where(xval_pml_right < 0.0, False, True)
x_mask = np.logical_or(x_pml_mask_left, x_pml_mask_right)
x_pml = np.zeros(nx)
x_pml[x_pml_mask_left] = xval_pml_left[x_pml_mask_left]
x_pml[x_pml_mask_right] = xval_pml_right[x_pml_mask_right]
x_norm = x_pml / thickness_pml_x
d_x = np.expand_dims((d0_x * x_norm ** NPOWER).astype(flt32), axis=1)
k_x = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * x_norm ** NPOWER).astype(flt32), axis=1)
alpha_x = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(x_mask, x_norm, 1.0))).astype(flt32), axis=1)
b_x = np.exp(-(d_x / k_x + alpha_x) * dt).astype(flt32)
a_x = np.zeros((nx, 1), dtype=flt32)
i = np.where(d_x > 1e-6)
a_x[i] = d_x[i] * (b_x[i] - 1.0) / (k_x[i] * (d_x[i] + k_x[i] * alpha_x[i]))

# Perfil de amortecimento na direcao "x" dentro do meio grid (staggered grid)
xval_pml_left = x_orig_left - (xval + dx / 2.0)
xval_pml_right = (xval + dx / 2.0) - x_orig_right
x_pml_mask_left = np.where(xval_pml_left < 0.0, False, True)
x_pml_mask_right = np.where(xval_pml_right < 0.0, False, True)
x_mask_half = np.logical_or(x_pml_mask_left, x_pml_mask_right)
x_pml = np.zeros(nx)
x_pml[x_pml_mask_left] = xval_pml_left[x_pml_mask_left]
x_pml[x_pml_mask_right] = xval_pml_right[x_pml_mask_right]
x_norm = x_pml / thickness_pml_x
d_x_half = np.expand_dims((d0_x * x_norm ** NPOWER).astype(flt32), axis=1)
k_x_half = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * x_norm ** NPOWER).astype(flt32), axis=1)
alpha_x_half = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(x_mask_half, x_norm, 1.0))).astype(flt32), axis=1)
b_x_half = np.exp(-(d_x_half / k_x_half + alpha_x_half) * dt).astype(flt32)
a_x_half = np.zeros((nx, 1), dtype=flt32)
i = np.where(d_x_half > 1e-6)
a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (k_x_half[i] * (d_x_half[i] + k_x_half[i] * alpha_x_half[i]))

# Amortecimento na direcao "y" (vertical)
# Origem da PML (posicao das bordas superior e inferior menos a espessura, em unidades de distancia)
y_orig_top = thickness_pml_y
y_orig_bottom = (ny - 1) * dy - thickness_pml_y

# Perfil de amortecimento na direcao "y" dentro do grid
j = np.arange(ny)
yval = dy * j
y_pml_top = y_orig_top - yval
y_pml_bottom = yval - y_orig_bottom
y_pml_mask_top = np.where(y_pml_top < 0.0, False, True)
y_pml_mask_bottom = np.where(y_pml_bottom < 0.0, False, True)
y_mask = np.logical_or(y_pml_mask_top, y_pml_mask_bottom)
y_pml = np.zeros(ny)
y_pml[y_pml_mask_top] = y_pml_top[y_pml_mask_top]
y_pml[y_pml_mask_bottom] = y_pml_bottom[y_pml_mask_bottom]
y_norm = y_pml / thickness_pml_y
d_y = np.expand_dims((d0_y * y_norm ** NPOWER).astype(flt32), axis=0)
k_y = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * y_norm ** NPOWER).astype(flt32), axis=0)
alpha_y = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(y_mask, y_norm, 1.0))).astype(flt32), axis=0)
b_y = np.exp(-(d_y / k_y + alpha_y) * dt).astype(flt32)
a_y = np.zeros((1, ny), dtype=flt32)
j = np.where(d_y > 1e-6)
a_y[j] = d_y[j] * (b_y[j] - 1.0) / (k_y[j] * (d_y[j] + k_y[j] * alpha_y[j]))

# Perfil de amortecimento na direcao "y" dentro do meio grid (staggered grid)
y_pml_top = y_orig_top - (yval + dy / 2.0)
y_pml_bottom = (yval + dy / 2.0) - y_orig_bottom
y_pml_mask_top = np.where(y_pml_top < 0.0, False, True)
y_pml_mask_bottom = np.where(y_pml_bottom < 0.0, False, True)
y_mask_half = np.logical_or(y_pml_mask_top, y_pml_mask_bottom)
y_pml = np.zeros(ny)
y_pml[y_pml_mask_top] = y_pml_top[y_pml_mask_top]
y_pml[y_pml_mask_bottom] = y_pml_bottom[y_pml_mask_bottom]
y_norm = y_pml / thickness_pml_y
d_y_half = np.expand_dims((d0_y * y_norm ** NPOWER).astype(flt32), axis=0)
k_y_half = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * y_norm ** NPOWER).astype(flt32), axis=0)
alpha_y_half = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(y_mask_half, y_norm, 1.0))).astype(flt32), axis=0)
b_y_half = np.exp(-(d_y_half / k_y_half + alpha_y_half) * dt).astype(flt32)
a_y_half = np.zeros((1, ny), dtype=flt32)
j = np.where(d_y_half > 1e-6)
a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (k_y_half[j] * (d_y_half[j] + k_y_half[j] * alpha_y_half[j]))

# Imprime a posicao da fonte
print(f'Posicao da fonte:')
print(f'x = {xsource}')
print(f'y = {ysource}')

# Define a localizacao dos receptores
print(f'Existem {NREC} receptores')

# Find closest grid point for each receiver
ix_rec = np.zeros(NREC, dtype=int)
iy_rec = np.zeros(NREC, dtype=int)
for irec in range(NREC):
    dist = HUGEVAL
    for j in range(ny):
        for i in range(nx):
            distval = math.sqrt((dx * i - xrec[irec]) ** 2 + (dx * j - yrec[irec]) ** 2)
            if distval < dist:
                dist = distval
                ix_rec[irec] = i
                iy_rec[irec] = j

    print(f'receiver {irec}: x_target = {xrec[irec]}, y_target = {yrec[irec]}')
    print(f'Ponto mais perto do grid encontrado na distancia {dist}, em i = {ix_rec[irec]}, j = {iy_rec[irec]}\n')

# Verifica a condicao de estabilidade de Courant
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
courant_number = cp * dt * math.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2)
print(f'Numero de Courant e {courant_number}')
if courant_number > 1:
    print("O passo de tempo e muito longo, a simulacao sera instavel")
    exit(1)


def sim_cpu():
    global source_term, force_x, force_y
    global a_x, a_x_half, b_x, b_x_half, k_x, k_x_half
    global a_y, a_y_half, b_y, b_y_half, k_y, k_y_half
    global vx, vy, sigmaxx, sigmayy, sigmaxy
    global memory_dvx_dx, memory_dvx_dy
    global memory_dvy_dx, memory_dvy_dy
    global memory_dsigmaxx_dx, memory_dsigmayy_dy
    global memory_dsigmaxy_dx, memory_dsigmaxy_dy
    global value_dvx_dx, value_dvy_dy
    global value_dsigmaxx_dx, value_dsigmaxy_dy
    global value_dsigmaxy_dx, value_dsigmayy_dy
    global sisvx, sisvy
    global total_energy, total_energy_kinetic, total_energy_potential, v_solid_norm
    global epsilon_xx, epsilon_yy, epsilon_xy

    # Configuracao e inicializacao da janela de exibicao
    App = pg.QtWidgets.QApplication([])
    windowVx = Window('Vx')
    windowVx.setGeometry(200, 50, vx.shape[0], vx.shape[1])
    windowVy = Window('Vy')
    windowVy.setGeometry(200, 50, vy.shape[0], vy.shape[1])
    windowVxVy = Window('Vx + Vy')
    windowVxVy.setGeometry(200, 50, vy.shape[0], vy.shape[1])

    DELTAT_over_rho = dt / rho
    denom = np.float32(4.0 * mu * (lambda_ + mu))

    v_min = -1.0
    v_max = - v_min
    # Inicio do laco de tempo
    for it in range(1, NSTEP + 1):
        print(f'it = {it}')

        # Calculo da tensao [stress] - {sigma} (equivalente a pressao nos gases-liquidos)
        # sigma_ii -> tensoes normais; sigma_ij -> tensoes cisalhantes
        # Primeiro "laco" i: 1,NX-1; j: 2,NY -> [1:-2, 2:-1]
        value_dvx_dx[1:-2, 2:-1] = ((27.0 * (vx[2:-1, 2:-1] - vx[1:-2, 2:-1]) - vx[3:, 2:-1] + vx[:-3, 2:-1]) *
                                    one_dx / 24.0)
        value_dvy_dy[1:-2, 2:-1] = ((27.0 * (vy[1:-2, 2:-1] - vy[1:-2, 1:-2]) - vy[1:-2, 3:] + vy[1:-2, :-3]) *
                                    one_dy / 24.0)

        memory_dvx_dx[1:-2, 2:-1] = (b_x_half[:-1, :] * memory_dvx_dx[1:-2, 2:-1] +
                                     a_x_half[:-1, :] * value_dvx_dx[1:-2, 2:-1])
        memory_dvy_dy[1:-2, 2:-1] = (b_y[:, 1:] * memory_dvy_dy[1:-2, 2:-1] +
                                     a_y[:, 1:] * value_dvy_dy[1:-2, 2:-1])

        value_dvx_dx[1:-2, 2:-1] = value_dvx_dx[1:-2, 2:-1] / k_x_half[:-1, :] + memory_dvx_dx[1:-2, 2:-1]
        value_dvy_dy[1:-2, 2:-1] = value_dvy_dy[1:-2, 2:-1] / k_y[:, 1:] + memory_dvy_dy[1:-2, 2:-1]

        # compute the stress using the Lame parameters
        sigmaxx = sigmaxx + (lambdaplus2mu * value_dvx_dx + lambda_ * value_dvy_dy) * dt
        sigmayy = sigmayy + (lambda_ * value_dvx_dx + lambdaplus2mu * value_dvy_dy) * dt

        # Segundo "laco" i: 2,NX; j: 1,NY-1 -> [2:-1, 1:-2]
        value_dvy_dx[2:-1, 1:-2] = ((27.0 * (vy[2:-1, 1:-2] - vy[1:-2, 1:-2]) - vy[3:, 1:-2] + vy[:-3, 1:-2]) *
                                    one_dx / 24.0)
        value_dvx_dy[2:-1, 1:-2] = ((27.0 * (vx[2:-1, 2:-1] - vx[2:-1, 1:-2]) - vx[2:-1, 3:] + vx[2:-1, :-3]) *
                                    one_dy / 24.0)

        memory_dvy_dx[2:-1, 1:-2] = (b_x[1:, :] * memory_dvy_dx[2:-1, 1:-2] +
                                     a_x[1:, :] * value_dvy_dx[2:-1, 1:-2])
        memory_dvx_dy[2:-1, 1:-2] = (b_y_half[:, :-1] * memory_dvx_dy[2:-1, 1:-2] +
                                     a_y_half[:, :-1] * value_dvx_dy[2:-1, 1:-2])

        value_dvy_dx[2:-1, 1:-2] = value_dvy_dx[2:-1, 1:-2] / k_x[1:, :] + memory_dvy_dx[2:-1, 1:-2]
        value_dvx_dy[2:-1, 1:-2] = value_dvx_dy[2:-1, 1:-2] / k_y_half[:, :-1] + memory_dvx_dy[2:-1, 1:-2]

        # compute the stress using the Lame parameters
        sigmaxy = sigmaxy + dt * mu * (value_dvx_dy + value_dvy_dx)

        # Calculo da velocidade
        # Primeiro "laco" i: 2,NX; j: 2,NY -> [2:-1, 2:-1]
        value_dsigmaxx_dx[2:-1, 2:-1] = (27.0 * (sigmaxx[2:-1, 2:-1] - sigmaxx[1:-2, 2:-1]) -
                                         sigmaxx[3:, 2:-1] + sigmaxx[:-3, 2:-1]) * one_dx / 24.0
        value_dsigmaxy_dy[2:-1, 2:-1] = (27.0 * (sigmaxy[2:-1, 2:-1] - sigmaxy[2:-1, 1:-2]) -
                                         sigmaxy[2:-1, 3:] + sigmaxy[2:-1, :-3]) * one_dy / 24.0

        memory_dsigmaxx_dx[2:-1, 2:-1] = (b_x[1:, :] * memory_dsigmaxx_dx[2:-1, 2:-1] +
                                          a_x[1:, :] * value_dsigmaxx_dx[2:-1, 2:-1])
        memory_dsigmaxy_dy[2:-1, 2:-1] = (b_y[:, 1:] * memory_dsigmaxy_dy[2:-1, 2:-1] +
                                          a_y[:, 1:] * value_dsigmaxy_dy[2:-1, 2:-1])

        value_dsigmaxx_dx[2:-1, 2:-1] = value_dsigmaxx_dx[2:-1, 2:-1] / k_x[1:, :] + memory_dsigmaxx_dx[2:-1, 2:-1]
        value_dsigmaxy_dy[2:-1, 2:-1] = value_dsigmaxy_dy[2:-1, 2:-1] / k_y[:, 1:] + memory_dsigmaxy_dy[2:-1, 2:-1]

        vx = DELTAT_over_rho * (value_dsigmaxx_dx + value_dsigmaxy_dy) + vx

        # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2]
        value_dsigmaxy_dx[1:-2, 1:-2] = (27.0 * (sigmaxy[2:-1, 1:-2] - sigmaxy[1:-2, 1:-2]) -
                                         sigmaxy[3:, 1:-2] + sigmaxy[:-3, 1:-2]) * one_dx / 24.0
        value_dsigmayy_dy[1:-2, 1:-2] = (27.0 * (sigmayy[1:-2, 2:-1] - sigmayy[1:-2, 1:-2]) -
                                         sigmayy[1:-2, 3:] + sigmayy[1:-2, :-3]) * one_dy / 24.0

        memory_dsigmaxy_dx[1:-2, 1:-2] = (b_x_half[:-1, :] * memory_dsigmaxy_dx[1:-2, 1:-2] +
                                          a_x_half[:-1, :] * value_dsigmaxy_dx[1:-2, 1:-2])
        memory_dsigmayy_dy[1:-2, 1:-2] = (b_y_half[:, :-1] * memory_dsigmayy_dy[1:-2, 1:-2] +
                                          a_y_half[:, :-1] * value_dsigmayy_dy[1:-2, 1:-2])

        value_dsigmaxy_dx[1:-2, 1:-2] = (value_dsigmaxy_dx[1:-2, 1:-2] / k_x_half[:-1, :] +
                                         memory_dsigmaxy_dx[1:-2, 1:-2])
        value_dsigmayy_dy[1:-2, 1:-2] = (value_dsigmayy_dy[1:-2, 1:-2] / k_y_half[:, :-1] +
                                         memory_dsigmayy_dy[1:-2, 1:-2])

        vy = DELTAT_over_rho * (value_dsigmaxy_dx + value_dsigmayy_dy) + vy

        # add the source (force vector located at a given grid point)
        vx[isource, jsource] += force_x[it - 1] * dt / rho
        vy[isource, jsource] += force_y[it - 1] * dt / rho

        # implement Dirichlet boundary conditions on the six edges of the grid
        # which is the right condition to implement in order for C-PML to remain stable at long times
        # xmin
        vx[:2, :] = ZERO
        vy[:2, :] = ZERO

        # xmax
        vx[-2:, :] = ZERO
        vy[-2:, :] = ZERO

        # ymin
        vx[:, :2] = ZERO
        vy[:, :2] = ZERO

        # ymax
        vx[:, -2:] = ZERO
        vy[:, -2:] = ZERO

        # Store seismograms
        for _irec in range(NREC):
            sisvx[it - 1, _irec] = vx[ix_rec[_irec], iy_rec[_irec]]
            sisvy[it - 1, _irec] = vy[ix_rec[_irec], iy_rec[_irec]]

        # Compute total energy in the medium (without the PML layers)
        imin = npoints_pml
        imax = nx - npoints_pml + 1
        jmin = npoints_pml
        jmax = ny - npoints_pml + 1

        local_energy_kinetic = 0.5 * rho * (np.sum(vx[imin: imax, jmin: jmax] ** 2) +
                                            np.sum(vy[imin: imax, jmin: jmax] ** 2))

        # compute total field from split components
        epsilon_xx[imin: imax, jmin: jmax] = (lambdaplus2mu * sigmaxx[imin: imax, jmin: jmax] -
                                              lambda_ * sigmayy[imin: imax, jmin: jmax]) / denom
        epsilon_yy[imin: imax, jmin: jmax] = (lambdaplus2mu * sigmayy[imin: imax, jmin: jmax] -
                                              lambda_ * sigmaxx[imin: imax, jmin: jmax]) / denom
        epsilon_xy[imin: imax, jmin: jmax] = sigmaxy[imin: imax, jmin: jmax] / (2.0 * mu)

        local_energy_potential = 0.5 * np.sum(epsilon_xx * sigmaxx + epsilon_yy * sigmayy + 2.0 * epsilon_xy * sigmaxy)
        total_energy[it - 1] = local_energy_kinetic + local_energy_potential

        v_solid_norm[it - 1] = np.max(np.sqrt(vx[:, :] ** 2 + vy[:, :] ** 2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            print(f'Time step # {it} out of {NSTEP}')
            print(f'Max Vx = {np.max(vx)}, Vy = {np.max(vy)}')
            print(f'Min Vx = {np.min(vx)}, Vy = {np.min(vy)}')
            print(f'Max norm velocity vector V (m/s) = {v_solid_norm[it - 1]}')
            print(f'Total energy = {total_energy[it - 1]}')

        windowVx.imv.setImage(vx[:, :], levels=[v_min / 1.0, v_max / 1.0])
        windowVy.imv.setImage(vy[:, :], levels=[v_min / 1.0, v_max / 1.0])
        windowVxVy.imv.setImage(vx[:, :] + vy[:, :], levels=[2.0 * v_min / 1.0, 2.0 * v_max / 1.0])
        App.processEvents()

        # Verifica a estabilidade da simulacao
        if v_solid_norm[it - 1] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instável")
            exit(2)

    App.exit()

    # End of the main loop
    # print("Simulacao terminada.")


# Shader [kernel] para a simulação em WEBGPU
shader_test = f"""
    struct SimIntValues {{
        x_sz: i32,          // x field size
        y_sz: i32,          // y field size
        x_source: i32,      // x source index
        y_source: i32,      // y source index
        x_sens: i32,        // x sensor
        y_sens: i32,        // y sensor
        np_pml: i32,        // PML size
        n_iter: i32,        // max iterations
        it: i32             // time iteraction
    }};

    struct SimFltValues {{
        cp: f32,            // longitudinal sound speed
        cs: f32,            // transverse sound speed
        dx: f32,            // delta x
        dy: f32,            // delta y
        dt: f32,            // delta t
        rho: f32,           // density
        lambda: f32,        // Lame parameter
        mu : f32,           // Lame parameter
        lambdaplus2mu: f32  // Lame parameter
    }};

    // Group 0 - parameters
    @group(0) @binding(0)   // param_flt32
    var<storage,read> sim_flt_par: SimFltValues;

    @group(0) @binding(1) // force terms
    var<storage,read> force: array<f32>;

    @group(0) @binding(2) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
    var<storage,read> coef_x: array<f32>;

    @group(0) @binding(3) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
    var<storage,read> coef_y: array<f32>;
    
    @group(0) @binding(4) // param_int32
    var<storage,read_write> sim_int_par: SimIntValues;

    // Group 1 - simulation arrays
    @group(1) @binding(5) // velocity fields (vx, vy)
    var<storage,read_write> vel: array<f32>;
    
    @group(1) @binding(6) // stress fields (sigmaxx, sigmayy, sigmaxy)
    var<storage,read_write> sig: array<f32>;

    @group(1) @binding(7) // memory fields
                          // memory_dvx_dx, memory_dvx_dy, memory_dvy_dx, memory_dvy_dy,
                          // memory_dsigmaxx_dx, memory_dsigmayy_dy, memory_dsigmaxy_dx, memory_dsigmaxy_dy
    var<storage,read_write> memo: array<f32>;
    
    // Group 2 - sensors arrays and energies
    @group(2) @binding(8) // sensors signals (sisvx, sisvy)
    var<storage,read_write> sensors: array<f32>;
    
    @group(2) @binding(9) // epsilon fields
                          // epsilon_xx, epsilon_yy, epsilon_xy, vx_vy_2
    var<storage,read_write> eps: array<f32>;
    
    @group(2) @binding(10) // energy fields
                           // total_energy, total_energy_kinetic, total_energy_potential, v_solid_norm
    var<storage,read_write> energy: array<f32>;
   
    
    // -------------------------------
    // --- Index access functions ----
    // -------------------------------
    // function to convert 2D [x,y] index into 1D [xy] index
    fn xy(x: i32, y: i32) -> i32 {{
        let index = y + x * sim_int_par.y_sz;

        return select(-1, index, x >= 0 && x < sim_int_par.x_sz && y >= 0 && y < sim_int_par.y_sz);
    }}

    // function to convert 2D [i,j] index into 1D [] index
    fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {{
        let index = j + i * j_max;

        return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
    }}

    // function to convert 3D [i,j,k] index into 1D [] index
    fn ijk(i: i32, j: i32, k: i32, i_max: i32, j_max: i32, k_max: i32) -> i32 {{
        let index = j + i * j_max + k * j_max * i_max;

        return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max);
    }}

    // ------------------------------------
    // --- Force array access funtions ---
    // ------------------------------------ 
    // function to get a force_x array value [force_x]
    fn get_force_x(n: i32) -> f32 {{
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 2);

        return select(0.0, force[index], index != -1);
    }}
    
    // function to get a force_y array value [force_y]
    fn get_force_y(n: i32) -> f32 {{
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 2);

        return select(0.0, force[index], index != -1);
    }}
    
    // -------------------------------------------------
    // --- CPML X coefficients array access funtions ---
    // -------------------------------------------------
    // function to get a a_x array value
    fn get_a_x(n: i32) -> f32 {{
        let index: i32 = ij(n, 0, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a a_x_h array value
    fn get_a_x_h(n: i32) -> f32 {{
        let index: i32 = ij(n, 3, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }}

    // function to get a b_x array value
    fn get_b_x(n: i32) -> f32 {{
        let index: i32 = ij(n, 1, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a b_x_h array value
    fn get_b_x_h(n: i32) -> f32 {{
        let index: i32 = ij(n, 4, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }}

    // function to get a k_x array value
    fn get_k_x(n: i32) -> f32 {{
        let index: i32 = ij(n, 2, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }}

    // function to get a k_x_h array value
    fn get_k_x_h(n: i32) -> f32 {{
        let index: i32 = ij(n, 5, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }}

    // -------------------------------------------------
    // --- CPML Y coefficients array access funtions ---
    // -------------------------------------------------
    // function to get a a_y array value
    fn get_a_y(n: i32) -> f32 {{
        let index: i32 = ij(0, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }}

    // function to get a a_y_h array value
    fn get_a_y_h(n: i32) -> f32 {{
        let index: i32 = ij(3, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }}

    // function to get a b_y array value
    fn get_b_y(n: i32) -> f32 {{
        let index: i32 = ij(1, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }}

    // function to get a b_y_h array value
    fn get_b_y_h(n: i32) -> f32 {{
        let index: i32 = ij(4, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }}

    // function to get a k_y array value
    fn get_k_y(n: i32) -> f32 {{
        let index: i32 = ij(2, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }}

    // function to get a k_y_h array value
    fn get_k_y_h(n: i32) -> f32 {{
        let index: i32 = ij(5, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }}
    
    // ---------------------------------------
    // --- Velocity arrays access funtions ---
    // ---------------------------------------
    // function to get a vx array value
    fn get_vx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        return select(0.0, vel[index], index != -1);
    }}

    // function to set a vx array value
    fn set_vx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        if(index != -1) {{
            vel[index] = val;
        }}
    }}

    // function to get a vy array value
    fn get_vy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        return select(0.0, vel[index], index != -1);
    }}

    // function to set a vy array value
    fn set_vy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        if(index != -1) {{
            vel[index] = val;
        }}
    }}

    // -------------------------------------
    // --- Stress arrays access funtions ---
    // -------------------------------------
    // function to get a sigmaxx array value
    fn get_sigmaxx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        return select(0.0, sig[index], index != -1);
    }}

    // function to set a sigmaxx array value
    fn set_sigmaxx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        if(index != -1) {{
            sig[index] = val;
        }}
    }}

    // function to get a sigmayy array value
    fn get_sigmayy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        return select(0.0, sig[index], index != -1);
    }}

    // function to set a sigmayy array value
    fn set_sigmayy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        if(index != -1) {{
            sig[index] = val;
        }}
    }}

    // function to get a sigmaxy array value
    fn get_sigmaxy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        return select(0.0, sig[index], index != -1);
    }}

    // function to set a sigmaxy array value
    fn set_sigmaxy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        if(index != -1) {{
            sig[index] = val;
        }}
    }}
    
    // -------------------------------------
    // --- Memory arrays access funtions ---
    // -------------------------------------
    // function to get a memory_dvx_dx array value
    fn get_mdvx_dx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dvx_dx array value
    fn set_mdvx_dx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dvx_dy array value
    fn get_mdvx_dy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dvx_dy array value
    fn set_mdvx_dy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dvy_dx array value
    fn get_mdvy_dx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dvy_dx array value
    fn set_mdvy_dx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dvy_dy array value
    fn get_mdvy_dy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dvy_dy array value
    fn set_mdvy_dy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dsigmaxx_dx array value
    fn get_mdsxx_dx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 4, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dsigmaxx_dx array value
    fn set_mdsxx_dx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 4, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dsigmayy_dy array value
    fn get_mdsyy_dy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 5, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dsigmayy_dy array value
    fn set_mdsyy_dy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 5, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dsigmaxy_dx array value
    fn get_mdsxy_dx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 6, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dsigmaxy_dx array value
    fn set_mdsxy_dx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 6, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // function to get a memory_dsigmaxy_dy array value
    fn get_mdsxy_dy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 7, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }}

    // function to set a memory_dsigmaxy_dy array value
    fn set_mdsxy_dy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 7, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {{
            memo[index] = val;
        }}
    }}
    
    // --------------------------------------
    // --- Sensors arrays access funtions ---
    // --------------------------------------
    // function to set a sens_vx array value
    fn set_sens_vx(n: i32, val : f32) {{
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 2);

        if(index != -1) {{
            sensors[index] = val;
        }}
    }}
    
    // function to set a sens_vy array value
    fn set_sens_vy(n: i32, val : f32) {{
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 2);

        if(index != -1) {{
            sensors[index] = val;
        }}
    }}
    
    // -------------------------------------
    // --- Epsilon arrays access funtions ---
    // -------------------------------------
    // function to get a epsilon_xx array value
    fn get_eps_xx(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }}

    // function to set a epsilon_xx array value
    fn set_eps_xx(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {{
            eps[index] = val;
        }}
    }}

    // function to get a epsilon_yy array value
    fn get_eps_yy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }}

    // function to set a epsilon_yy array value
    fn set_eps_yy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {{
            eps[index] = val;
        }}
    }}

    // function to get a epsilon_xy array value
    fn get_eps_xy(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }}

    // function to set a epsilon_xy array value
    fn set_eps_xy(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {{
            eps[index] = val;
        }}
    }}
    
    // function to get a vx_vy_2 array value
    fn get_vx_vy_2(x: i32, y: i32) -> f32 {{
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }}

    // function to set a vx_vy_2 array value
    fn set_vx_vy_2(x: i32, y: i32, val : f32) {{
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {{
            eps[index] = val;
        }}
    }}
    
    // ------------------------------------
    // --- Energy array access funtions ---
    // ------------------------------------ 
    // function to get a total_energy array value [tot_en]
    fn get_tot_en(n: i32) -> f32 {{
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }}
    
    // function to set a total_energy array value
    fn set_tot_en(n: i32, val : f32) {{
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 4);

        if(index != -1) {{
            energy[index] = val;
        }}
    }}
    
    // function to get a total_energy_kinetic array value [tot_en_k]
    fn get_tot_en_k(n: i32) -> f32 {{
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }}
    
    // function to set a total_energy_kinetic array value
    fn set_tot_en_k(n: i32, val : f32) {{
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 4);

        if(index != -1) {{
            energy[index] = val;
        }}
    }}
    
    // function to get a total_energy_potencial array value [tot_en_p]
    fn get_tot_en_p(n: i32) -> f32 {{
        let index: i32 = ij(n, 2, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }}
    
    // function to set a total_energy_potencial array value
    fn set_tot_en_p(n: i32, val : f32) {{
        let index: i32 = ij(n, 2, sim_int_par.n_iter, 4);

        if(index != -1) {{
            energy[index] = val;
        }}
    }}
    
    // function to get a v_solid_norm array value [v_sol_n]
    fn get_v_sol_n(n: i32) -> f32 {{
        let index: i32 = ij(n, 3, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }}
    
    // function to set a v_solid_norm array value
    fn set_v_sol_n(n: i32, val : f32) {{
        let index: i32 = ij(n, 3, sim_int_par.n_iter, 4);

        if(index != -1) {{
            energy[index] = val;
        }}
    }}
    
    // ---------------
    // --- Kernels ---
    // ---------------
    // Kernel to calculate stresses [sigmaxx, sigmayy, sigmaxy]
    @compute
    @workgroup_size({wsx}, {wsy})
    fn sigma_kernel(@builtin(global_invocation_id) index: vec3<u32>) {{
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dx: f32 = sim_flt_par.dx;
        let dy: f32 = sim_flt_par.dy;
        let dt: f32 = sim_flt_par.dt;
        let lambda: f32 = sim_flt_par.lambda;
        let mu: f32 = sim_flt_par.mu;
        let lambdaplus2mu: f32 = lambda + 2.0 * mu;
     
        // Normal stresses
        if(x >= 1 && x < (sim_int_par.x_sz - 2) && y >= 2 && y < (sim_int_par.y_sz - 1)) {{
            let vx_x1_y: f32 = get_vx(x + 1, y);
            let vx_x_y: f32 = get_vx(x, y);
            let vx_x2_y: f32 = get_vx(x + 2, y);
            let vx_1x_y: f32 = get_vx(x - 1, y);
            var vdvx_dx: f32 = (27.0*(vx_x1_y - vx_x_y) - vx_x2_y + vx_1x_y)/(24.0 * dx);
            
            let vy_x_y: f32 = get_vy(x, y);
            let vy_x_1y: f32 = get_vy(x, y - 1);
            let vy_x_y1: f32 = get_vy(x, y + 1);
            let vy_x_2y: f32 = get_vy(x, y - 2);
            var vdvy_dy: f32 = (27.0*(vy_x_y - vy_x_1y) - vy_x_y1 + vy_x_2y)/(24.0 * dy);

            let mdvx_dx: f32 = get_mdvx_dx(x, y);
            let a_x_h = get_a_x_h(x - 1);
            let b_x_h = get_b_x_h(x - 1);
            var mdvx_dx_new: f32 = b_x_h * mdvx_dx + a_x_h * vdvx_dx;
            
            let mdvy_dy: f32 = get_mdvy_dy(x, y);
            let a_y = get_a_y(y - 1);
            let b_y = get_b_y(y - 1);
            var mdvy_dy_new: f32 = b_y * mdvy_dy + a_y * vdvy_dy;
                    
            let k_x_h = get_k_x_h(x - 1);
            vdvx_dx = vdvx_dx/k_x_h + mdvx_dx_new;
            
            let k_y = get_k_y(y - 1);
            vdvy_dy = vdvy_dy/k_y  + mdvy_dy_new;
            
            set_mdvx_dx(x, y, mdvx_dx_new);
            set_mdvy_dy(x, y, mdvy_dy_new);
        
            let sigmaxx: f32 = get_sigmaxx(x, y);
            set_sigmaxx(x, y, sigmaxx + (lambdaplus2mu * vdvx_dx + lambda        * vdvy_dy) * dt);
            
            let sigmayy: f32 = get_sigmayy(x, y);
            set_sigmayy(x, y, sigmayy + (lambda        * vdvx_dx + lambdaplus2mu * vdvy_dy) * dt);
        }}
        
        // Shear stress
        if(x >= 2 && x < (sim_int_par.x_sz - 1) && y >= 1 && y < (sim_int_par.y_sz - 2)) {{
            let vy_x_y: f32 = get_vy(x, y);
            let vy_1x_y: f32 = get_vy(x - 1, y);
            let vy_x1_y: f32 = get_vy(x + 1, y);
            let vy_2x_y: f32 = get_vy(x - 2, y);
            var vdvy_dx: f32 = (27.0*(vy_x_y - vy_1x_y) - vy_x1_y + vy_2x_y)/(24.0 * dx);
            
            let vx_x_y1: f32 = get_vx(x, y + 1);
            let vx_x_y: f32 = get_vx(x, y);
            let vx_x_y2: f32 = get_vx(x, y + 2);
            let vx_x_1y: f32 = get_vx(x, y - 1);
            var vdvx_dy: f32 = (27.0*(vx_x_y1 - vx_x_y) - vx_x_y2 + vx_x_1y)/(24.0 * dy);

            let mdvy_dx: f32 = get_mdvy_dx(x, y);
            let a_x = get_a_x(x - 1);
            let b_x = get_b_x(x - 1);
            var mdvy_dx_new: f32 = b_x * mdvy_dx + a_x * vdvy_dx;
            
            let mdvx_dy: f32 = get_mdvx_dy(x, y);
            let a_y_h = get_a_y_h(y - 1);
            let b_y_h = get_b_y_h(y - 1);
            var mdvx_dy_new: f32 = b_y_h * mdvx_dy + a_y_h * vdvx_dy;
        
            let k_x = get_k_x(x - 1);
            vdvy_dx = vdvy_dx/k_x   + mdvy_dx_new;
            
            let k_y_h = get_k_y_h(y - 1);
            vdvx_dy = vdvx_dy/k_y_h + mdvx_dy_new;
            
            set_mdvy_dx(x, y, mdvy_dx_new);
            set_mdvx_dy(x, y, mdvx_dy_new);
        
            let sigmaxy: f32 = get_sigmaxy(x, y);
            set_sigmaxy(x, y, sigmaxy + (vdvx_dy + vdvy_dx) * mu * dt);
        }}
    }}

    // Kernel to calculate velocities [vx, vy]
    @compute
    @workgroup_size({wsx}, {wsy})
    fn velocity_kernel(@builtin(global_invocation_id) index: vec3<u32>) {{
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
        let dx: f32 = sim_flt_par.dx;
        let dy: f32 = sim_flt_par.dy;
        
        // first step
        if(x >= 2 && x < (sim_int_par.x_sz - 1) && y >= 2 && y < (sim_int_par.y_sz - 1)) {{
            let sigmaxx_x_y: f32 = get_sigmaxx(x, y);
            let sigmaxx_1x_y: f32 = get_sigmaxx(x - 1, y);
            let sigmaxx_x1_y: f32 = get_sigmaxx(x + 1, y);
            let sigmaxx_2x_y: f32 = get_sigmaxx(x - 2, y); 
            var vdsigmaxx_dx: f32 = (27.0*(sigmaxx_x_y - sigmaxx_1x_y) - sigmaxx_x1_y + sigmaxx_2x_y)/(24.0 * dx);
            
            let sigmaxy_x_y: f32 = get_sigmaxy(x, y);
            let sigmaxy_x_1y: f32 = get_sigmaxy(x, y - 1);
            let sigmaxy_x_y1: f32 = get_sigmaxy(x, y + 1);
            let sigmaxy_x_2y: f32 = get_sigmaxy(x, y - 2);
            var vdsigmaxy_dy: f32 = (27.0*(sigmaxy_x_y - sigmaxy_x_1y) - sigmaxy_x_y1 + sigmaxy_x_2y)/(24.0 * dy);

            let mdsxx_dx: f32 = get_mdsxx_dx(x, y);
            let a_x = get_a_x(x - 1);
            let b_x = get_b_x(x - 1);
            var mdsxx_dx_new: f32 = b_x * mdsxx_dx + a_x * vdsigmaxx_dx;
            
            let mdsxy_dy: f32 = get_mdsxy_dy(x, y);
            let a_y = get_a_y(y - 1);
            let b_y = get_b_y(y - 1);
            var mdsxy_dy_new: f32 = b_y * mdsxy_dy + a_y * vdsigmaxy_dy;
        
            let k_x = get_k_x(x - 1);
            vdsigmaxx_dx = vdsigmaxx_dx/k_x + mdsxx_dx_new;
            
            let k_y = get_k_y(y - 1);
            vdsigmaxy_dy = vdsigmaxy_dy/k_y + mdsxy_dy_new;
            
            set_mdsxx_dx(x, y, mdsxx_dx_new);
            set_mdsxy_dy(x, y, mdsxy_dy_new);
            
            let vx: f32 = get_vx(x, y);
            set_vx(x, y, dt_over_rho * (vdsigmaxx_dx + vdsigmaxy_dy) + vx);
        }}
        
        // second step
        if(x >= 1 && x < (sim_int_par.x_sz - 2) && y >= 1 && y < (sim_int_par.y_sz - 2)) {{
            let sigmaxy_x1_y: f32 = get_sigmaxy(x + 1, y);
            let sigmaxy_x_y: f32 = get_sigmaxy(x, y);
            let sigmaxy_x2_y: f32 = get_sigmaxy(x + 2, y);
            let sigmaxy_1x_y: f32 = get_sigmaxy(x - 1, y);
            var vdsigmaxy_dx: f32 = (27.0*(sigmaxy_x1_y - sigmaxy_x_y) - sigmaxy_x2_y + sigmaxy_1x_y)/(24.0 * dx);
            
            let sigmayy_x_y1: f32 = get_sigmayy(x, y + 1);
            let sigmayy_x_y: f32 = get_sigmayy(x, y);
            let sigmayy_x_y2: f32 = get_sigmayy(x, y + 2);
            let sigmayy_x_1y: f32 = get_sigmayy(x, y - 1);    
            var vdsigmayy_dy: f32 = (27.0*(sigmayy_x_y1 - sigmayy_x_y) - sigmayy_x_y2 + sigmayy_x_1y)/(24.0 * dy);

            let mdsxy_dx: f32 = get_mdsxy_dx(x, y);
            let a_x_h = get_a_x_h(x - 1);
            let b_x_h = get_b_x_h(x - 1);
            var mdsxy_dx_new: f32 = b_x_h * mdsxy_dx + a_x_h * vdsigmaxy_dx;
            
            let mdsyy_dy: f32 = get_mdsyy_dy(x, y);
            let a_y_h = get_a_y_h(y - 1);
            let b_y_h = get_b_y_h(y - 1);
            var mdsyy_dy_new: f32 = b_y_h * mdsyy_dy + a_y_h * vdsigmayy_dy;
        
            let k_x_h = get_k_x_h(x - 1);
            vdsigmaxy_dx = vdsigmaxy_dx/k_x_h + mdsxy_dx_new;
            
            let k_y_h = get_k_y_h(y - 1);
            vdsigmayy_dy = vdsigmayy_dy/k_y_h + mdsyy_dy_new;
            
            set_mdsxy_dx(x, y, mdsxy_dx_new);
            set_mdsyy_dy(x, y, mdsyy_dy_new);
            
            let vy: f32 = get_vy(x, y);    
            set_vy(x, y, dt_over_rho * (vdsigmaxy_dx + vdsigmayy_dy) + vy);
        }}        
    }}

    // Kernel to finish iteration term
    @compute
    @workgroup_size({wsx}, {wsy})
    fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {{
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
        let x_source: i32 = sim_int_par.x_source;
        let y_source: i32 = sim_int_par.y_source;
        let x_sens: i32 = sim_int_par.x_sens;
        let y_sens: i32 = sim_int_par.y_sens;
        let it: i32 = sim_int_par.it;
        let xmin: i32 = sim_int_par.np_pml;
        let xmax: i32 = sim_int_par.x_sz - sim_int_par.np_pml;
        let ymin: i32 = sim_int_par.np_pml;
        let ymax: i32 = sim_int_par.y_sz - sim_int_par.np_pml;
        let rho_2: f32 = sim_flt_par.rho * 0.5;
        let lambda: f32 = sim_flt_par.lambda;
        let mu: f32 = sim_flt_par.mu;
        let lambdaplus2mu: f32 = lambda + 2.0 * mu;
        let denom: f32 = 4.0 * mu * (lambda + mu);
        let mu2: f32 = 2.0 * mu;
        let vx: f32 = get_vx(x, y);
        let vy: f32 = get_vy(x, y);
        
        // Add the source force
        if(x == x_source && y == y_source) {{
            set_vx(x, y, vx + get_force_x(it) * dt_over_rho);
            set_vy(x, y, vy + get_force_y(it) * dt_over_rho);
        }}
        
        // Apply Dirichlet conditions
        if(x <= 1 || x >= (sim_int_par.x_sz - 2) || y <= 1 || y >= (sim_int_par.y_sz - 2)) {{
            set_vx(x, y, 0.0);
            set_vy(x, y, 0.0);
        }}

        // Store sensor velocities
        if(x == x_sens && y == y_sens) {{
            set_sens_vx(it, get_vx(x, y));
            set_sens_vy(it, get_vy(x, y));
        }}
        
        // Compute total energy in the medium (without the PML layers)
        set_vx_vy_2(x, y, get_vx(x, y)*get_vx(x, y) + get_vy(x, y)*get_vy(x, y));
        if(x >= xmin && x < xmax && y >= ymin && y < ymax) {{
            set_tot_en_k(it, rho_2 * get_vx_vy_2(x, y) + get_tot_en_k(it));
            
            set_eps_xx(x, y, (lambdaplus2mu*get_sigmaxx(x, y) - lambda*get_sigmayy(x, y))/denom);
            set_eps_yy(x, y, (lambdaplus2mu*get_sigmayy(x, y) - lambda*get_sigmaxx(x, y))/denom);
            set_eps_xy(x, y, get_sigmaxy(x, y)/mu2);
            
            set_tot_en_p(it, 0.5 * (get_eps_xx(x, y)*get_sigmaxx(x, y) +
                                    get_eps_yy(x, y)*get_sigmayy(x, y) +
                                    2.0 * get_eps_xy(x, y)* get_sigmaxy(x, y)));
            
            set_tot_en(it, get_tot_en_k(it) + get_tot_en_p(it));
        }} 
    }}
    
    // Kernel to increase time iteraction [it]
    @compute
    @workgroup_size(1)
    fn incr_it_kernel() {{
        sim_int_par.it += 1;
    }}
    """


# Simulação completa em WEB GPU
def sim_webgpu():
    global source_term, force_x, force_y
    global a_x, a_x_half, b_x, b_x_half, k_x, k_x_half
    global a_y, a_y_half, b_y, b_y_half, k_y, k_y_half
    global vx, vy, sigmaxx, sigmayy, sigmaxy
    global memory_dvx_dx, memory_dvx_dy
    global memory_dvy_dx, memory_dvy_dy
    global memory_dsigmaxx_dx, memory_dsigmayy_dy
    global memory_dsigmaxy_dx, memory_dsigmaxy_dy
    global value_dvx_dx, value_dvy_dy
    global value_dsigmaxx_dx, value_dsigmaxy_dy
    global value_dsigmaxy_dx, value_dsigmayy_dy
    global sisvx, sisvy
    global total_energy, total_energy_kinetic, total_energy_potential, v_solid_norm
    global epsilon_xx, epsilon_yy, epsilon_xy

    # Arrays com parametros inteiros (i32) e ponto flutuante (f32) para rodar o simulador
    params_i32 = np.array([nx + 2, ny + 2, isource, jsource, ix_rec[0], iy_rec[0], npoints_pml, NSTEP, 0],
                          dtype=np.int32)
    params_f32 = np.array([cp, cs, dx, dy, dt, rho, lambda_, mu, lambdaplus2mu], dtype=flt32)

    # =====================
    # webgpu configurations
    if gpu_type == "NVIDIA":
        device = wgpu.utils.get_default_device()
    else:
        adapter = wgpu.request_adapter(canvas=None, power_preference="low-power")
        device = adapter.request_device()

    # Cria o shader para calculo contido na string ``shader_test''
    # shader_file = open('shader.wgsl', 'r')
    # cshader = device.create_shader_module(code=f"let wsx: i32 = {wsx};\n let wsy: i32 = {wsy};\n" + shader_file.read())
    # shader_file.close()
    cshader = device.create_shader_module(code=shader_test)

    # Definicao dos buffers que terao informacoes compartilhadas entre CPU e GPU
    # ------- Buffers para o binding de parametros -------------
    # Buffer de parametros com valores em ponto flutuante
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 0
    b_param_flt32 = device.create_buffer_with_data(data=params_f32,
                                                   usage=wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_SRC)

    # Forcas da fonte
    # Binding 1
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_force = device.create_buffer_with_data(data=np.column_stack((force_x, force_y)),
                                             usage=wgpu.BufferUsage.STORAGE |
                                                   wgpu.BufferUsage.COPY_SRC)

    # Coeficientes de absorcao
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 2
    b_coef_x = device.create_buffer_with_data(data=np.column_stack((a_x, b_x, k_x, a_x_half, b_x_half, k_x_half)),
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_SRC)

    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 3
    b_coef_y = device.create_buffer_with_data(data=np.row_stack((a_y, b_y, k_y, a_y_half, b_y_half, k_y_half)),
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_SRC)

    # Buffer de parametros com valores inteiros
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 4
    b_param_int32 = device.create_buffer_with_data(data=params_i32,
                                                   usage=wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_SRC)

    # Buffers com os arrays de simulacao
    # Velocidades
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    # Binding 5
    b_vel = device.create_buffer_with_data(data=np.vstack((vx, vy)),
                                           usage=wgpu.BufferUsage.STORAGE |
                                                 wgpu.BufferUsage.COPY_DST |
                                                 wgpu.BufferUsage.COPY_SRC)

    # Estresses
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    # Binding 6
    b_sig = device.create_buffer_with_data(data=np.vstack((sigmaxx, sigmayy, sigmaxy)),
                                           usage=wgpu.BufferUsage.STORAGE |
                                                 wgpu.BufferUsage.COPY_DST |
                                                 wgpu.BufferUsage.COPY_SRC)

    # Arrays de memoria do simulador
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 7
    b_memo = device.create_buffer_with_data(data=np.vstack((memory_dvx_dx, memory_dvx_dy,
                                                            memory_dvy_dx, memory_dvy_dy,
                                                            memory_dsigmaxx_dx, memory_dsigmayy_dy,
                                                            memory_dsigmaxy_dx, memory_dsigmaxy_dy)),
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_SRC)

    # Sinal do sensor
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    # Binding 8
    b_sens = device.create_buffer_with_data(data=np.column_stack((sisvx[:, 0], sisvy[:, 0])),
                                            usage=wgpu.BufferUsage.STORAGE |
                                                  wgpu.BufferUsage.COPY_DST |
                                                  wgpu.BufferUsage.COPY_SRC)

    # Arrays epsilon
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 9
    b_eps = device.create_buffer_with_data(data=np.vstack((epsilon_xx, epsilon_yy, epsilon_xy, vx_vy_2)),
                                           usage=wgpu.BufferUsage.STORAGE |
                                                 wgpu.BufferUsage.COPY_SRC)

    # Arrays de energia total
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    # Binding 10
    b_energy = device.create_buffer_with_data(data=np.column_stack((total_energy, total_energy_kinetic,
                                                                    total_energy_potential, v_solid_norm)),
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_DST |
                                                    wgpu.BufferUsage.COPY_SRC)

    # Esquema de amarracao dos parametros (binding layouts [bl])
    # Parametros
    bl_params = [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         } for ii in range(4)
    ]
    # b_param_i32
    bl_params.append({
        "binding": 4,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    })

    # Arrays da simulacao
    bl_sim_arrays = [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         } for ii in range(5, 8)
    ]

    # Sensores
    bl_sensors = [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         } for ii in range(8, 11)
    ]

    # Configuracao das amarracoes (bindings)
    b_params = [
        {
            "binding": 0,
            "resource": {"buffer": b_param_flt32, "offset": 0, "size": b_param_flt32.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b_force, "offset": 0, "size": b_force.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": b_coef_x, "offset": 0, "size": b_coef_x.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b_coef_y, "offset": 0, "size": b_coef_y.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": b_param_int32, "offset": 0, "size": b_param_int32.size},
        },
    ]
    b_sim_arrays = [
        {
            "binding": 5,
            "resource": {"buffer": b_vel, "offset": 0, "size": b_vel.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": b_sig, "offset": 0, "size": b_sig.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b_memo, "offset": 0, "size": b_memo.size},
        },
    ]
    b_sensors = [
        {
            "binding": 8,
            "resource": {"buffer": b_sens, "offset": 0, "size": b_sens.size},
        },
        {
            "binding": 9,
            "resource": {"buffer": b_eps, "offset": 0, "size": b_eps.size},
        },
        {
            "binding": 10,
            "resource": {"buffer": b_energy, "offset": 0, "size": b_energy.size},
        },
    ]

    # Coloca tudo junto
    bgl_0 = device.create_bind_group_layout(entries=bl_params)
    bgl_1 = device.create_bind_group_layout(entries=bl_sim_arrays)
    bgl_2 = device.create_bind_group_layout(entries=bl_sensors)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_0, bgl_1, bgl_2])
    bg_0 = device.create_bind_group(layout=bgl_0, entries=b_params)
    bg_1 = device.create_bind_group(layout=bgl_1, entries=b_sim_arrays)
    bg_2 = device.create_bind_group(layout=bgl_2, entries=b_sensors)

    # Cria os pipelines de execucao
    compute_sigma_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                          compute={"module": cshader, "entry_point": "sigma_kernel"})
    compute_velocity_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                             compute={"module": cshader,
                                                                      "entry_point": "velocity_kernel"})
    compute_finish_it_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                              compute={"module": cshader,
                                                                       "entry_point": "finish_it_kernel"})
    compute_incr_it_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                            compute={"module": cshader,
                                                                     "entry_point": "incr_it_kernel"})

    # Configuracao e inicializacao da janela de exibicao
    App = pg.QtWidgets.QApplication([])
    windowVx = Window('Vx')
    windowVx.setGeometry(200, 50, vx.shape[0], vx.shape[1])
    windowVy = Window('Vy')
    windowVy.setGeometry(200, 50, vy.shape[0], vy.shape[1])
    # windowVxVy = Window('Vx + Vy')
    # windowVxVy.setGeometry(200, 50, vy.shape[0], vy.shape[1])

    v_min = -1.0
    v_max = - v_min
    # Laco de tempo para execucao da simulacao
    for it in range(NSTEP):
        # Cria o codificador de comandos
        command_encoder = device.create_command_encoder()

        # Inicia os passos de execucao do decodificador
        compute_pass = command_encoder.begin_compute_pass()

        # Ajusta os grupos de amarracao
        compute_pass.set_bind_group(0, bg_0, [], 0, 999999)  # last 2 elements not used
        compute_pass.set_bind_group(1, bg_1, [], 0, 999999)  # last 2 elements not used
        compute_pass.set_bind_group(2, bg_2, [], 0, 999999)  # last 2 elements not used

        # Ativa o pipeline de execucao do calculo dos es stresses
        compute_pass.set_pipeline(compute_sigma_kernel)
        compute_pass.dispatch_workgroups((nx + 2) // wsx, (ny + 2) // wsy)

        # Ativa o pipeline de execucao do calculo das velocidades
        compute_pass.set_pipeline(compute_velocity_kernel)
        compute_pass.dispatch_workgroups((nx + 2) // wsx, (ny + 2) // wsy)

        # Ativa o pipeline de execucao dos procedimentos finais da iteracao
        compute_pass.set_pipeline(compute_finish_it_kernel)
        compute_pass.dispatch_workgroups((nx + 2) // wsx, (ny + 2) // wsy)

        # Ativa o pipeline de atualizacao da amostra de tempo
        compute_pass.set_pipeline(compute_incr_it_kernel)
        compute_pass.dispatch_workgroups(1)

        # Termina o passo de execucao
        compute_pass.end()

        # Efetua a execucao dos comandos na GPU
        device.queue.submit([command_encoder.finish()])

        en = np.asarray(device.queue.read_buffer(b_energy).cast("f")).reshape((NSTEP, 4))
        vsn2 = np.asarray(device.queue.read_buffer(b_eps,
                                                   buffer_offset=(vx_vy_2.size * 4) * 3).cast("f")).reshape((nx + 2,
                                                                                                             ny + 2))
        v_out = np.asarray(device.queue.read_buffer(b_vel).cast("f")).reshape((2*(nx + 2), ny + 2))
        sens_out = np.asarray(device.queue.read_buffer(b_sens).cast("f")).reshape((NSTEP, 2))
        sigma_out = np.asarray(device.queue.read_buffer(b_sig).cast("f")).reshape((3*(nx + 2), ny + 2))
        memo_out = np.asarray(device.queue.read_buffer(b_memo).cast("f")).reshape((8*(nx + 2), ny + 2))
        eps_out = np.asarray(device.queue.read_buffer(b_eps).cast("f")).reshape((4*(nx + 2), ny + 2))
        v_sol_n = np.sqrt(np.max(vsn2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            print(f'Time step # {it} out of {NSTEP}')
            print(f'Max Vx = {np.max(v_out[:nx + 2, :])}, Vy = {np.max(v_out[nx + 2:, :])}')
            print(f'Min Vx = {np.min(v_out[:nx + 2, :])}, Vy = {np.min(v_out[nx + 2:, :])}')
            print(f'Max norm velocity vector V (m/s) = {v_sol_n}')
            print(f'Total energy = {en[it, 2]}')
            windowVx.imv.setImage(v_out[:nx + 2, :], levels=[v_min / 1.0, v_max / 1.0])
            windowVy.imv.setImage(v_out[nx + 2:, :], levels=[v_min / 1.0, v_max / 1.0])
            App.processEvents()

        # Verifica a estabilidade da simulacao
        if en[it, 3] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instável")
            exit(2)

    App.exit()

    # Pega o sinal do sensor
    sens = np.array(device.queue.read_buffer(b_sens).cast("f")).reshape((NSTEP, 2))
    adapter_info = device.adapter.request_adapter_info()
    return sens, adapter_info["device"]


times_webgpu = list()
times_cpu = list()

# WebGPU
if do_sim_gpu:
    for n in range(n_iter_gpu):
        print(f'Simulacao WEBGPU')
        print(f'Iteracao {n}')
        t_webgpu = time()
        sensor, gpu_str = sim_webgpu()
        times_webgpu.append(time() - t_webgpu)
        print(gpu_str)
        print(f'{times_webgpu[-1]:.3}s')

# CPU
if do_sim_cpu:
    for n in range(n_iter_cpu):
        print(f'SIMULAÇÃO CPU')
        print(f'Iteracao {n}')
        t_ser = time()
        sim_cpu()
        times_cpu.append(time() - t_ser)
        print(f'{times_cpu[-1]:.3}s')

        # Plota as velocidades tomadas no sensores
        for irec in range(NREC):
            fig, ax = plt.subplots(3, sharex=True, sharey=True)
            fig.suptitle(f'Receptor {irec + 1}')
            ax[0].plot(sisvx[:, irec])
            ax[0].set_title(r'$V_x$')
            ax[1].plot(sisvy[:, irec])
            ax[1].set_title(r'$V_y$')
            ax[2].plot(sisvx[:, irec] + sisvy[:, irec], 'tab:orange')
            ax[2].set_title(r'$V_x + V_y$')

        plt.show()

# times_for = np.array(times_for)
# times_ser = np.array(times_ser)
# if do_sim_gpu:
#     print(f'workgroups X: {wsx}; workgroups Y: {wsy}')
#
# print(f'TEMPO - {nt} pontos de tempo')
# if do_sim_gpu and n_iter_gpu > 5:
#     print(f'GPU: {times_for[5:].mean():.3}s (std = {times_for[5:].std()})')
#
# if do_sim_cpu and n_iter_cpu > 5:
#     print(f'CPU: {times_ser[5:].mean():.3}s (std = {times_ser[5:].std()})')
#
# if do_sim_gpu and do_sim_cpu:
#     print(f'MSE entre as simulações: {mean_squared_error(u_ser, u_for)}')
#
# if plot_results:
#     if do_sim_gpu:
#         gpu_sim_result = plt.figure()
#         plt.title(f'GPU simulation ({nz}x{nx})')
#         plt.imshow(u_for, aspect='auto', cmap='turbo_r')
#
#         sensor_gpu_result = plt.figure()
#         plt.title(f'Sensor at z = {sens_z} and x = {sens_x}')
#         plt.plot(t, sensor)
#
#     if do_sim_cpu:
#         cpu_sim_result = plt.figure()
#         plt.title(f'CPU simulation ({nz}x{nx})')
#         plt.imshow(u_ser, aspect='auto', cmap='turbo_r')
#
#     if do_comp_fig_cpu_gpu and do_sim_cpu and do_sim_gpu:
#         comp_sim_result = plt.figure()
#         plt.title(f'CPU vs GPU ({gpu_type}) error simulation ({nz}x{nx})')
#         plt.imshow(u_ser - u_for, aspect='auto', cmap='turbo_r')
#         plt.colorbar()
#
#     if show_results:
#         plt.show()
#
# if save_results:
#     now = datetime.now()
#     name = f'results/result_{now.strftime("%Y%m%d-%H%M%S")}_{nz}x{nx}_{nt}_iter'
#     if plot_results:
#         if do_sim_gpu:
#             gpu_sim_result.savefig(name + '_gpu_' + gpu_type + '.png')
#             sensor_gpu_result.savefig(name + '_sensor_' + gpu_type + '.png')
#
#         if do_sim_cpu:
#             cpu_sim_result.savefig(name + 'cpu.png')
#
#         if do_comp_fig_cpu_gpu and do_sim_cpu and do_sim_gpu:
#             comp_sim_result.savefig(name + 'comp_cpu_gpu_' + gpu_type + '.png')
#
#     np.savetxt(name + '_GPU_' + gpu_type + '.csv', times_for, '%10.3f', delimiter=',')
#     np.savetxt(name + '_CPU.csv', times_ser, '%10.3f', delimiter=',')
#     with open(name + '_desc.txt', 'w') as f:
#         f.write('Parametros do ensaio\n')
#         f.write('--------------------\n')
#         f.write('\n')
#         f.write(f'Quantidade de iteracoes no tempo: {nt}\n')
#         f.write(f'Tamanho da ROI: {nz}x{nx}\n')
#         f.write(f'Refletores na ROI: {"Sim" if use_refletors else "Nao"}\n')
#         f.write(f'Simulacao GPU: {"Sim" if do_sim_gpu else "Nao"}\n')
#         if do_sim_gpu:
#             f.write(f'GPU: {gpu_str}\n')
#             f.write(f'Numero de simulacoes GPU: {n_iter_gpu}\n')
#             if n_iter_gpu > 5:
#                 f.write(f'Tempo medio de execucao: {times_for[5:].mean():.3}s\n')
#                 f.write(f'Desvio padrao: {times_for[5:].std()}\n')
#             else:
#                 f.write(f'Tempo execucao: {times_for[0]:.3}s\n')
#
#         f.write(f'Simulacao CPU: {"Sim" if do_sim_cpu else "Nao"}\n')
#         if do_sim_cpu:
#             f.write(f'Numero de simulacoes CPU: {n_iter_cpu}\n')
#             if n_iter_cpu > 5:
#                 f.write(f'Tempo medio de execucao: {times_ser[5:].mean():.3}s\n')
#                 f.write(f'Desvio padrao: {times_ser[5:].std()}\n')
#             else:
#                 f.write(f'Tempo execucao: {times_ser[0]:.3}s\n')
#
#         if do_sim_gpu and do_sim_cpu:
#             f.write(f'MSE entre as simulacoes: {mean_squared_error(u_ser, u_for)}')
