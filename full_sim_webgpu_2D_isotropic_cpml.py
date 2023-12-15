import wgpu
if wgpu.version_info[1] > 11:
    import wgpu.backends.wgpu_native  # Select backend 0.13.X
else:
    import wgpu.backends.rs  # Select backend 0.9.5

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time
# from datetime import datetime
from PyQt6.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget


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


# Window class
class Window(QMainWindow):
    def __init__(self, title=None, geometry=None):
        super().__init__()

        # setting title
        if title is None:
            self.setWindowTitle(f"{ny}x{nx} Grid x {NSTEP} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
        else:
            self.setWindowTitle(title)

        # setting geometry
        if geometry is None:
            self.setGeometry(200, 50, 300, 300)
        else:
            self.setGeometry(*geometry)

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
show_anim = True
show_debug = False
plot_results = True
show_results = True
save_results = True
gpu_type = "NVIDIA"

device_gpu = None
if do_sim_gpu:
    # =====================
    # webgpu configurations
    if gpu_type == "NVIDIA":
        device_gpu = wgpu.utils.get_default_device()
    else:
        if wgpu.version_info[1] > 11:
            adapter = wgpu.gpu.request_adapter(power_preference="low-power")  # 0.13.X
        else:
            adapter = wgpu.request_adapter(canvas=None, power_preference="low-power")  # 0.9.5

        device_gpu = adapter.request_device()

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
NSTEP = 500

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
xrec = xsource + np.array([-10, 0]) * dx
yrec = ysource + np.array([10, 30]) * dy
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
            distval = np.sqrt((dx * i - xrec[irec]) ** 2 + (dx * j - yrec[irec]) ** 2)
            if distval < dist:
                dist = distval
                ix_rec[irec] = i
                iy_rec[irec] = j

    print(f'receiver {irec}: x_target = {xrec[irec]}, y_target = {yrec[irec]}')
    print(f'Ponto mais perto do grid encontrado na distancia {dist}, em i = {ix_rec[irec]}, j = {iy_rec[irec]}\n')

# Verifica a condicao de estabilidade de Courant
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
courant_number = cp * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2)
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
    global windows_cpu

    DELTAT_over_rho = dt / rho
    denom = np.float32(4.0 * mu * (lambda_ + mu))

    v_min = -1.0
    v_max = - v_min
    # Inicio do laco de tempo
    for it in range(1, NSTEP + 1):
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
            if show_debug:
                print(f'Time step # {it} out of {NSTEP}')
                print(f'Max Vx = {np.max(vx)}, Vy = {np.max(vy)}')
                print(f'Min Vx = {np.min(vx)}, Vy = {np.min(vy)}')
                print(f'Max norm velocity vector V (m/s) = {v_solid_norm[it - 1]}')
                print(f'Total energy = {total_energy[it - 1]}')

            if show_anim:
                windows_cpu[0].imv.setImage(vx[:, :], levels=[v_min / 1.0, v_max / 1.0])
                windows_cpu[1].imv.setImage(vy[:, :], levels=[v_min / 1.0, v_max / 1.0])
                windows_cpu[2].imv.setImage(vx[:, :] + vy[:, :], levels=[2.0 * v_min / 1.0, 2.0 * v_max / 1.0])
                App.processEvents()

        # Verifica a estabilidade da simulacao
        if v_solid_norm[it - 1] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instável")
            exit(2)


# Simulação completa em WEB GPU
def sim_webgpu(device):
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
    global windows_gpu

    # Arrays com parametros inteiros (i32) e ponto flutuante (f32) para rodar o simulador
    params_i32 = np.array([nx + 2, ny + 2, isource, jsource, ix_rec[0], iy_rec[0], npoints_pml, NSTEP, 0],
                          dtype=np.int32)
    params_f32 = np.array([cp, cs, dx, dy, dt, rho, lambda_, mu, lambdaplus2mu], dtype=flt32)

    # Cria o shader para calculo contido no arquivo ``shader_2D_elast_cpml.wgsl''
    cshader = None
    with open('shader_2D_elast_cpml.wgsl') as shader_file:
        cshader_string = shader_file.read().replace('wsx', f'{wsx}').replace('wsy', f'{wsy}')
        cshader = device.create_shader_module(code=cshader_string)

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

    v_min = -1.0
    v_max = - v_min
    # Laco de tempo para execucao da simulacao
    times_submit = list()
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
        v_sol_n = np.sqrt(np.max(vsn2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            if show_debug or show_anim:
                t_submit = time()
                v_out = np.asarray(device.queue.read_buffer(b_vel).cast("f")).reshape((2 * (nx + 2), ny + 2))
                times_submit.append(time() - t_submit)

            if show_debug:
                print(f'Time step # {it} out of {NSTEP}')
                print(f'Max Vx = {np.max(v_out[:nx + 2, :])}, Vy = {np.max(v_out[nx + 2:, :])}')
                print(f'Min Vx = {np.min(v_out[:nx + 2, :])}, Vy = {np.min(v_out[nx + 2:, :])}')
                print(f'Max norm velocity vector V (m/s) = {v_sol_n}')
                print(f'Total energy = {en[it, 2]}')

            if show_anim:
                windows_gpu[0].imv.setImage(v_out[:nx + 2, :], levels=[v_min / 1.0, v_max / 1.0])
                windows_gpu[1].imv.setImage(v_out[nx + 2:, :], levels=[v_min / 1.0, v_max / 1.0])
                windows_gpu[2].imv.setImage(v_out[:nx + 2, :] + v_out[nx + 2:, :], levels=[2.0 * v_min / 1.0,
                                                                                           2.0 * v_max / 1.0])
                App.processEvents()

        # Verifica a estabilidade da simulacao
        if en[it, 3] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instável")
            exit(2)

    # Pega os resultados da simulacao
    v_out = np.asarray(device.queue.read_buffer(b_vel).cast("f")).reshape((2 * (nx + 2), ny + 2))
    sens = np.array(device.queue.read_buffer(b_sens).cast("f")).reshape((NSTEP, 2))
    adapter_info = device.adapter.request_adapter_info()
    return v_out[:nx + 2, :], v_out[nx + 2:, :], sens, adapter_info["device"]


times_webgpu = list()
times_cpu = list()

# Configuracao e inicializacao da janela de exibicao
if show_anim:
    App = pg.QtWidgets.QApplication([])
    if do_sim_cpu:
        x_pos = 200 + np.arange(3) * (nx + 10)
        y_pos = 400
        windows_cpu_data = [
            {"title": "Vx [CPU]", "geometry": (x_pos[0], y_pos, vx.shape[0], vx.shape[1])},
            {"title": "Vy [CPU]", "geometry": (x_pos[1], y_pos, vy.shape[0], vy.shape[1])},
            {"title": "Vx + Vy [CPU]", "geometry": (x_pos[2], y_pos, vy.shape[0], vy.shape[1])},
        ]
        windows_cpu = [Window(title=data["title"], geometry=data["geometry"]) for data in windows_cpu_data]

    if do_sim_gpu:
        x_pos = 200 + np.arange(3) * (nx + 10)
        y_pos = 50
        windows_gpu_data = [
            {"title": "Vx [GPU]", "geometry": (x_pos[0], y_pos, vx.shape[0], vx.shape[1])},
            {"title": "Vy [GPU]", "geometry": (x_pos[1], y_pos, vy.shape[0], vy.shape[1])},
            {"title": "Vx + Vy [GPU]", "geometry": (x_pos[2], y_pos, vy.shape[0], vy.shape[1])},
        ]
        windows_gpu = [Window(title=data["title"], geometry=data["geometry"]) for data in windows_gpu_data]

# WebGPU
if do_sim_gpu:
    for n in range(n_iter_gpu):
        print(f'Simulacao WEBGPU')
        print(f'Iteracao {n}')
        t_webgpu = time()
        vx_gpu, vy_gpu, sensor, gpu_str = sim_webgpu(device_gpu)
        times_webgpu.append(time() - t_webgpu)
        print(gpu_str)
        print(f'{times_webgpu[-1]:.3}s')

        # Plota as velocidades tomadas no sensores
        if plot_results:
            fig, ax = plt.subplots(3, sharex=True, sharey=True)
            fig.suptitle(f'Receptor 1 [GPU]')
            ax[0].plot(sensor[:, 0])
            ax[0].set_title(r'$V_x$')
            ax[1].plot(sensor[:, 1])
            ax[1].set_title(r'$V_y$')
            ax[2].plot(sensor[:, 0] + sensor[:, 1], 'tab:orange')
            ax[2].set_title(r'$V_x + V_y$')

# CPU
if do_sim_cpu:
    for n in range(n_iter_cpu):
        print(f'SIMULAÇÃO CPU')
        print(f'Iteracao {n}')
        t_cpu = time()
        sim_cpu()
        times_cpu.append(time() - t_cpu)
        print(f'{times_cpu[-1]:.3}s')

        # Plota as velocidades tomadas no sensores
        if plot_results:
            for irec in range(NREC):
                fig, ax = plt.subplots(3, sharex=True, sharey=True)
                fig.suptitle(f'Receptor {irec + 1} [CPU]')
                ax[0].plot(sisvx[:, irec])
                ax[0].set_title(r'$V_x$')
                ax[1].plot(sisvy[:, irec])
                ax[1].set_title(r'$V_y$')
                ax[2].plot(sisvx[:, irec] + sisvy[:, irec], 'tab:orange')
                ax[2].set_title(r'$V_x + V_y$')

            plt.show()

if show_anim and App:
    App.exit()

times_webgpu = np.array(times_webgpu)
times_cpu = np.array(times_cpu)
if do_sim_gpu:
    print(f'workgroups X: {wsx}; workgroups Y: {wsy}')

print(f'TEMPO - {NSTEP} pontos de tempo')
if do_sim_gpu and n_iter_gpu > 5:
    print(f'GPU: {times_webgpu[5:].mean():.3}s (std = {times_webgpu[5:].std()})')

if do_sim_cpu and n_iter_cpu > 5:
    print(f'CPU: {times_cpu[5:].mean():.3}s (std = {times_cpu[5:].std()})')

if do_sim_gpu and do_sim_cpu:
    print(f'MSE entre as simulações [Vx]: {mean_squared_error(vx_gpu, vx)}')
    print(f'MSE entre as simulações [Vy]: {mean_squared_error(vy_gpu, vy)}')

if plot_results:
    if do_sim_gpu:
        vx_gpu_sim_result = plt.figure()
        plt.title(f'GPU simulation Vx ({nx}x{ny})')
        plt.imshow(vx_gpu, aspect='auto', cmap='turbo_r')

        vy_gpu_sim_result = plt.figure()
        plt.title(f'GPU simulation Vy({nx}x{ny})')
        plt.imshow(vy_gpu, aspect='auto', cmap='turbo_r')

    if do_sim_cpu:
        vx_cpu_sim_result = plt.figure()
        plt.title(f'CPU simulation Vx ({nx}x{ny})')
        plt.imshow(vx, aspect='auto', cmap='turbo_r')

        vy_cpu_sim_result = plt.figure()
        plt.title(f'CPU simulation Vy ({nx}x{ny})')
        plt.imshow(vy, aspect='auto', cmap='turbo_r')

    if do_comp_fig_cpu_gpu and do_sim_cpu and do_sim_gpu:
        vx_comp_sim_result = plt.figure()
        plt.title(f'CPU vs GPU Vx ({gpu_type}) error simulation ({nx}x{ny})')
        plt.imshow(vx - vx_gpu, aspect='auto', cmap='turbo_r')
        plt.colorbar()

        vy_comp_sim_result = plt.figure()
        plt.title(f'CPU vs GPU Vy ({gpu_type}) error simulation ({nx}x{ny})')
        plt.imshow(vy - vy_gpu, aspect='auto', cmap='turbo_r')
        plt.colorbar()

    if show_results:
        plt.show()

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
