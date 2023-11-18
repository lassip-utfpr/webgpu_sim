import math
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


# TODO: portar essa funcao
def compute_attenuation_coeffs(is_kappa):  #  n, q_kappa, f0, f_min,f_max):
    if is_kappa:
        return (np.array([3.4331474384407847E-002, 3.6311125270723529E-003], dtype=flt32),
                np.array([2.9287653312114702E-002, 3.0503144159812171E-003], dtype=flt32))

    return (np.array([3.7739400980721378E-002, 4.1548430957513323E-003], dtype=flt32),
            np.array([2.7848924623855534E-002, 2.8973181158942259E-003], dtype=flt32))


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
nx = 70  # colunas
ny = 70  # linhas
nz = 24   # altura

# Tamanho do grid (aparentemente em metros)
dx = 4.0
dz = dy = dx
one_dx = one_dy = one_dz = 1.0/dx

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
USE_PML_ZMIN = True
USE_PML_ZMAX = True

# Espessura da PML in pixels
npoints_pml = 10

# Velocidades do som e densidade do meio
cp = 3000.0  # [m/s]
cs = 2000.0  # [m/s]
rho = 2000.0
mu = rho * cs * cs
lambda_ = rho * (cp * cp - 2.0 * cs * cs)
lambdaplus2mu = rho * cp * cp

# Numero total de passos de tempo
NSTEP = 1000

# Passo de tempo em segundos
dt = 4.0e-4

# Numero de iteracoes de tempo para apresentar e armazenar informacoes
IT_DISPLAY = 10

# Parametros da fonte
f0 = 18.0  # frequencia
t0 = 1.20 / f0  # delay
factor = 1.0e7
a = PI**2 * f0**2
t = np.arange(NSTEP) * dt
ANGLE_FORCE = 0.0

# First derivative of a Gaussian
source_term = -(factor * 2.0 * a * (t - t0) * np.exp(-a * (t - t0)**2)).astype(flt32)

# Funcao de Ricker (segunda derivada de uma gaussiana)
# source_term = (factor * (1.0 - 2.0 * a * (t - t0) ** 2) * np.exp(-a * (t - t0) ** 2)).astype(flt32)

force_x = np.sin(ANGLE_FORCE * DEGREES_TO_RADIANS) * source_term
force_y = np.cos(ANGLE_FORCE * DEGREES_TO_RADIANS) * source_term

# Parametro de atenuacao
N_SLS = 2

# Qp approximately equal to 13, Qkappa approximately to 20 and Qmu / Qs approximately to 10
q_kappa_att = 20.0
q_mu_att = 10.0
f0_attenuation = 16  # in Hz

# Posicao da fonte
isource = int(nx / 4)
jsource = int(ny / 2)
ksource = int(nz / 2)
xsource = isource * dx
ysource = jsource * dy
zsource = ksource * dz

# Receptores
NREC = 3
# xdeb = xsource - 100.0  # em unidade de distancia
# ydeb = 2300.0  # em unidade de distancia
# xfin = xsource
# yfin = 300.0
xrec = xsource + np.array([40.0, 0.0, 40.0])
yrec = ysource + np.array([40.0, 100.0, 100.0])
# sens_x = int(xdeb / dx) + 1
# sens_y = int(ydeb / dy) + 1
# sensor = np.zeros(NSTEP, dtype=flt32)  # buffer para sinal do sensor
sisvx = np.zeros((NSTEP, NREC), dtype=flt32)
sisvy = np.zeros((NSTEP, NREC), dtype=flt32)

# for evolution of total energy in the medium
epsilon_xx = epsilon_yy = epsilon_zz = epsilon_xy = epsilon_xz = epsilon_yz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
total_energy = np.zeros(NSTEP, dtype=flt32)
total_energy_kinetic = np.zeros(NSTEP, dtype=flt32)
total_energy_potential = np.zeros(NSTEP, dtype=flt32)

# Valor da potencia para calcular "d0"
NPOWER = 2.0

# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = 7.0
ALPHA_MAX_PML = 2.0 * PI * (f0 / 2.0)  # from Festa and Vilotte

# Escolha do valor de wsx (GPU)
wsx = 1
for n in range(15, 0, -1):
    if (ny % n) == 0:
        wsx = n  # workgroup x size
        break

# Escolha do valor de wsy (GPU)
wsy = 1
for n in range(15, 0, -1):
    if (nx % n) == 0:
        wsy = n  # workgroup x size
        break

# TODO: Escolha do valor de wsz (GPU) --- Verificar necessidade
wsz = 1
for n in range(15, 0, -1):
    if (nz % n) == 0:
        wsz = n  # workgroup x size
        break

# Arrays para as variaveis de memoria do calculo
memory_dvx_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvx_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvx_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvy_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvy_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvy_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvz_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvz_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dvz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmaxx_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmayy_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmazz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmaxy_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmaxy_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmaxz_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmaxz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmayz_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
memory_dsigmayz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)

vx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
vy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
vz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmaxx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmayy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmazz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmaxy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmaxz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmayz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmaxx_r = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmayy_r = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmazz_r = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmaxy_r = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmaxz_r = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
sigmayz_r = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)

e1 = np.zeros((N_SLS, nx + 2, ny + 2, nz + 2), dtype=flt32)
e11 = np.zeros((N_SLS, nx + 2, ny + 2, nz + 2), dtype=flt32)
e22 = np.zeros((N_SLS, nx + 2, ny + 2, nz + 2), dtype=flt32)
e12 = np.zeros((N_SLS, nx + 2, ny + 2, nz + 2), dtype=flt32)
e13 = np.zeros((N_SLS, nx + 2, ny + 2, nz + 2), dtype=flt32)
e23 = np.zeros((N_SLS, nx + 2, ny + 2, nz + 2), dtype=flt32)

# Total de arrays
N_ARRAYS = 9 + 2*9 + 12

value_dvx_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvx_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvx_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvy_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvy_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvy_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvz_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvz_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dvz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmaxx_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmayy_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmazz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmaxy_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmaxy_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmaxz_dx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmaxz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmayz_dy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
value_dsigmayz_dz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)

duxdx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duxdy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duxdz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duydx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duydy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duydz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duzdx = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duzdy = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
duzdz = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)
div = np.zeros((nx + 2, ny + 2, nz + 2), dtype=flt32)

# Inicializacao dos parametros da PML (definicao dos perfis de absorcao na regiao da PML)
thickness_pml_x = npoints_pml * dx
thickness_pml_y = npoints_pml * dy
thickness_pml_z = npoints_pml * dz

# Coeficiente de reflexao (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
rcoef = 0.0001

# Calculo da faixa de atenuacao em frequencia: f_max/f_min=12 and (log(f_min)+log(f_max))/2 = log(f0)
f_min_attenuation = np.exp(np.log(f0_attenuation) - np.log(12.0)/2.0)
f_max_attenuation = 12.0 * f_min_attenuation

# use new SolvOpt nonlinear optimization with constraints from Emilie Blanc, Bruno Lombard and Dimitri Komatitsch
# to compute attenuation mechanisms
# tau_epsilon_nu1, tau_sigma_nu1 = compute_attenuation_coeffs(N_SLS, q_kappa_att, f0_attenuation,
#                                                             f_min_attenuation,f_max_attenuation)
# tau_epsilon_nu2, tau_sigma_nu2 = compute_attenuation_coeffs(N_SLS, q_mu_att, f0_attenuation,
#                                                             f_min_attenuation, f_max_attenuation)
tau_epsilon_nu1, tau_sigma_nu1 = compute_attenuation_coeffs(is_kappa=True)
tau_epsilon_nu2, tau_sigma_nu2 = compute_attenuation_coeffs(is_kappa=False)

print(f'\nAtenuacao calculada pela nova rotina SolvOpt:')
print(f'NSLs = {N_SLS}, QKappa_att = {q_kappa_att}, QMu_att = {q_mu_att}')
print(f'f0_attenuation = {f0_attenuation}, f_min_attenuation = {f_min_attenuation}, '
      f'f_max_attenuation = {f_max_attenuation}')
print(f'tau_epsilon_nu1 = {tau_epsilon_nu1}')
print(f'tau_sigma_nu1 = {tau_sigma_nu1}')
print(f'tau_epsilon_nu2 = {tau_epsilon_nu2}')
print(f'tau_sigma_nu2 = {tau_sigma_nu2}\n')

tau1 = tau_sigma_nu1[0]/tau_epsilon_nu1[0]
tau2 = tau_sigma_nu2[0]/tau_epsilon_nu2[0]
tau3 = tau_sigma_nu1[1]/tau_epsilon_nu1[1]
tau4 = tau_sigma_nu2[1]/tau_epsilon_nu2[1]

taumax = np.max([1.0/tau1, 1.0/tau2, 1.0/tau3, 1.0/tau4])
taumin = np.min([1.0/tau1, 1.0/tau2, 1.0/tau3, 1.0/tau4])

inv_tau_sigma_nu1 = 1.0/tau_sigma_nu1
inv_tau_sigma_nu2 = 1.0/tau_sigma_nu2

phi_nu1 = (1.0 - tau_epsilon_nu1 / tau_sigma_nu1) / tau_sigma_nu1
phi_nu2 = (1.0 - tau_epsilon_nu2 / tau_sigma_nu2) / tau_sigma_nu2

Mu_nu1 = 1.0 - (1.0 - tau_epsilon_nu1[0] / tau_sigma_nu1[0]) - (1.0 - tau_epsilon_nu1[1] / tau_sigma_nu1[1])
Mu_nu2 = 1.0 - (1.0 - tau_epsilon_nu2[0] / tau_sigma_nu2[0]) - (1.0 - tau_epsilon_nu2[1] / tau_sigma_nu2[1])

print(f'3D elastic finite-difference code in velocity and stress formulation with C-PML')
print(f'NX = {nx}')
print(f'NY = {ny}')
print(f'NZ = {nz}\n')
print(f'Tamanho do modelo ao longo do eixo X = {(nx + 1) * dx}')
print(f'Tamanho do modelo ao longo do eixo Y = {(ny + 1) * dy}')
print(f'Tamanho do modelo ao longo do eixo Z = {(nz + 1) * dz}')
print(f'Total de pontos no grid = {(nx+2) * (ny+2) * (nz+2)}')
print(f'Number of points of all the arrays = {(nx+2)*(ny+2)*(nz+2)*N_ARRAYS}')
print(f'Size in GB of all the arrays = {(nx+2)*(ny+2)*(nz+2)*N_ARRAYS*4/(1024*1024*1024)}\n')

if NPOWER < 1:
    raise ValueError('NPOWER deve ser maior que 1')

# Calculo de d0 do relatorio da INRIA section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
d0_x = -(NPOWER + 1) * cp * math.sqrt(taumax) * math.log(rcoef) / (2.0 * thickness_pml_x)
d0_y = -(NPOWER + 1) * cp * math.sqrt(taumax) * math.log(rcoef) / (2.0 * thickness_pml_y)
d0_z = -(NPOWER + 1) * cp * math.sqrt(taumax) * math.log(rcoef) / (2.0 * thickness_pml_z)

print(f'd0_x = {d0_x}')
print(f'd0_y = {d0_y}')
print(f'd0_z = {d0_z}\n')

# Amortecimento na direcao "x" (horizontal)
# Origem da PML (posicao das bordas direita e esquerda menos a espessura, em unidades de distancia)
x_orig_left = thickness_pml_x
x_orig_right = (nx - 1) * dx - thickness_pml_x

# Perfil de amortecimento na direcao "x" dentro do grid
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
d_x = np.expand_dims((d0_x * x_norm ** NPOWER).astype(flt32), axis=(1, 2))
k_x = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * x_norm ** NPOWER).astype(flt32), axis=(1, 2))
alpha_x = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(x_mask, x_norm, 1.0))).astype(flt32), axis=(1, 2))
b_x = np.exp(-(d_x / k_x + alpha_x) * dt).astype(flt32)
i = np.where(d_x > 1e-6)
a_x = np.zeros((nx, 1, 1), dtype=flt32)
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
d_x_half = np.expand_dims((d0_x * x_norm ** NPOWER).astype(flt32), axis=(1, 2))
k_x_half = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * x_norm ** NPOWER).astype(flt32), axis=(1, 2))
alpha_x_half = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(x_mask_half, x_norm, 1.0))).astype(flt32), axis=(1, 2))
b_x_half = np.exp(-(d_x_half / k_x_half + alpha_x_half) * dt).astype(flt32)
i = np.where(d_x_half > 1e-6)
a_x_half = np.zeros((nx, 1, 1), dtype=flt32)
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
d_y = np.expand_dims((d0_y * y_norm ** NPOWER).astype(flt32), axis=(0, 2))
k_y = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * y_norm ** NPOWER).astype(flt32), axis=(0, 2))
alpha_y = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(y_mask, y_norm, 1.0))).astype(flt32), axis=(0, 2))
b_y = np.exp(-(d_y / k_y + alpha_y) * dt).astype(flt32)
j = np.where(d_y > 1e-6)
a_y = np.zeros((1, ny, 1), dtype=flt32)
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
d_y_half = np.expand_dims((d0_y * y_norm ** NPOWER).astype(flt32), axis=(0, 2))
k_y_half = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * y_norm ** NPOWER).astype(flt32), axis=(0, 2))
alpha_y_half = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(y_mask_half, y_norm, 1.0))).astype(flt32), axis=(0, 2))
b_y_half = np.exp(-(d_y_half / k_y_half + alpha_y_half) * dt).astype(flt32)
j = np.where(d_y_half > 1e-6)
a_y_half = np.zeros((1, ny, 1), dtype=flt32)
a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (k_y_half[j] * (d_y_half[j] + k_y_half[j] * alpha_y_half[j]))

# Amortecimento na direcao "z" (profundidade)
# Origem da PML (posicao das bordas frontal e fundos menos a espessura, em unidades de distancia)
z_orig_front = thickness_pml_z
z_orig_back = (nz - 1) * dz - thickness_pml_z

# Perfil de amortecimento na direcao "z" dentro do grid
k = np.arange(nz)
zval = dz * k
z_pml_front = z_orig_front - zval
z_pml_back = zval - z_orig_back
z_pml_mask_front = np.where(z_pml_front < 0.0, False, True)
z_pml_mask_back = np.where(z_pml_back < 0.0, False, True)
z_mask = np.logical_or(z_pml_mask_front, z_pml_mask_back)
z_pml = np.zeros(nz)
z_pml[z_pml_mask_front] = z_pml_front[z_pml_mask_front]
z_pml[z_pml_mask_back] = z_pml_back[z_pml_mask_back]
z_norm = z_pml / thickness_pml_z
d_z = np.expand_dims((d0_z * z_norm ** NPOWER).astype(flt32), axis=(0, 1))
k_z = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * z_norm ** NPOWER).astype(flt32), axis=(0, 1))
alpha_z = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(z_mask, z_norm, 1.0))).astype(flt32), axis=(0, 1))
b_z = np.exp(-(d_z / k_z + alpha_z) * dt).astype(flt32)
k = np.where(d_z > 1e-6)
a_z = np.zeros((1, 1, nz), dtype=flt32)
a_z[k] = d_z[k] * (b_z[k] - 1.0) / (k_z[k] * (d_z[k] + k_z[k] * alpha_z[k]))

# Perfil de amortecimento na direcao "z" dentro do meio grid (staggered grid)
z_pml_front = z_orig_front - (zval + dz / 2.0)
z_pml_back = (zval + dz / 2.0) - z_orig_back
z_pml_mask_front = np.where(z_pml_front < 0.0, False, True)
z_pml_mask_back = np.where(z_pml_back < 0.0, False, True)
z_mask_half = np.logical_or(z_pml_mask_front, z_pml_mask_back)
z_pml = np.zeros(nz)
z_pml[z_pml_mask_front] = z_pml_front[z_pml_mask_front]
z_pml[z_pml_mask_back] = z_pml_back[z_pml_mask_back]
z_norm = z_pml / thickness_pml_z
d_z_half = np.expand_dims((d0_z * z_norm ** NPOWER).astype(flt32), axis=(0, 1))
k_z_half = np.expand_dims((1.0 + (K_MAX_PML - 1.0) * z_norm ** NPOWER).astype(flt32), axis=(0, 1))
alpha_z_half = np.expand_dims((ALPHA_MAX_PML * (1.0 - np.where(z_mask_half, z_norm, 1.0))).astype(flt32), axis=(0, 1))
b_z_half = np.exp(-(d_z_half / k_z_half + alpha_z_half) * dt).astype(flt32)
k = np.where(d_z_half > 1e-6)
a_z_half = np.zeros((1, 1, nz), dtype=flt32)
a_z_half[k] = d_z_half[k] * (b_z_half[k] - 1.0) / (k_z_half[k] * (d_z_half[k] + k_z_half[k] * alpha_z_half[k]))

# Imprime a posicao da fonte
print(f'Posicao da fonte:')
print(f'x = {xsource}')
print(f'y = {ysource}')
print(f'z = {zsource}\n')

# Define a localizacao dos receptores
print(f'Existem {NREC} receptores')

# Find closest grid point for each receiver
ix_rec = np.zeros(NREC, dtype=int)
iy_rec = np.zeros(NREC, dtype=int)
for irec in range(NREC):
    dist = HUGEVAL
    for j in range(ny):
        for i in range(nx):
            distval = math.sqrt((dx * i - xrec[irec])**2 + (dx * j - yrec[irec])**2)
            if distval < dist:
                dist = distval
                ix_rec[irec] = i
                iy_rec[irec] = j

    print(f'receiver {irec}: x_target = {xrec[irec]}, y_target = {yrec[irec]}')
    print(f'Ponto mais perto do grid encontrado na distancia {dist}, em i = {ix_rec[irec]}, j = {iy_rec[irec]}\n')

# Verifica a condicao de estabilidade de Courant
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
courant_number = cp * math.sqrt(taumax) * dt * math.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2 + 1.0 / dz ** 2)
print(f'Numero de Courant e {courant_number}')
print(f'Vpmax = {cp * math.sqrt(taumax)}')
if courant_number > 1:
    print("O passo de tempo e muito longo, a simulação sera instavel")
    exit(1)
print(f'Number of points per wavelength = {cs * math.sqrt(taumin)/(2.5 * f0)/dx}, Vsmin = {cs * math.sqrt(taumin)}')


def sim_cpu():
    global vx, vy, vz, sigmaxx, sigmayy, sigmazz, sigmaxx_r, sigmayy_r, sigmazz_r
    global sigmaxy, sigmaxz, sigmayz, sigmaxy_r, sigmaxz_r, sigmayz_r
    global e1, e11, e12, e13, e23, e22
    global memory_dvx_dx, memory_dvx_dy, memory_dvx_dz
    global memory_dvy_dx, memory_dvy_dy, memory_dvy_dz
    global memory_dvz_dx, memory_dvz_dy, memory_dvz_dz
    global memory_dsigmaxx_dx, memory_dsigmayy_dy, memory_dsigmazz_dz
    global memory_dsigmaxy_dx, memory_dsigmaxy_dy
    global memory_dsigmaxz_dx, memory_dsigmaxz_dz
    global memory_dsigmayz_dy, memory_dsigmayz_dz
    global sisvx, sisvy
    global total_energy, total_energy_kinetic, total_energy_potential
    global value_dvx_dx, value_dvy_dy, value_dvz_dz
    global a_x, a_x_half, b_x, b_x_half, k_x, k_x_half
    global a_y, a_y_half, b_y, b_y_half, k_y, k_y_half
    global a_z, a_z_half, b_z, b_z_half, k_z, k_z_half
    global duxdx, duxdy, duxdz, duydx, duydy, duydz, duzdx, duzdy, duzdz, div
    global value_dsigmaxx_dx, value_dsigmaxy_dy, value_dsigmaxz_dz
    global memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxz_dz
    global value_dsigmaxy_dx, value_dsigmayy_dy, value_dsigmayz_dz
    global source_term, force_x, force_y, total_energy
    global epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_xz, epsilon_yz

    # Configuracao e inicializacao da janela de exibicao
    App = pg.QtWidgets.QApplication([])
    windowVx = Window('Vx')
    windowVy = Window('Vy')
    # windowVz = Window('Vz')

    DELTAT_over_rho = dt / rho

    # Inicio do laco de tempo
    for it in range(1, NSTEP+1):
        print(f'it = {it}')

        # Calculo da tensao [stress] - {sigma} (equivalente a pressao nos gases-liquidos)
        # sigma_ii -> tensoes normais; sigma_ij -> tensoes cisalhantes
        # Primeiro "laco" i: 1,NX-1; j: 2,NY; k: 2,NZ -> [1:-2, 2:-1, 2:-1]
        mul_relaxed = np.float32(mu)
        lambdal_relaxed = np.float32(lambda_)
        lambdalplus2mul_relaxed = np.float32(lambdal_relaxed + 2.0 * mul_relaxed)
        lambdal_unrelaxed = np.float32((lambdal_relaxed + 2.0/3.0 * mul_relaxed) * Mu_nu1 -
                                       2.0/3.0 * mul_relaxed * Mu_nu2)
        mul_unrelaxed = np.float32(mul_relaxed * Mu_nu2)
        lambdalplus2mul_unrelaxed = np.float32(lambdal_unrelaxed + 2.0 * mul_unrelaxed)

        value_dvx_dx[1:-2, 2:-1, 2:-1] = (27.0*(vx[2:-1, 2:-1, 2:-1] - vx[1:-2, 2:-1, 2:-1]) -
                                          vx[3:, 2:-1, 2:-1] + vx[:-3, 2:-1, 2:-1]) * one_dx/24.0
        value_dvy_dy[1:-2, 2:-1, 2:-1] = (27.0*(vy[1:-2, 2:-1, 2:-1] - vy[1:-2, 1:-2, 2:-1]) -
                                          vy[1:-2, 3:, 2:-1] + vy[1:-2, :-3, 2:-1]) * one_dy/24.0
        value_dvz_dz[1:-2, 2:-1, 2:-1] = (27.0*(vz[1:-2, 2:-1, 2:-1] - vz[1:-2, 2:-1, 1:-2]) -
                                          vz[1:-2, 2:-1, 3:] + vz[1:-2, 2:-1, :-3]) * one_dz/24.0

        memory_dvx_dx[1:-2, 2:-1, 2:-1] = (b_x_half[:-1, :, :] * memory_dvx_dx[1:-2, 2:-1, 2:-1] +
                                           a_x_half[:-1, :, :] * value_dvx_dx[1:-2, 2:-1, 2:-1])
        memory_dvy_dy[1:-2, 2:-1, 2:-1] = (b_y[:, -1, :] * memory_dvy_dy[1:-2, 2:-1, 2:-1] +
                                           a_y[:, -1, :] * value_dvy_dy[1:-2, 2:-1, 2:-1])
        memory_dvz_dz[1:-2, 2:-1, 2:-1] = (b_z[:, :, -1] * memory_dvz_dz[1:-2, 2:-1, 2:-1] +
                                           a_z[:, :, -1] * value_dvz_dz[1:-2, 2:-1, 2:-1])

        duxdx[1:-2, 2:-1, 2:-1] = value_dvx_dx[1:-2, 2:-1, 2:-1] / k_x_half[:-1, :, :] + memory_dvx_dx[1:-2, 2:-1, 2:-1]
        duydy[1:-2, 2:-1, 2:-1] = value_dvy_dy[1:-2, 2:-1, 2:-1] / k_y[:, :-1, :] + memory_dvy_dy[1:-2, 2:-1, 2:-1]
        duzdz[1:-2, 2:-1, 2:-1] = value_dvz_dz[1:-2, 2:-1, 2:-1] / k_z[:, :, :-1] + memory_dvz_dz[1:-2, 2:-1, 2:-1]

        div = duxdx + duydy + duzdz

        # evolution e1(0)
        tauinv = - inv_tau_sigma_nu1[0]
        Un = e1[0]
        Sn = div * phi_nu1[0]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e1[0] = Unp1

        # evolution e1(1)
        tauinv = - inv_tau_sigma_nu1[1]
        Un = e1[1]
        Sn = div * phi_nu1[1]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e1[1] = Unp1

        # evolution e11(0)
        tauinv = - inv_tau_sigma_nu2[0]
        Un = e11[0]
        Sn = (duxdx - div / np.float32(3.0)) * phi_nu2[0]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e11[1] = Unp1

        # evolution e11(1)
        tauinv = - inv_tau_sigma_nu2[1]
        Un = e11[1]
        Sn = (duxdx - div / np.float32(3.0)) * phi_nu2[1]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e11[1] = Unp1

        # evolution e22(0)
        tauinv = - inv_tau_sigma_nu2[0]
        Un = e22[0]
        Sn = (duydy - div / np.float32(3.0)) * phi_nu2[0]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e22[0] = Unp1

        # evolution e22(1)
        tauinv = - inv_tau_sigma_nu2[1]
        Un = e22[1]
        Sn = (duydy - div / np.float32(3.0)) * phi_nu2[1]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e22[1] = Unp1

        # add the memory variables using the relaxed parameters (Carcione page 111)
        # : there is a bug in Carcione's equation for sigma_zz
        sigmaxx = sigmaxx + dt * ((lambdal_relaxed + 2.0/3.0 * mul_relaxed) * (e1[0] + e1[1]) +
                                  2.0 * mul_relaxed * (e11[0] + e11[1]))
        sigmayy = sigmayy + dt * ((lambdal_relaxed + 2.0/3.0 * mul_relaxed) * (e1[0] + e1[1]) +
                                  2.0 * mul_relaxed * (e22[0] + e22[1]))
        sigmazz = sigmazz + dt * ((lambdal_relaxed + np.float32(2.0) * mul_relaxed) * (e1[0] + e1[1]) -
                                  2.0/3.0 * mul_relaxed * (e11[0] + e11[1] + e22[0] + e22[1]))

        # compute the stress using the unrelaxed Lame parameters (Carcione page 111)
        sigmaxx = sigmaxx + (lambdalplus2mul_unrelaxed*duxdx + lambdal_unrelaxed*duydy + lambdal_unrelaxed*duzdz) * dt
        sigmayy = sigmayy + (lambdal_unrelaxed*duxdx + lambdalplus2mul_unrelaxed*duydy + lambdal_unrelaxed*duzdz) * dt
        sigmazz = sigmazz + (lambdal_unrelaxed*duxdx + lambdal_unrelaxed*duydy + lambdalplus2mul_unrelaxed*duzdz) * dt
        sigmaxx_r = sigmaxx_r + (lambdalplus2mul_relaxed*duxdx + lambdal_relaxed*duydy + lambdal_relaxed*duzdz) * dt
        sigmayy_r = sigmayy_r + (lambdal_relaxed*duxdx + lambdalplus2mul_relaxed*duydy + lambdal_relaxed*duzdz) * dt
        sigmazz_r = sigmazz_r + (lambdal_relaxed*duxdx + lambdal_relaxed*duydy + lambdalplus2mul_relaxed*duzdz) * dt

        # Segundo "laco" i: 2,NX; j: 1,NY-1; k: 1,NZ -> [2:-1, 1:-2, 1:-1]
        mul_relaxed = np.float32(mu)
        mul_unrelaxed = np.float32(mul_relaxed * Mu_nu2)

        value_dvy_dx[2:-1, 1:-2, 1:-1] = (27.0*(vy[2:-1, 1:-2, 1:-1] - vy[1:-2, 1:-2, 1:-1]) -
                                          vy[3:, 1:-2, 1:-1] + vy[:-3, 1:-2, 1:-1]) * one_dx/24.0
        value_dvx_dy[2:-1, 1:-2, 1:-1] = (27.0*(vx[2:-1, 2:-1, 1:-1] - vx[2:-1, 1:-2, 1:-1]) -
                                          vx[2:-1, 3:, 1:-1] + vx[2:-1, :-3, 1:-1]) * one_dy/24.0

        memory_dvy_dx[2:-1, 1:-2, 1:-1] = (b_x[:-1, :, :] * memory_dvy_dx[2:-1, 1:-2, 1:-1] +
                                           a_x[:-1, :, :] * value_dvy_dx[2:-1, 1:-2, 1:-1])
        memory_dvx_dy[2:-1, 1:-2, 1:-1] = (b_y_half[:, :-1, :] * memory_dvx_dy[2:-1, 1:-2, 1:-1] +
                                           a_y_half[:, :-1, :] * value_dvx_dy[2:-1, 1:-2, 1:-1])

        duydx[2:-1, 1:-2, 1:-1] = value_dvy_dx[2:-1, 1:-2, 1:-1]/k_x[:-1, :, :] + memory_dvy_dx[2:-1, 1:-2, 1:-1]
        duxdy[2:-1, 1:-2, 1:-1] = value_dvx_dy[2:-1, 1:-2, 1:-1]/k_y_half[:, :-1, :] + memory_dvx_dy[2:-1, 1:-2, 1:-1]

        # evolution e12(0)
        tauinv = - inv_tau_sigma_nu2[0]
        Un = e12[0]
        Sn = (duxdy + duydx) * phi_nu2[0]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e12[0] = Unp1

        # evolution e12(1)
        tauinv = - inv_tau_sigma_nu2[1]
        Un = e12[1]
        Sn = (duxdy + duydx) * phi_nu2[1]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e12[1] = Unp1

        sigmaxy = sigmaxy + dt * mul_relaxed * (e12[0] + e12[1])
        sigmaxy = sigmaxy + mul_unrelaxed * (duxdy + duydx) * dt
        sigmaxy_r = sigmaxy_r + mul_relaxed * (duxdy + duydx) * dt

        # Terceiro "laco" k: 1,NZ-1;
        # primeira parte:  i: 2,NX; j: 1,NY -> [2:-1, 1:-1, 1:-2]
        mul_relaxed = np.float32(mu)
        mul_unrelaxed = np.float32(mul_relaxed * Mu_nu2)

        value_dvz_dx[2:-1, 1:-1, 1:-2] = (27.0*(vz[2:-1, 1:-1, 1:-2] - vz[1:-2, 1:-1, 1:-2]) -
                                          vz[3:, 1:-1, 1:-2] + vz[:-3, 1:-1, 1:-2]) * one_dx/24.0
        value_dvx_dz[2:-1, 1:-1, 1:-2] = (27.0*(vx[2:-1, 1:-1, 2:-1] - vx[2:-1, 1:-1, 1:-2]) -
                                          vx[2:-1, 1:-1, 3:] + vx[2:-1, 1:-1, :-3]) * one_dz/24.0

        memory_dvz_dx[2:-1, 1:-1, 1:-2] = (b_x[:-1, :, :] * memory_dvz_dx[2:-1, 1:-1, 1:-2] +
                                           a_x[:-1, :, :] * value_dvz_dx[2:-1, 1:-1, 1:-2])
        memory_dvx_dz[2:-1, 1:-1, 1:-2] = (b_z_half[:, :, :-1] * memory_dvx_dz[2:-1, 1:-1, 1:-2] +
                                           a_z_half[:, :, :-1] * value_dvx_dz[2:-1, 1:-1, 1:-2])

        duzdx[2:-1, 1:-1, 1:-2] = value_dvz_dx[2:-1, 1:-1, 1:-2]/k_x[:-1, :, :] + memory_dvz_dx[2:-1, 1:-1, 1:-2]
        duxdz[2:-1, 1:-1, 1:-2] = value_dvx_dz[2:-1, 1:-1, 1:-2]/k_z_half[:, :, :-1] + memory_dvx_dz[2:-1, 1:-1, 1:-2]

        # evolution e13(0)
        tauinv = - inv_tau_sigma_nu2[0]
        Un = e13[0]
        Sn = (duxdz + duzdx) * phi_nu2[0]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e13[0] = Unp1

        # evolution e13[1]
        tauinv = - inv_tau_sigma_nu2[1]
        Un = e13[1]
        Sn = (duxdz + duzdx) * phi_nu2[1]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e13[1] = Unp1

        sigmaxz = sigmaxz + dt * mul_relaxed * (e13[0] + e13[1])
        sigmaxz = sigmaxz + mul_unrelaxed * (duxdz + duzdx) * dt
        sigmaxz_r = sigmaxz_r + mul_relaxed * (duxdz + duzdx) * dt

        # segunda parte:  i: 1,NX; j: 1,NY-1 -> [1:-1, 1:-2, 1:-2]
        mul_relaxed = np.float32(mu)
        mul_unrelaxed = np.float32(mul_relaxed * Mu_nu2)

        value_dvz_dy[1:-1, 1:-2, 1:-2] = (27.0*(vz[1:-1, 2:-1, 1:-2] - vz[1:-1, 1:-2, 1:-2]) -
                                          vz[1:-1, 3:, 1:-2] + vz[1:-1, :-3, 1:-2]) * one_dy/24.0
        value_dvy_dz[1:-1, 1:-2, 1:-2] = (27.0*(vy[1:-1, 1:-2, 2:-1] -vy[1:-1, 1:-2, 1:-2]) -
                                          vy[1:-1, 1:-2, 3:] + vy[1:-1, 1:-2, :-3]) * one_dz/24.0

        memory_dvz_dy[1:-1, 1:-2, 1:-2] = (b_y_half[:, :-1, :] * memory_dvz_dy[1:-1, 1:-2, 1:-2] +
                                           a_y_half[:, :-1, :] * value_dvz_dy[1:-1, 1:-2, 1:-2])
        memory_dvy_dz[1:-1, 1:-2, 1:-2] = (b_z_half[:, :, :-1] * memory_dvy_dz[1:-1, 1:-2, 1:-2] +
                                           a_z_half[:, :, :-1] * value_dvy_dz[1:-1, 1:-2, 1:-2])

        duzdy[1:-1, 1:-2, 1:-2] = value_dvz_dy[1:-1, 1:-2, 1:-2]/k_y_half[:, :-1, :] + memory_dvz_dy[1:-1, 1:-2, 1:-2]
        duydz[1:-1, 1:-2, 1:-2] = value_dvy_dz[1:-1, 1:-2, 1:-2]/k_z_half[:, :, :-1] + memory_dvy_dz[1:-1, 1:-2, 1:-2]

        # evolution e23(0)
        tauinv = - inv_tau_sigma_nu2[0]
        Un = e23[0]
        Sn = (duydz + duzdy) * phi_nu2[0]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e23[0] = Unp1

        # evolution e23(1)
        tauinv = - inv_tau_sigma_nu2[1]
        Un = e23[1]
        Sn = (duydz + duzdy) * phi_nu2[1]
        tauinvUn = tauinv * Un
        Unp1 = (Un + dt * (Sn + 0.5 * tauinvUn)) / (1.0 - dt * 0.5 * tauinv)
        e23[1] = Unp1

        sigmayz = sigmayz + dt * mul_relaxed * (e23[0] + e23[1])
        sigmayz = sigmayz + mul_unrelaxed * (duydz + duzdy) * dt
        sigmayz_r = sigmayz_r + mul_relaxed * (duydz + duzdy) * dt

        # Calculo da velocidade
        # Primeiro "laco" k: 2,NZ;
        # primeira parte:  i: 2,NX; j: 2,NY -> [2:-1, 2:-1, 2:-1]
        value_dsigmaxx_dx[2:-1, 2:-1, 2:-1] = (27.0*(sigmaxx[2:-1, 2:-1, 2:-1] - sigmaxx[1:-2, 2:-1, 2:-1]) -
                                               sigmaxx[3:, 2:-1, 2:-1] + sigmaxx[:-3, 2:-1, 2:-1]) * one_dx/24.0
        value_dsigmaxy_dy[2:-1, 2:-1, 2:-1] = (27.0*(sigmaxy[2:-1, 2:-1, 2:-1] - sigmaxy[2:-1, 1:-2, 2:-1]) -
                                               sigmaxy[2:-1, 3:, 2:-1] + sigmaxy[2:-1, :-3, 2:-1]) * one_dy/24.0
        value_dsigmaxz_dz[2:-1, 2:-1, 2:-1] = (27.0*(sigmaxz[2:-1, 2:-1, 2:-1] - sigmaxz[2:-1, 2:-1, 1:-2]) -
                                               sigmaxz[2:-1, 2:-1, 3:] + sigmaxz[2:-1, 2:-1, :-3]) * one_dz/24.0

        memory_dsigmaxx_dx[2:-1, 2:-1, 2:-1] = (b_x[:-1, :,:] * memory_dsigmaxx_dx[2:-1, 2:-1, 2:-1] +
                                                a_x[:-1, :, :] * value_dsigmaxx_dx[2:-1, 2:-1, 2:-1])
        memory_dsigmaxy_dy[2:-1, 2:-1, 2:-1] = (b_y[:, :-1, :] * memory_dsigmaxy_dy[2:-1, 2:-1, 2:-1] +
                                                a_y[:, :-1, :] * value_dsigmaxy_dy[2:-1, 2:-1, 2:-1])
        memory_dsigmaxz_dz[2:-1, 2:-1, 2:-1] = (b_z[:, :, :-1] * memory_dsigmaxz_dz[2:-1, 2:-1, 2:-1] +
                                                a_z[:, :, :-1] * value_dsigmaxz_dz[2:-1, 2:-1, 2:-1])

        value_dsigmaxx_dx[2:-1, 2:-1, 2:-1] = (value_dsigmaxx_dx[2:-1, 2:-1, 2:-1]/k_x[:-1, :, :] +
                                               memory_dsigmaxx_dx[2:-1, 2:-1, 2:-1])
        value_dsigmaxy_dy[2:-1, 2:-1, 2:-1] = (value_dsigmaxy_dy[2:-1, 2:-1, 2:-1]/k_y[:, :-1, :] +
                                               memory_dsigmaxy_dy[2:-1, 2:-1, 2:-1])
        value_dsigmaxz_dz[2:-1, 2:-1, 2:-1] = (value_dsigmaxz_dz[2:-1, 2:-1, 2:-1]/k_z[:, :, :-1] +
                                               memory_dsigmaxz_dz[2:-1, 2:-1, 2:-1])

        vx = DELTAT_over_rho * (value_dsigmaxx_dx + value_dsigmaxy_dy + value_dsigmaxz_dz) + vx

        # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2, 2:-1]
        value_dsigmaxy_dx[1:-2, 1:-2, 2:-1] = (27.0*(sigmaxy[2:-1, 1:-2, 2:-1] - sigmaxy[1:-2, 1:-2, 2:-1]) -
                                               sigmaxy[3:, 1:-2, 2:-1] + sigmaxy[:-3, 1:-2, 2:-1]) * one_dx/24.0
        value_dsigmayy_dy[1:-2, 1:-2, 2:-1] = (27.0*(sigmayy[1:-2, 2:-1, 2:-1] - sigmayy[1:-2, 1:-2, 2:-1]) -
                                               sigmayy[1:-2, 3:, 2:-1] + sigmayy[1:-2, :-3, 2:-1]) * one_dy/24.0
        value_dsigmayz_dz[1:-2, 1:-2, 2:-1] = (27.0*(sigmayz[1:-2, 1:-2, 2:-1] - sigmayz[1:-2, 1:-2, 1:-2]) -
                                               sigmayz[1:-2, 1:-2, 3:] + sigmayz[1:-2, 1:-2, :-3]) * one_dz/24.0

        memory_dsigmaxy_dx[1:-2, 1:-2, 2:-1] = (b_x_half[:-1, :, :] * memory_dsigmaxy_dx[1:-2, 1:-2, 2:-1] +
                                                a_x_half[:-1, :, :] * value_dsigmaxy_dx[1:-2, 1:-2, 2:-1])
        memory_dsigmayy_dy[1:-2, 1:-2, 2:-1] = (b_y_half[:, :-1, :] * memory_dsigmayy_dy[1:-2, 1:-2, 2:-1] +
                                                a_y_half[:, :-1, :] * value_dsigmayy_dy[1:-2, 1:-2, 2:-1])
        memory_dsigmayz_dz[1:-2, 1:-2, 2:-1] = (b_z[:, :, :-1] * memory_dsigmayz_dz[1:-2, 1:-2, 2:-1] +
                                                a_z[:, :, :-1] * value_dsigmayz_dz[1:-2, 1:-2, 2:-1])

        value_dsigmaxy_dx[1:-2, 1:-2, 2:-1] = (value_dsigmaxy_dx[1:-2, 1:-2, 2:-1]/k_x_half[:-1, :, :] +
                                               memory_dsigmaxy_dx[1:-2, 1:-2, 2:-1])
        value_dsigmayy_dy[1:-2, 1:-2, 2:-1] = (value_dsigmayy_dy[1:-2, 1:-2, 2:-1]/k_y_half[:, :-1, :] +
                                               memory_dsigmayy_dy[1:-2, 1:-2, 2:-1])
        value_dsigmayz_dz[1:-2, 1:-2, 2:-1] = (value_dsigmayz_dz[1:-2, 1:-2, 2:-1]/k_z[:, :, :-1] +
                                               memory_dsigmayz_dz[1:-2, 1:-2, 2:-1])

        vy = DELTAT_over_rho * (value_dsigmaxy_dx + value_dsigmayy_dy + value_dsigmayz_dz) + vy

        # Segundo "laco" i: 1,NX-1; j: 2,NY; k: 1,NZ-1; -> [1:-2, 2:-1, 1:-2]
        value_dsigmaxz_dx[1:-2, 2:-1, 1:-2] = (27.0*(sigmaxz[2:-1, 2:-1, 1:-2] - sigmaxz[1:-2, 2:-1, 1:-2]) -
                                               sigmaxz[3:, 2:-1, 1:-2] + sigmaxz[:-3, 2:-1, 1:-2]) * one_dx/24.0
        value_dsigmayz_dy[1:-2, 2:-1, 1:-2] = (27.0*(sigmayz[1:-2, 2:-1, 1:-2] - sigmayz[1:-2, 1:-2, 1:-2]) -
                                               sigmayz[1:-2, 3:, 1:-2] + sigmayz[1:-2, :-3, 1:-2]) * one_dy/24.0
        value_dsigmazz_dz[1:-2, 2:-1, 1:-2] = (27.0*(sigmazz[1:-2, 2:-1, 2:-1] - sigmazz[1:-2, 2:-1, 1:-2]) -
                                               sigmazz[1:-2, 2:-1, 3:] + sigmazz[1:-2, 2:-1, :-3]) * one_dz/24.0

        memory_dsigmaxz_dx[1:-2, 2:-1, 1:-2] = (b_x_half[:-1, :, :] * memory_dsigmaxz_dx[1:-2, 2:-1, 1:-2] +
                                                a_x_half[:-1, :, :] * value_dsigmaxz_dx[1:-2, 2:-1, 1:-2])
        memory_dsigmayz_dy[1:-2, 2:-1, 1:-2] = (b_y[:, :-1, :] * memory_dsigmayz_dy[1:-2, 2:-1, 1:-2] +
                                                a_y[:, :-1, :] * value_dsigmayz_dy[1:-2, 2:-1, 1:-2])
        memory_dsigmazz_dz[1:-2, 2:-1, 1:-2] = (b_z_half[:, :, :-1] * memory_dsigmazz_dz[1:-2, 2:-1, 1:-2] +
                                                a_z_half[:, :, :-1] * value_dsigmazz_dz[1:-2, 2:-1, 1:-2])

        value_dsigmaxz_dx[1:-2, 2:-1, 1:-2] = (value_dsigmaxz_dx[1:-2, 2:-1, 1:-2]/k_x_half[:-1, :, :] +
                                               memory_dsigmaxz_dx[1:-2, 2:-1, 1:-2])
        value_dsigmayz_dy[1:-2, 2:-1, 1:-2] = (value_dsigmayz_dy[1:-2, 2:-1, 1:-2]/k_y[:, :-1, :] +
                                               memory_dsigmayz_dy[1:-2, 2:-1, 1:-2])
        value_dsigmazz_dz[1:-2, 2:-1, 1:-2] = (value_dsigmazz_dz[1:-2, 2:-1, 1:-2]/k_z_half[:, :, :-1] +
                                               memory_dsigmazz_dz[1:-2, 2:-1, 1:-2])

        vz = DELTAT_over_rho * (value_dsigmaxz_dx + value_dsigmayz_dy + value_dsigmazz_dz) + vz

        # add the source (force vector located at a given grid point)
        vx[isource, jsource, ksource] += force_x[it - 1] * dt/rho
        vy[isource, jsource, ksource] += force_y[it - 1] * dt/rho

        # implement Dirichlet boundary conditions on the six edges of the grid
        # which is the right condition to implement in order for C-PML to remain stable at long times
        # xmin
        vx[0:1, :, :] = ZERO
        vy[0:1, :, :] = ZERO
        vz[0:1, :, :] = ZERO

        # xmax
        vx[-2:-1, :, :] = ZERO
        vy[-2:-1, :, :] = ZERO
        vz[-2:-1, :, :] = ZERO

        # ymin
        vx[:, 0:1, :] = ZERO
        vy[:, 0:1, :] = ZERO
        vz[:, 0:1, :] = ZERO

        # ymax
        vx[:, -2:-1, :] = ZERO
        vy[:, -2:-1, :] = ZERO
        vz[:, -2:-1, :] = ZERO

        # zmin
        vx[:, :, 0:1] = ZERO
        vy[:, :, 0:1] = ZERO
        vz[:, :, 0:1] = ZERO

        # zmax
        vx[:, :, -2:-1] = ZERO
        vy[:, :, -2:-1] = ZERO
        vz[:, :, -2:-1] = ZERO

        # Store seismograms
        for _irec in range(NREC):
            sisvx[it - 1, _irec] = vx[ix_rec[_irec], iy_rec[_irec], ksource]
            sisvy[it - 1, _irec] = vy[ix_rec[_irec], iy_rec[_irec], ksource]

        # Compute total energy in the medium (without the PML layers)
        imin = npoints_pml
        imax = nx - npoints_pml + 1
        jmin = npoints_pml
        jmax = ny - npoints_pml + 1
        kmin = npoints_pml
        kmax = nz - npoints_pml + 1

        local_energy_kinetic = 0.5 * rho * (np.sum(vx[imin: imax, jmin: jmax, kmin: kmax] ** 2) +
                                            np.sum(vy[imin: imax, jmin: jmax, kmin: kmax] ** 2) +
                                            np.sum(vz[imin: imax, jmin: jmax, kmin: kmax] ** 2))

        # compute total field from split components
        epsilon_xx[imin: imax, jmin: jmax, kmin: kmax] =(
                (2.0 * (lambda_ + mu) * sigmaxx[imin: imax, jmin: jmax, kmin: kmax] -
                 lambda_ * sigmayy[imin: imax, jmin: jmax, kmin: kmax] -
                 lambda_ * sigmazz[imin: imax, jmin: jmax, kmin: kmax])/(2.0 * mu * (3.0 * lambda_ + 2.0 * mu)))
        epsilon_yy[imin: imax, jmin: jmax, kmin: kmax] =(
                (2.0 * (lambda_ + mu) * sigmayy[imin: imax, jmin: jmax, kmin: kmax] -
                 lambda_ * sigmaxx[imin: imax, jmin: jmax, kmin: kmax] -
                 lambda_ * sigmazz[imin: imax, jmin: jmax, kmin: kmax])/(2.0 * mu * (3.0 * lambda_ + 2.0 * mu)))
        epsilon_zz[imin: imax, jmin: jmax, kmin: kmax] =(
                (2.0 * (lambda_ + mu) * sigmazz[imin: imax, jmin: jmax, kmin: kmax] -
                 lambda_ * sigmaxx[imin: imax, jmin: jmax, kmin: kmax] -
                 lambda_ * sigmayy[imin: imax, jmin: jmax, kmin: kmax])/(2.0 * mu * (3.0 * lambda_ + 2.0 * mu)))
        epsilon_xy[imin: imax, jmin: jmax, kmin: kmax] = sigmaxy_r[imin: imax, jmin: jmax, kmin: kmax]/(2.0 * mu)
        epsilon_xz[imin: imax, jmin: jmax, kmin: kmax] = sigmaxz_r[imin: imax, jmin: jmax, kmin: kmax]/(2.0 * mu)
        epsilon_yz[imin: imax, jmin: jmax, kmin: kmax] = sigmayz_r[imin: imax, jmin: jmax, kmin: kmax]/(2.0 * mu)

        local_energy_potential = 0.5 * np.sum(epsilon_xx * sigmaxx_r +
                                              epsilon_yy * sigmayy_r +
                                              epsilon_zz * sigmazz_r +
                                              2.0 * epsilon_xy * sigmaxy_r +
                                              2.0 * epsilon_xz * sigmaxz_r +
                                              2.0 * epsilon_yz * sigmayz_r)

        total_energy[it - 1] = local_energy_kinetic + local_energy_potential

        v_solid_norm = np.max(np.sqrt(vx[:, :, 1: -1] ** 2 + vy[:, :, 1: -1] ** 2 + vz[:, :, 1: -1] ** 2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            print(f'Time step # {it} out of {NSTEP}')
            print(f'Max Vx = {np.max(vx)}, Vy = {np.max(vy)}, Vz = {np.max(vz)}')
            print(f'Min Vx = {np.min(vx)}, Vy = {np.min(vy)}, Vz = {np.min(vz)}')
            print(f'Max norm velocity vector V (m/s) = {v_solid_norm}')
            print(f'Total energy = {total_energy[it - 1]}')

        windowVx.imv.setImage(vx[:, :, ksource], levels=[-0.5, 0.5])
        # windowVx.imv.setImage(vx[:, :, ksource], levels=[np.min(vx), np.max(vx)])
        windowVy.imv.setImage(vy[:, :, ksource], levels=[-6.0, 6.0])
        # windowVy.imv.setImage(vy[:, :, ksource], levels=[np.min(vy), np.max(vy)])
        # windowVz.imv.setImage(vz[:, :, ksource], levels=[-1.0, 1.0])
        App.processEvents()

        # Verifica a estabilidade da simulacao
        if v_solid_norm > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instável")
            exit(2)

    App.exit()

    # End of the main loop
    print("Simulacao terminada.")


# --------------------------
# Shader [kernel] para a simulação em WEBGPU
shader_test = f"""
    struct SimIntValues {{
        y_sz: i32,          // Y field size
        x_sz: i32,          // X field size
        y_sens: i32,        // Y sensor
        x_sens: i32,        // X sensor
        k: i32              // iteraction
    }};
    
    struct SimFltValues {{
        cp_unrelaxed: f32,  // sound speed
        dx: f32,            // delta x
        dy: f32,            // delta y
        dt: f32             // delta t
    }};

    // Group 0 - parameters
    @group(0) @binding(0)   // param_flt32
    var<storage,read> sim_flt_par: SimFltValues;

    @group(0) @binding(1) // source term
    var<storage,read> src: array<f32>;
    
    @group(0) @binding(2) // kronecker_src, rho_half_x, rho_half_y, kappa
    var<storage,read> img_params: array<f32>;

    @group(0) @binding(3) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
    var<storage,read> coef_x: array<f32>;
    
    @group(0) @binding(4) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
    var<storage,read> coef_y: array<f32>;
    
    @group(0) @binding(5) // param_int32
    var<storage,read_write> sim_int_par: SimIntValues;

    // Group 1 - simulation arrays
    @group(1) @binding(6) // pressure future (p_0)
    var<storage,read_write> pr_future: array<f32>;

    @group(1) @binding(7) // pressure fields p_1, p_2
    var<storage,read_write> pr_fields: array<f32>;

    @group(1) @binding(8) // derivative fields x (v_x, mdp_x, dp_x, dmdp_x)
    var<storage,read_write> der_x: array<f32>;

    @group(1) @binding(9) // derivative fields y (v_y, mdp_y, dp_y, dmdp_y)
    var<storage,read_write> der_y: array<f32>;

    // Group 2 - sensors arrays
    @group(2) @binding(10) // sensor signal
    var<storage,read_write> sensor: array<f32>;

    // function to convert 2D [y,x] index into 1D [yx] index
    fn yx(y: i32, x: i32) -> i32 {{
        let index = y + x * sim_int_par.y_sz;

        return select(-1, index, y >= 0 && y < sim_int_par.y_sz && x >= 0 && x < sim_int_par.x_sz);
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
    
    // function to get a kappa array value
    fn get_kappa(i: i32, j: i32) -> f32 {{
        let index: i32 = ijk(i, j, 3, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }}
    
    // function to get a kronecker_src array value
    fn get_kronecker_src(i: i32, j: i32) -> f32 {{
        let index: i32 = ijk(i, j, 0, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }}
    
    // function to get a rho_h_x array value
    fn get_rho_h_x(i: i32, j: i32) -> f32 {{
        let index: i32 = ijk(i, j, 1, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }}
    
    // function to get a rho_h_y array value
    fn get_rho_h_y(i: i32, j: i32) -> f32 {{
        let index: i32 = ijk(i, j, 2, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }}
    
    // function to get a src array value
    fn get_src(n: i32) -> f32 {{
        return select(0.0, src[n], n >= 0);
    }}
    
    // function to get a a_x array value
    fn get_a_x(n: i32) -> f32 {{
        let index: i32 = ij(0, n, 6, sim_int_par.x_sz);
        
        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a b_x array value
    fn get_b_x(n: i32) -> f32 {{
        let index: i32 = ij(1, n, 6, sim_int_par.x_sz);
        
        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a k_x array value
    fn get_k_x(n: i32) -> f32 {{
        let index: i32 = ij(2, n, 6, sim_int_par.x_sz);
        
        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a a_y array value
    fn get_a_y(n: i32) -> f32 {{
        let index: i32 = ij(0, n, 6, sim_int_par.y_sz);
        
        return select(0.0, coef_y[index], index != -1);
    }}
    
    // function to get a b_y array value
    fn get_b_y(n: i32) -> f32 {{
        let index: i32 = ij(1, n, 6, sim_int_par.y_sz);
        
        return select(0.0, coef_y[index], index != -1);
    }}
    
    // function to get a k_y array value
    fn get_k_y(n: i32) -> f32 {{
        let index: i32 = ij(2, n, 6, sim_int_par.y_sz);
        
        return select(0.0, coef_y[index], index != -1);
    }}
    
    // function to get a a_x_h array value
    fn get_a_x_h(n: i32) -> f32 {{
        let index: i32 = ij(3, n, 6, sim_int_par.x_sz);
        
        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a b_x_h array value
    fn get_b_x_h(n: i32) -> f32 {{
        let index: i32 = ij(4, n, 6, sim_int_par.x_sz);
        
        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a k_x_h array value
    fn get_k_x_h(n: i32) -> f32 {{
        let index: i32 = ij(5, n, 6, sim_int_par.x_sz);
        
        return select(0.0, coef_x[index], index != -1);
    }}
    
    // function to get a a_y_h array value
    fn get_a_y_h(n: i32) -> f32 {{
        let index: i32 = ij(3, n, 6, sim_int_par.y_sz);
        
        return select(0.0, coef_y[index], index != -1);
    }}
    
    // function to get a b_y_h array value
    fn get_b_y_h(n: i32) -> f32 {{
        let index: i32 = ij(4, n, 6, sim_int_par.y_sz);
        
        return select(0.0, coef_y[index], index != -1);
    }}
    
    // function to get a k_y_h array value
    fn get_k_y_h(n: i32) -> f32 {{
        let index: i32 = ij(5, n, 6, sim_int_par.y_sz);
        
        return select(0.0, coef_y[index], index != -1);
    }}

    // function to get an p_0 (pr_future) array value
    fn get_p_0(y: i32, x: i32) -> f32 {{
        let index: i32 = yx(y, x);

        return select(0.0, pr_future[index], index != -1);
    }}

    // function to set a p_0 array value
    fn set_p_0(y: i32, x: i32, val : f32) {{
        let index: i32 = yx(y, x);

        if(index != -1) {{
            pr_future[index] = val;
        }}
    }}

    // function to get an p_1 array value
    fn get_p_1(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 0, sim_int_par.y_sz, sim_int_par.x_sz, 2);

        return select(0.0, pr_fields[index], index != -1);
    }}

    // function to set a p_1 array value
    fn set_p_1(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 0, sim_int_par.y_sz, sim_int_par.x_sz, 2);

        if(index != -1) {{
            pr_fields[index] = val;
        }}
    }}

    // function to get an p_2 array value
    fn get_p_2(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 1, sim_int_par.y_sz, sim_int_par.x_sz, 2);

        return select(0.0, pr_fields[index], index != -1);
    }}

    // function to set a p_2 array value
    fn set_p_2(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 1, sim_int_par.y_sz, sim_int_par.x_sz, 2);

        if(index != -1) {{
            pr_fields[index] = val;
        }}
    }}
    
    // function to get an v_x array value
    fn get_v_x(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 0, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }}

    // function to set a v_x array value
    fn set_v_x(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 0, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_x[index] = val;
        }}
    }}
    
    // function to get an v_y array value
    fn get_v_y(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 0, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }}

    // function to set a v_y array value
    fn set_v_y(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 0, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_y[index] = val;
        }}
    }}
    
    // function to get an mdp_x array value
    fn get_mdp_x(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 1, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }}

    // function to set a mdp_x array value
    fn set_mdp_x(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 1, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_x[index] = val;
        }}
    }}
    
    // function to get an mdp_y array value
    fn get_mdp_y(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 1, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }}

    // function to set a mdp_y array value
    fn set_mdp_y(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 1, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_y[index] = val;
        }}
    }}
    
    // function to get an dp_x array value
    fn get_dp_x(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 2, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }}

    // function to set a dp_x array value
    fn set_dp_x(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 2, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_x[index] = val;
        }}
    }}
    
    // function to get an dp_y array value
    fn get_dp_y(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 2, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }}

    // function to set a dp_y array value
    fn set_dp_y(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 2, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_y[index] = val;
        }}
    }}
    
    // function to get an dmdp_x array value
    fn get_dmdp_x(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 3, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }}

    // function to set a dmdp_x array value
    fn set_dmdp_x(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 3, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_x[index] = val;
        }}
    }}
    
    // function to get an dmdp_y array value
    fn get_dmdp_y(y: i32, x: i32) -> f32 {{
        let index: i32 = ijk(y, x, 3, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }}

    // function to set a dmdp_y array value
    fn set_dmdp_y(y: i32, x: i32, val : f32) {{
        let index: i32 = ijk(y, x, 3, sim_int_par.y_sz, sim_int_par.x_sz, 4);

        if(index != -1) {{
            der_y[index] = val;
        }}
    }}

    // function to calculate first derivatives
    @compute
    @workgroup_size({wsy}, {wsx})
    fn space_sim1(@builtin(global_invocation_id) index: vec3<u32>) {{
        let y: i32 = i32(index.x);          // y thread index
        let x: i32 = i32(index.y);          // x thread index
        var vdp_x: f32 = 0.0;
        var vdp_y: f32 = 0.0;
        
        // Calcula a primeira derivada espacial dividida pela densidade
        vdp_x = (get_p_1(y + 1, x) - get_p_1(y, x)) / sim_flt_par.dx;
        set_mdp_x(y, x, get_b_x_h(y)*get_mdp_x(y, x) + get_a_x_h(y)*vdp_x);
        vdp_y = (get_p_1(y, x + 1) - get_p_1(y, x)) / sim_flt_par.dy;
        set_mdp_y(y, x, get_b_y_h(x)*get_mdp_y(y, x) + get_a_y_h(x)*vdp_y);
        set_dp_x(y, x, (vdp_x / get_k_x_h(y) + get_mdp_x(y, x))/get_rho_h_x(y, x));
        set_dp_y(y, x, (vdp_y / get_k_y_h(x) + get_mdp_y(y, x))/get_rho_h_y(y, x));      
    }}
    
    // function to calculate second derivatives
    @compute
    @workgroup_size({wsy}, {wsx})
    fn space_sim2(@builtin(global_invocation_id) index: vec3<u32>) {{
        let y: i32 = i32(index.x);          // y thread index
        let x: i32 = i32(index.y);          // x thread index
        var vdp_xx: f32 = 0.0;
        var vdp_yy: f32 = 0.0;
            
        // Calcula a segunda derivada espacial
        vdp_xx = (get_dp_x(y, x) - get_dp_x(y - 1, x)) / sim_flt_par.dx;
        set_dmdp_x(y, x, get_b_x(y)*get_dmdp_x(y, x) + get_a_x(y)*vdp_xx);
        vdp_yy = (get_dp_y(y, x) - get_dp_y(y, x - 1)) / sim_flt_par.dy;
        set_dmdp_y(y, x, get_b_y(x)*get_dmdp_y(y, x) + get_a_y(x)*vdp_yy);
        set_v_x(y, x, vdp_xx / get_k_x(y) + get_dmdp_x(y, x));
        set_v_y(y, x, vdp_yy / get_k_y(x) + get_dmdp_y(y, x));        
    }}

    @compute
    @workgroup_size(1)
    fn incr_k() {{
        sim_int_par.k += 1;
    }}

    @compute
    @workgroup_size({wsy}, {wsx})
    fn time_sim(@builtin(global_invocation_id) index: vec3<u32>) {{
        var add_src: f32 = 0.0;             // Source term
        let y: i32 = i32(index.x);          // y thread index
        let x: i32 = i32(index.y);          // x thread index
        let dt: f32 = sim_flt_par.dt;
        let pi_4: f32 = 12.5663706144;

        // --------------------
        // Update pressure field
        add_src = pi_4*sim_flt_par.cp_unrelaxed*sim_flt_par.cp_unrelaxed*src[sim_int_par.k]*get_kronecker_src(y, x);
        set_p_0(y, x, -1.0*get_p_2(y, x) + 2.0*get_p_1(y, x) +
            dt*dt*((get_v_x(y, x) + get_v_y(y, x))*get_kappa(y, x) + add_src));

        // Aplly Dirichlet conditions
        if(y == 0 || y == (sim_int_par.y_sz - 1) || x == 0 || x == (sim_int_par.x_sz - 1)) {{
            set_p_0(y, x, 0.0);
        }}
            
        // --------------------
        // Circular buffer
        set_p_2(y, x, get_p_1(y, x));
        set_p_1(y, x, get_p_0(y, x));

        if(y == sim_int_par.y_sens && x == sim_int_par.x_sens) {{
            sensor[sim_int_par.k] = get_p_0(y, x);
        }}
    }}
    """


# Simulação completa em WEB GPU
# def sim_webgpu():
#     global p_2, p_1, p_0, mdp_x, mdp_y, dp_x, dp_y, dmdp_x, dmdp_y, v_x, v_y, a_x, NSTEP
#
#     # Arrays com parametros inteiros (i32) e ponto flutuante (f32) para rodar o simulador
#     params_i32 = np.array([ny, nx, sens_y, sens_x, 0], dtype=np.int32)
#     params_f32 = np.array([cp_unrelaxed, dx, dy, dt], dtype=flt32)
#
#     # =====================
#     # webgpu configurations
#     if gpu_type == "NVIDIA":
#         device = wgpu.utils.get_default_device()
#     else:
#         adapter = wgpu.request_adapter(canvas=None, power_preference="low-power")
#         device = adapter.request_device()
#
#     # Cria o shader para calculo contido na string ``shader_test''
#     cshader = device.create_shader_module(code=shader_test)
#
#     # Definicao dos buffers que terao informacoes compartilhadas entre CPU e GPU
#     # ------- Buffers para o binding de parametros -------------
#     # Buffer de parametros com valores em ponto flutuante
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     # Binding 0
#     b_param_flt32 = device.create_buffer_with_data(data=params_f32,
#                                                    usage=wgpu.BufferUsage.STORAGE |
#                                                          wgpu.BufferUsage.COPY_SRC)
#
#     # Termo de fonte
#     # Binding 1
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     b_src = device.create_buffer_with_data(data=source_term,
#                                            usage=wgpu.BufferUsage.STORAGE |
#                                                  wgpu.BufferUsage.COPY_SRC)
#
#     # Mapa da posicao das fontes
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     # Binding 2
#     b_img_params = device.create_buffer_with_data(data=np.vstack((kronecker_source,
#                                                                   rho_half_x,
#                                                                   rho_half_y,
#                                                                   kappa_unrelaxed)),
#                                                    usage=wgpu.BufferUsage.STORAGE |
#                                                          wgpu.BufferUsage.COPY_SRC)
#
#     # Coeficientes de absorcao
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     # Binding 3
#     b_coef_x = device.create_buffer_with_data(data=np.row_stack((a_x, b_x, k_x, a_x_half, b_x_half, k_x_half)),
#                                               usage=wgpu.BufferUsage.STORAGE |
#                                                     wgpu.BufferUsage.COPY_SRC)
#
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     # Binding 4
#     b_coef_y = device.create_buffer_with_data(data=np.row_stack((a_y, b_y, k_y, a_y_half, b_y_half, k_y_half)),
#                                               usage=wgpu.BufferUsage.STORAGE |
#                                                     wgpu.BufferUsage.COPY_SRC)
#
#     # Buffer de parametros com valores inteiros
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     # Binding 5
#     b_param_int32 = device.create_buffer_with_data(data=params_i32,
#                                                    usage=wgpu.BufferUsage.STORAGE |
#                                                          wgpu.BufferUsage.COPY_SRC)
#
#     # Buffers com os campos de pressao
#     # Pressao futura (amostra de tempo n+1)
#     # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
#     # Binding 6
#     b_p_0 = device.create_buffer_with_data(data=p_0,
#                                            usage=wgpu.BufferUsage.STORAGE |
#                                                  wgpu.BufferUsage.COPY_DST |
#                                                  wgpu.BufferUsage.COPY_SRC)
#     # Campos de pressao atual
#     # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
#     # Binding 7
#     b_pr_fields = device.create_buffer_with_data(data=np.vstack((p_1, p_2)),
#                                                  usage=wgpu.BufferUsage.STORAGE |
#                                                        wgpu.BufferUsage.COPY_SRC)
#     # Matrizes para o calculo das derivadas (primeira e segunda)
#     # Binding 8 - matrizes para o eixo x
#     b_der_x = device.create_buffer_with_data(data=np.vstack((v_x, mdp_x, dp_x, dmdp_x)),
#                                              usage=wgpu.BufferUsage.STORAGE |
#                                                    wgpu.BufferUsage.COPY_SRC)
#     # Binding 9 - matrizes para o eixo y
#     b_der_y = device.create_buffer_with_data(data=np.vstack((v_y, mdp_y, dp_y, dmdp_y)),
#                                              usage=wgpu.BufferUsage.STORAGE |
#                                                    wgpu.BufferUsage.COPY_SRC)
#
#     # Sinal do sensor
#     # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
#     # Binding 10
#     b_sens = device.create_buffer_with_data(data=sensor,
#                                             usage=wgpu.BufferUsage.STORAGE |
#                                                   wgpu.BufferUsage.COPY_DST |
#                                                   wgpu.BufferUsage.COPY_SRC)
#
#     # Esquema de amarracao dos parametros (binding layouts [bl])
#     # Parametros
#     bl_params = [
#         {"binding": ii,
#          "visibility": wgpu.ShaderStage.COMPUTE,
#          "buffer": {
#              "type": wgpu.BufferBindingType.read_only_storage}
#          } for ii in range(5)
#     ]
#     # b_param_i32
#     bl_params.append({
#                          "binding": 5,
#                          "visibility": wgpu.ShaderStage.COMPUTE,
#                          "buffer": {
#                              "type": wgpu.BufferBindingType.storage,
#                          },
#                      })
#
#     # Arrays da simulacao
#     bl_sim_arrays = [
#         {"binding": ii,
#          "visibility": wgpu.ShaderStage.COMPUTE,
#          "buffer": {
#              "type": wgpu.BufferBindingType.storage}
#          } for ii in range(6, 10)
#     ]
#
#     # Sensores
#     bl_sensors = [
#         {
#             "binding": 10,
#             "visibility": wgpu.ShaderStage.COMPUTE,
#             "buffer": {
#                 "type": wgpu.BufferBindingType.storage,
#             },
#         },
#     ]
#
#     # Configuracao das amarracoes (bindings)
#     b_params = [
#         {
#             "binding": 0,
#             "resource": {"buffer": b_param_flt32, "offset": 0, "size": b_param_flt32.size},
#         },
#         {
#             "binding": 1,
#             "resource": {"buffer": b_src, "offset": 0, "size": b_src.size},
#         },
#         {
#             "binding": 2,
#             "resource": {"buffer": b_img_params, "offset": 0, "size": b_img_params.size},
#         },
#         {
#             "binding": 3,
#             "resource": {"buffer": b_coef_x, "offset": 0, "size": b_coef_x.size},
#         },
#         {
#             "binding": 4,
#             "resource": {"buffer": b_coef_y, "offset": 0, "size": b_coef_y.size},
#         },
#         {
#             "binding": 5,
#             "resource": {"buffer": b_param_int32, "offset": 0, "size": b_param_int32.size},
#         },
#     ]
#     b_sim_arrays = [
#         {
#             "binding": 6,
#             "resource": {"buffer": b_p_0, "offset": 0, "size": b_p_0.size},
#         },
#         {
#             "binding": 7,
#             "resource": {"buffer": b_pr_fields, "offset": 0, "size": b_pr_fields.size},
#         },
#         {
#             "binding": 8,
#             "resource": {"buffer": b_der_x, "offset": 0, "size": b_der_x.size},
#         },
#         {
#             "binding": 9,
#             "resource": {"buffer": b_der_y, "offset": 0, "size": b_der_y.size},
#         },
#     ]
#     b_sensors = [
#         {
#             "binding": 10,
#             "resource": {"buffer": b_sens, "offset": 0, "size": b_sens.size},
#         },
#     ]
#
#     # Coloca tudo junto
#     bgl_0 = device.create_bind_group_layout(entries=bl_params)
#     bgl_1 = device.create_bind_group_layout(entries=bl_sim_arrays)
#     bgl_2 = device.create_bind_group_layout(entries=bl_sensors)
#     pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_0, bgl_1, bgl_2])
#     bg_0 = device.create_bind_group(layout=bgl_0, entries=b_params)
#     bg_1 = device.create_bind_group(layout=bgl_1, entries=b_sim_arrays)
#     bg_2 = device.create_bind_group(layout=bgl_2, entries=b_sensors)
#
#     # Cria os pipelines de execucao
#     compute_space_sim1 = device.create_compute_pipeline(layout=pipeline_layout,
#                                                         compute={"module": cshader, "entry_point": "space_sim1"})
#     compute_space_sim2 = device.create_compute_pipeline(layout=pipeline_layout,
#                                                         compute={"module": cshader, "entry_point": "space_sim2"})
#     compute_time_sim = device.create_compute_pipeline(layout=pipeline_layout,
#                                                       compute={"module": cshader, "entry_point": "time_sim"})
#     compute_incr_k = device.create_compute_pipeline(layout=pipeline_layout,
#                                                     compute={"module": cshader, "entry_point": "incr_k"})
#
#     # Configuracao e inicializacao da janela de exibicao
#     App = pg.QtWidgets.QApplication([])
#     window = Window()
#
#     # Laco de tempo para execucao da simulacao
#     for it in range(NSTEP):
#         # Cria o codificador de comandos
#         command_encoder = device.create_command_encoder()
#
#         # Inicia os passos de execucao do decodificador
#         compute_pass = command_encoder.begin_compute_pass()
#
#         # Ajusta os grupos de amarracao
#         compute_pass.set_bind_group(0, bg_0, [], 0, 999999)  # last 2 elements not used
#         compute_pass.set_bind_group(1, bg_1, [], 0, 999999)  # last 2 elements not used
#         compute_pass.set_bind_group(2, bg_2, [], 0, 999999)  # last 2 elements not used
#
#         # Ativa o pipeline de execucao da simulacao no espaco (calculo da primeira derivada espacial)
#         compute_pass.set_pipeline(compute_space_sim1)
#         compute_pass.dispatch_workgroups(ny // wsy, nx // wsx)
#
#         # Ativa o pipeline de execucao da simulacao no espaco (calculo da segunda derivada espacial)
#         compute_pass.set_pipeline(compute_space_sim2)
#         compute_pass.dispatch_workgroups(ny // wsy, nx // wsx)
#
#         # Ativa o pipeline de execucao da simulacao no tempo (calculo das derivadas temporais)
#         compute_pass.set_pipeline(compute_time_sim)
#         compute_pass.dispatch_workgroups(ny // wsy, nx // wsx)
#
#         # Ativa o pipeline de atualizacao da amostra de tempo
#         compute_pass.set_pipeline(compute_incr_k)
#         compute_pass.dispatch_workgroups(1)
#
#         # Termina o passo de execucao
#         compute_pass.end()
#
#         # Efetua a execucao dos comandos na GPU
#         device.queue.submit([command_encoder.finish()])
#
#         # Pega o resultado do campo de pressao
#         if (it % 10) == 0:
#             out = device.queue.read_buffer(b_p_0).cast("f")  # reads from buffer 3
#             window.imv.setImage(np.asarray(out).reshape((ny, nx)).T, levels=[-1.0, 1.0])
#             App.processEvents()
#
#     App.exit()
#
#     # Pega o sinal do sensor
#     sens = np.array(device.queue.read_buffer(b_sens).cast("f"))
#     adapter_info = device.adapter.request_adapter_info()
#     return sens, adapter_info["device"]


times_webgpu = list()
times_cpu = list()

# WebGPU
# if do_sim_gpu:
#     for n in range(n_iter_gpu):
#         print(f'Simulacao WEBGPU')
#         print(f'Iteracao {n}')
#         t_webgpu = time()
#         sensor, gpu_str = sim_webgpu()
#         times_webgpu.append(time() - t_webgpu)
#         print(gpu_str)
#         print(f'{times_webgpu[-1]:.3}s')

# CPU
if do_sim_cpu:
    for n in range(n_iter_cpu):
        print(f'SIMULAÇÃO CPU')
        print(f'Iteracao {n}')
        t_ser = time()
        sim_cpu()
        times_cpu.append(time() - t_ser)
        print(f'{times_cpu[-1]:.3}s')

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
