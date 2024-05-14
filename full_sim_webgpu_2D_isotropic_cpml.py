import wgpu

if wgpu.version_info[1] > 11:
    import wgpu.backends.wgpu_native  # Select backend 0.13.X
else:
    import wgpu.backends.rs  # Select backend 0.9.5

from datetime import datetime
import numpy as np
import ast
import matplotlib.pyplot as plt
from time import time
from PyQt6.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from simul_utils import SimulationROI, SimulationProbeLinearArray

# ==========================================================
# Esse arquivo contem as simulacoes realizadas dentro da GPU.
# ==========================================================
flt32 = np.float32


# -----------------------------------------------
# Codigo para visualizacao da janela de simulacao
# -----------------------------------------------
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
            self.setWindowTitle(f"{nx}x{ny} Grid x {NSTEP} iterations - dx = {dx} m x dy = {dy} m x dt = {dt} s")
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
        self.image = np.random.normal(size=(geometry[2], geometry[3]))

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


# --------------------------
# Funcao do simulador em CPU
# --------------------------
def sim_cpu():
    global simul_probe, coefs
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
    global v_solid_norm, v_2
    global windows_cpu

    DELTAT_over_rho = flt32(dt / rho)
    _ord = coefs.shape[0]
    idx_fd = np.array([[c + _ord,  # ini half grid
                        -c + _ord - 1,  # ini full grid
                        c - _ord + 1,  # fin half grid
                        -c - _ord]  # fin full grid
                       for c in range(_ord)], dtype=np.int32)

    v_max = 100.0
    v_min = - v_max
    ix_min = simul_roi.get_ix_min()
    ix_max = simul_roi.get_ix_max()
    iy_min = simul_roi.get_iz_min()
    iy_max = simul_roi.get_iz_max()

    # Source terms
    source_term, idx_src = simul_probe.get_source_term(samples=NSTEP, dt=dt, sim_roi=simul_roi, simul_type="2D")

    # Inicio do laco de tempo
    for it in range(1, NSTEP + 1):
        # Calculo da tensao [stress] - {sigma} (equivalente a pressao nos gases-liquidos)
        # sigma_ii -> tensoes normais; sigma_ij -> tensoes cisalhantes
        # Primeiro "laco" i: 1,NX-1; j: 2,NY -> [1:-2, 2:-1]
        i_dix = idx_fd[0, 1]
        i_dfx = idx_fd[0, 3]
        i_diy = idx_fd[0, 0]
        i_dfy = idx_fd[0, 2]
        for c in range(_ord):
            # Eixo "x"
            i_iax = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fax = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_ibx = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fbx = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            # eixo "y"
            i_iay = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fay = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_iby = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fby = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            if c:
                value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] += \
                    (coefs[c] * (vx[i_iax:i_fax, i_diy:i_dfy] - vx[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] += \
                    (coefs[c] * (vy[i_dix:i_dfx, i_iay:i_fay] - vy[i_dix:i_dfx, i_iby:i_fby]) * one_dy)
            else:
                value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = \
                    (coefs[c] * (vx[i_iax:i_fax, i_diy:i_dfy] - vx[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = \
                    (coefs[c] * (vy[i_dix:i_dfx, i_iay:i_fay] - vy[i_dix:i_dfx, i_iby:i_fby]) * one_dy)

        memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = (b_x_half[:-1, :] * memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] +
                                                   a_x_half[:-1, :] * value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy])
        memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = (b_y[:, 1:] * memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] +
                                                   a_y[:, 1:] * value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy])

        value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] = (value_dvx_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x_half[:-1, :] +
                                                  memory_dvx_dx[i_dix:i_dfx, i_diy:i_dfy])
        value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] = (value_dvy_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y[:, 1:] +
                                                  memory_dvy_dy[i_dix:i_dfx, i_diy:i_dfy])

        # compute the stress using the Lame parameters
        sigmaxx = sigmaxx + (lambdaplus2mu * value_dvx_dx + lambda_ * value_dvy_dy) * dt
        sigmayy = sigmayy + (lambda_ * value_dvx_dx + lambdaplus2mu * value_dvy_dy) * dt

        # Segundo "laco" i: 2,NX; j: 1,NY-1 -> [2:-1, 1:-2]
        i_dix = idx_fd[0, 0]
        i_dfx = idx_fd[0, 2]
        i_diy = idx_fd[0, 1]
        i_dfy = idx_fd[0, 3]
        for c in range(_ord):
            # Eixo "x"
            i_iax = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fax = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_ibx = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fbx = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            # eixo "y"
            i_iay = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fay = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_iby = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fby = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            if c:
                value_dvy_dx[i_dix:i_dfx, i_diy:i_dfy] += \
                    (coefs[c] * (vy[i_iax:i_fax, i_diy:i_dfy] - vy[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dvx_dy[i_dix:i_dfx, i_diy:i_dfy] += \
                    (coefs[c] * (vx[i_dix:i_dfx, i_iay:i_fay] - vx[i_dix:i_dfx, i_iby:i_fby]) * one_dy)
            else:
                value_dvy_dx[i_dix:i_dfx, i_diy:i_dfy] = \
                    (coefs[c] * (vy[i_iax:i_fax, i_diy:i_dfy] - vy[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dvx_dy[i_dix:i_dfx, i_diy:i_dfy] = \
                    (coefs[c] * (vx[i_dix:i_dfx, i_iay:i_fay] - vx[i_dix:i_dfx, i_iby:i_fby]) * one_dy)

        memory_dvy_dx[i_dix:i_dfx, i_diy:i_dfy] = (b_x[1:, :] * memory_dvy_dx[i_dix:i_dfx, i_diy:i_dfy] +
                                                   a_x[1:, :] * value_dvy_dx[i_dix:i_dfx, i_diy:i_dfy])
        memory_dvx_dy[i_dix:i_dfx, i_diy:i_dfy] = (b_y_half[:, :-1] * memory_dvx_dy[i_dix:i_dfx, i_diy:i_dfy] +
                                                   a_y_half[:, :-1] * value_dvx_dy[i_dix:i_dfx, i_diy:i_dfy])

        value_dvy_dx[i_dix:i_dfx, i_diy:i_dfy] = (value_dvy_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x[1:, :] +
                                                  memory_dvy_dx[i_dix:i_dfx, i_diy:i_dfy])
        value_dvx_dy[i_dix:i_dfx, i_diy:i_dfy] = (value_dvx_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y_half[:, :-1] +
                                                  memory_dvx_dy[i_dix:i_dfx, i_diy:i_dfy])

        # compute the stress using the Lame parameters
        sigmaxy = sigmaxy + dt * mu * (value_dvx_dy + value_dvy_dx)

        # Calculo da velocidade
        # Primeiro "laco" i: 2,NX; j: 2,NY -> [2:-1, 2:-1]
        i_dix = idx_fd[0, 0]
        i_dfx = idx_fd[0, 2]
        i_diy = idx_fd[0, 0]
        i_dfy = idx_fd[0, 2]
        for c in range(_ord):
            # Eixo "x"
            i_iax = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fax = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_ibx = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fbx = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            # eixo "y"
            i_iay = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fay = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_iby = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fby = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            if c:
                value_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy] += \
                    (coefs[c] * (sigmaxx[i_iax:i_fax, i_diy:i_dfy] - sigmaxx[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy] += \
                    (coefs[c] * (sigmaxy[i_dix:i_dfx, i_iay:i_fay] - sigmaxy[i_dix:i_dfx, i_iby:i_fby]) * one_dy)
            else:
                value_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy] = \
                    (coefs[c] * (sigmaxx[i_iax:i_fax, i_diy:i_dfy] - sigmaxx[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy] = \
                    (coefs[c] * (sigmaxy[i_dix:i_dfx, i_iay:i_fay] - sigmaxy[i_dix:i_dfx, i_iby:i_fby]) * one_dy)

        memory_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy] = (b_x[1:, :] * memory_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy] +
                                                        a_x[1:, :] * value_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy])
        memory_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy] = (b_y[:, 1:] * memory_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy] +
                                                        a_y[:, 1:] * value_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy])

        value_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy] = (value_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x[1:, :] +
                                                       memory_dsigmaxx_dx[i_dix:i_dfx, i_diy:i_dfy])
        value_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy] = (value_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y[:, 1:] +
                                                       memory_dsigmaxy_dy[i_dix:i_dfx, i_diy:i_dfy])

        vx = DELTAT_over_rho * (value_dsigmaxx_dx + value_dsigmaxy_dy) + vx

        # segunda parte:  i: 1,NX-1; j: 1,NY-1 -> [1:-2, 1:-2]
        i_dix = idx_fd[0, 1]
        i_dfx = idx_fd[0, 3]
        i_diy = idx_fd[0, 1]
        i_dfy = idx_fd[0, 3]
        for c in range(_ord):
            # Eixo "x"
            i_iax = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fax = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_ibx = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fbx = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            # eixo "y"
            i_iay = None if idx_fd[c, 0] == 0 else idx_fd[c, 0]
            i_fay = None if idx_fd[c, 2] == 0 else idx_fd[c, 2]
            i_iby = None if idx_fd[c, 1] == 0 else idx_fd[c, 1]
            i_fby = None if idx_fd[c, 3] == 0 else idx_fd[c, 3]
            if c:
                value_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy] += (
                        coefs[c] * (sigmaxy[i_iax:i_fax, i_diy:i_dfy] - sigmaxy[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy] += (
                        coefs[c] * (sigmayy[i_dix:i_dfx, i_iay:i_fay] - sigmayy[i_dix:i_dfx, i_iby:i_fby]) * one_dy)
            else:
                value_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                        coefs[c] * (sigmaxy[i_iax:i_fax, i_diy:i_dfy] - sigmaxy[i_ibx:i_fbx, i_diy:i_dfy]) * one_dx)
                value_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                        coefs[c] * (sigmayy[i_dix:i_dfx, i_iay:i_fay] - sigmayy[i_dix:i_dfx, i_iby:i_fby]) * one_dy)

        memory_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy] = (
                b_x_half[:-1, :] * memory_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy] +
                a_x_half[:-1, :] * value_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy])
        memory_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy] = (
                b_y_half[:, :-1] * memory_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy] +
                a_y_half[:, :-1] * value_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy])

        value_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy] = (value_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy] / k_x_half[:-1, :] +
                                                       memory_dsigmaxy_dx[i_dix:i_dfx, i_diy:i_dfy])
        value_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy] = (value_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy] / k_y_half[:, :-1] +
                                                       memory_dsigmayy_dy[i_dix:i_dfx, i_diy:i_dfy])

        vy = DELTAT_over_rho * (value_dsigmaxy_dx + value_dsigmayy_dy) + vy

        # add the source (force vector located at a given grid point)
        for _isrc in range(NSRC):
            vy[ix_src[_isrc], iy_src[_isrc]] += source_term[it - 1, idx_src[_isrc]] * dt / rho

        # implement Dirichlet boundary conditions on the six edges of the grid
        # which is the right condition to implement in order for C-PML to remain stable at long times
        # xmin
        vx[:_ord, :] = ZERO
        vy[:_ord, :] = ZERO

        # xmax
        vx[-_ord:, :] = ZERO
        vy[-_ord:, :] = ZERO

        # ymin
        vx[:, :_ord] = ZERO
        vy[:, :_ord] = ZERO

        # ymax
        vx[:, -_ord:] = ZERO
        vy[:, -_ord:] = ZERO

        # Store seismograms
        for _irec in range(NREC):
            sisvx[it - 1, _irec] = vx[ix_rec[_irec], iy_rec[_irec]]
            sisvy[it - 1, _irec] = vy[ix_rec[_irec], iy_rec[_irec]]

        v_2 = vx[:, :] ** 2 + vy[:, :] ** 2
        v_solid_norm[it - 1] = np.sqrt(np.max(v_2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            if show_debug:
                print(f'Time step # {it} out of {NSTEP}')
                print(f'Max Vx = {np.max(vx)}, Vy = {np.max(vy)}')
                print(f'Min Vx = {np.min(vx)}, Vy = {np.min(vy)}')
                print(f'Max norm velocity vector V (m/s) = {v_solid_norm[it - 1]}')

            if show_anim:
                windows_cpu[0].imv.setImage(vx[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                windows_cpu[1].imv.setImage(vy[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                windows_cpu[2].imv.setImage(vx[ix_min:ix_max, iy_min:iy_max] + vy[ix_min:ix_max, iy_min:iy_max],
                                            levels=[2.0 * v_min, 2.0 * v_max])

                App.processEvents()

        # Verifica a estabilidade da simulacao
        if v_solid_norm[it - 1] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instavel")
            exit(2)


# -----------------------------
# Funcao do simulador em WebGPU
# -----------------------------
def sim_webgpu(device):
    global simul_probe, coefs
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
    global ix_src, iy_src, ix_rec, iy_rec
    global v_2, v_solid_norm
    global simul_roi
    global windows_gpu

    # Arrays com parametros inteiros (i32) e ponto flutuante (f32) para rodar o simulador
    _ord = coefs.shape[0]
    params_i32 = np.array([nx, ny, NSTEP, simul_probe.num_elem, NSRC, NREC, _ord, 0], dtype=np.int32)
    params_f32 = np.array([cp, cs, dx, dy, dt, rho, lambda_, mu, lambdaplus2mu], dtype=flt32)

    # Source terms
    source_term, idx_src = simul_probe.get_source_term(samples=NSTEP, dt=dt, sim_roi=simul_roi, simul_type="2D")
    pos_sources = -np.ones((nx, ny), dtype=np.int32)
    pos_sources[ix_src, iy_src] = idx_src.astype(np.int32)

    # Cria o shader para calculo contido no arquivo ``shader_2D_elast_cpml.wgsl''
    with open('shader_2D_elast_cpml.wgsl') as shader_file:
        cshader_string = shader_file.read()
        cshader_string = cshader_string.replace('wsx', f'{wsx}')
        cshader_string = cshader_string.replace('wsy', f'{wsy}')
        cshader = device.create_shader_module(code=cshader_string)

    # Definicao dos buffers que terao informacoes compartilhadas entre CPU e GPU
    # ------- Buffers para o binding de parametros -------------
    # Buffer de parametros com valores em ponto flutuante
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 0
    b_param_flt32 = device.create_buffer_with_data(data=params_f32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)

    # Forcas da fonte
    # Binding 1
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_force = device.create_buffer_with_data(data=source_term,
                                             usage=wgpu.BufferUsage.STORAGE |
                                                   wgpu.BufferUsage.COPY_SRC)

    # Binding 24
    b_idx_src = device.create_buffer_with_data(data=pos_sources, usage=wgpu.BufferUsage.STORAGE |
                                                                       wgpu.BufferUsage.COPY_SRC)

    # Coeficientes de absorcao
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 2
    b_coef_x = device.create_buffer_with_data(data=np.column_stack((a_x.flatten(),
                                                                    b_x.flatten(),
                                                                    k_x.flatten(),
                                                                    a_x_half.flatten(),
                                                                    b_x_half.flatten(),
                                                                    k_x_half.flatten())),
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_SRC)

    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 3
    b_coef_y = device.create_buffer_with_data(data=np.column_stack((a_y.flatten(),
                                                                    b_y.flatten(),
                                                                    k_y.flatten(),
                                                                    a_y_half.flatten(),
                                                                    b_y_half.flatten(),
                                                                    k_y_half.flatten())),
                                              usage=wgpu.BufferUsage.STORAGE |
                                                    wgpu.BufferUsage.COPY_SRC)

    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 4
    b_param_int32 = device.create_buffer_with_data(data=params_i32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)

    # Buffers com os indices para o calculo das derivadas com acuracia maior
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 25
    idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(_ord)], dtype=np.int32)
    b_idx_fd = device.create_buffer_with_data(data=idx_fd, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Buffer com os coeficientes para ao calculo das derivadas
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    # Binding 28
    b_fd_coeffs = device.create_buffer_with_data(data=coefs, usage=wgpu.BufferUsage.STORAGE |
                                                                   wgpu.BufferUsage.COPY_SRC)

    # Buffers com os arrays de simulacao
    # Velocidades
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    # Binding 5
    b_vel = device.create_buffer_with_data(data=np.vstack((vx, vy, v_2)), usage=wgpu.BufferUsage.STORAGE |
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
    b_sens_x = device.create_buffer_with_data(data=sisvx, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)

    # Binding 9
    b_sens_y = device.create_buffer_with_data(data=sisvy, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)

    # Binding 10
    b_sens_pos_x = device.create_buffer_with_data(data=ix_rec, usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)

    # Binding 11
    b_sens_pos_y = device.create_buffer_with_data(data=iy_rec, usage=wgpu.BufferUsage.STORAGE |
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
    bl_params += [{
        "binding": 4,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage}
        },
        {"binding": 24,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 25,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 28,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         }
    ]

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
        {"binding": 8,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 9,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 10,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 11,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         }
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
        {
            "binding": 24,
            "resource": {"buffer": b_idx_src, "offset": 0, "size": b_idx_src.size},
        },
        {
            "binding": 25,
            "resource": {"buffer": b_idx_fd, "offset": 0, "size": b_idx_fd.size},
        },
        {
            "binding": 28,
            "resource": {"buffer": b_fd_coeffs, "offset": 0, "size": b_fd_coeffs.size},
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
            "resource": {"buffer": b_sens_x, "offset": 0, "size": b_sens_x.size},
        },
        {
            "binding": 9,
            "resource": {"buffer": b_sens_y, "offset": 0, "size": b_sens_y.size},
        },
        {
            "binding": 10,
            "resource": {"buffer": b_sens_pos_x, "offset": 0, "size": b_sens_pos_x.size},
        },
        {
            "binding": 11,
            "resource": {"buffer": b_sens_pos_y, "offset": 0, "size": b_sens_pos_y.size},
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
    compute_teste_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                          compute={"module": cshader,
                                                                   "entry_point": "teste_kernel"})
    compute_sigma_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                          compute={"module": cshader,
                                                                   "entry_point": "sigma_kernel"})
    compute_velocity_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                             compute={"module": cshader,
                                                                      "entry_point": "velocity_kernel"})
    compute_sources_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                            compute={"module": cshader,
                                                                     "entry_point": "sources_kernel"})
    compute_finish_it_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                              compute={"module": cshader,
                                                                       "entry_point": "finish_it_kernel"})
    compute_incr_it_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                            compute={"module": cshader,
                                                                     "entry_point": "incr_it_kernel"})

    v_max = 100.0
    v_min = - v_max
    v_sol_n = np.zeros(NSTEP, dtype=flt32)
    ix_min = simul_roi.get_ix_min()
    ix_max = simul_roi.get_ix_max()
    iy_min = simul_roi.get_iz_min()
    iy_max = simul_roi.get_iz_max()
    # Laco de tempo para execucao da simulacao
    for it in range(1, NSTEP + 1):
        # Cria o codificador de comandos
        command_encoder = device.create_command_encoder()

        # Inicia os passos de execucao do decodificador
        compute_pass = command_encoder.begin_compute_pass()

        # Ajusta os grupos de amarracao
        compute_pass.set_bind_group(0, bg_0, [], 0, 999999)  # last 2 elements not used
        compute_pass.set_bind_group(1, bg_1, [], 0, 999999)  # last 2 elements not used
        compute_pass.set_bind_group(2, bg_2, [], 0, 999999)  # last 2 elements not used

        # Ativa o pipeline de teste
        # compute_pass.set_pipeline(compute_teste_kernel)
        # compute_pass.dispatch_workgroups(nx // wsx, ny // wsy)

        # Ativa o pipeline de execucao do calculo dos estresses
        compute_pass.set_pipeline(compute_sigma_kernel)
        compute_pass.dispatch_workgroups(nx // wsx, ny // wsy)

        # Ativa o pipeline de execucao do calculo das velocidades
        compute_pass.set_pipeline(compute_velocity_kernel)
        compute_pass.dispatch_workgroups(nx // wsx, ny // wsy)

        # Ativa o pipeline de adicao dos termos de fonte
        compute_pass.set_pipeline(compute_sources_kernel)
        compute_pass.dispatch_workgroups(nx // wsx, ny // wsy)

        # Ativa o pipeline de execucao dos procedimentos finais da iteracao
        compute_pass.set_pipeline(compute_finish_it_kernel)
        compute_pass.dispatch_workgroups(nx // wsx, ny // wsy)

        # Ativa o pipeline de atualizacao da amostra de tempo
        compute_pass.set_pipeline(compute_incr_it_kernel)
        compute_pass.dispatch_workgroups(1)

        # Termina o passo de execucao
        compute_pass.end()

        # Efetua a execucao dos comandos na GPU
        device.queue.submit([command_encoder.finish()])

        # en = np.asarray(device.queue.read_buffer(b_energy).cast("f")).reshape((NSTEP, 4))
        b_v_2_offset = (vx.size + vy.size) * 4
        vsn2 = np.asarray(device.queue.read_buffer(b_vel,
                                                   buffer_offset=b_v_2_offset).cast("f")).reshape((nx, ny))
        v_sol_n[it - 1] = np.sqrt(np.max(vsn2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            if show_debug or show_anim:
                vxgpu = np.asarray(device.queue.read_buffer(b_vel,
                                                            buffer_offset=0,
                                                            size=vx.size * 4).cast("f")).reshape((nx, ny))
                vygpu = np.asarray(device.queue.read_buffer(b_vel,
                                                            buffer_offset=vx.size * 4,
                                                            size=vy.size * 4).cast("f")).reshape((nx, ny))

                if show_debug:
                    print(f'Time step # {it} out of {NSTEP}')
                    print(f'Max Vx = {np.max(vxgpu)}, Vy = {np.max(vygpu)}')
                    print(f'Min Vx = {np.min(vxgpu)}, Vy = {np.min(vygpu)}')
                    print(f'Max norm velocity vector V (m/s) = {v_sol_n[it - 1]}')

                if show_anim:
                    windows_gpu[0].imv.setImage(vxgpu[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    windows_gpu[1].imv.setImage(vygpu[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                    windows_gpu[2].imv.setImage(
                        vxgpu[ix_min:ix_max, iy_min:iy_max] + vygpu[ix_min:ix_max, iy_min:iy_max],
                        levels=[2.0 * v_min, 2.0 * v_max])

                    App.processEvents()

        # Verifica a estabilidade da simulacao
        if v_sol_n[it - 1] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instavel")
            exit(2)

    # Pega os resultados da simulacao
    vxgpu = np.asarray(device.queue.read_buffer(b_vel,
                                                buffer_offset=0,
                                                size=vx.size * 4).cast("f")).reshape((nx, ny))
    vygpu = np.asarray(device.queue.read_buffer(b_vel,
                                                buffer_offset=vx.size * 4,
                                                size=vy.size * 4).cast("f")).reshape((nx, ny))
    sens_vx = np.array(device.queue.read_buffer(b_sens_x).cast("f")).reshape((NSTEP, NREC))
    sens_vy = np.array(device.queue.read_buffer(b_sens_y).cast("f")).reshape((NSTEP, NREC))
    adapter_info = device.adapter.request_adapter_info()
    return vxgpu, vygpu, sens_vx, sens_vy, v_sol_n, adapter_info["device"]


# ----------------------------------------------------------
# Aqui comeca o codigo principal de execucao dos simuladores
# ----------------------------------------------------------
# Constantes
PI = flt32(np.pi)
DEGREES_TO_RADIANS = flt32(PI / 180.0)
ZERO = flt32(0.0)
STABILITY_THRESHOLD = flt32(1.0e25)  # Limite para considerar que a simulacao esta instavel

# Definicao das constantes para a o calculo das derivadas, seguindo Lui 2009 (10.1111/j.1365-246X.2009.04305.x)
coefs_Lui = [
    [9.0 / 8.0, -1.0 / 24.0],
    [75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0],
    [1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0],
    [19845.0 / 16384.0, -735.0 / 8192.0, 567.0 / 40960.0, -405.0 / 229376.0, 35.0 / 294912.0],
    [160083.0 / 131072.0, -12705.0 / 131072.0, 22869.0 / 1310720.0, -5445.0 / 1835008.0, 847.0 / 2359296.0,
     -63.0 / 2883584.0]
]

# -----------------------
# Leitura da configuracao no formato JSON
# -----------------------
with open('config.json', 'r') as f:
    configs = ast.literal_eval(f.read())
    data_rec = np.array(configs["receivers"])
    coefs = np.array(coefs_Lui[configs["simul_params"]["ord"] - 2], dtype=flt32)
    simul_roi = SimulationROI(**configs["roi"], pad=coefs.shape[0] - 1)
    if "linear" in configs["probe"]:
        simul_probe = SimulationProbeLinearArray(**configs["probe"]["linear"])
    print(f'Ordem da acuracia: {coefs.shape[0] * 2}')

    # Configuracao dos ensaios
    n_iter_gpu = configs["simul_configs"]["n_iter_gpu"]
    n_iter_cpu = configs["simul_configs"]["n_iter_cpu"]
    do_sim_gpu = bool(configs["simul_configs"]["do_sim_gpu"])
    do_sim_cpu = bool(configs["simul_configs"]["do_sim_cpu"])
    do_comp_fig_cpu_gpu = bool(configs["simul_configs"]["do_comp_fig_cpu_gpu"])
    use_refletors = bool(configs["simul_configs"]["use_refletors"])
    show_anim = bool(configs["simul_configs"]["show_anim"])
    show_debug = bool(configs["simul_configs"]["show_debug"])
    plot_results = bool(configs["simul_configs"]["plot_results"])
    plot_sensors = bool(configs["simul_configs"]["plot_sensors"])
    show_results = bool(configs["simul_configs"]["show_results"])
    save_results = bool(configs["simul_configs"]["save_results"])
    gpu_type = configs["simul_configs"]["gpu_type"]

# -----------------------
# Inicializacao do WebGPU
# -----------------------
device_gpu = None
if do_sim_gpu:
    # =====================
    # webgpu configurations
    if gpu_type == "high-perf":
        device_gpu = wgpu.utils.get_default_device()
    else:
        if wgpu.version_info[1] > 11:
            adapter = wgpu.gpu.request_adapter(power_preference="low-power")  # 0.13.X
        else:
            adapter = wgpu.request_adapter(canvas=None, power_preference="low-power")  # 0.9.5

        device_gpu = adapter.request_device()

    # Escolha dos valores de wsx, wsy e wsz (GPU)
    wsx = np.gcd(simul_roi.get_nx(), 16)
    wsy = np.gcd(simul_roi.get_nz(), 16)

# Parametros da simulacao
nx = simul_roi.get_nx()
ny = simul_roi.get_nz()

# Escala do grid (valor do passo no espaco em milimetros)
dx = flt32(simul_roi.w_step)
dy = flt32(simul_roi.h_step)
one_dx = flt32(1.0 / dx)
one_dy = flt32(1.0 / dy)

# Velocidades do som e densidade do meio
cp = flt32(configs["specimen_params"]["cp"])  # [mm/us]
cs = flt32(configs["specimen_params"]["cs"])  # [mm/us]
rho = flt32(configs["specimen_params"]["rho"])
mu = flt32(rho * cs * cs)
lambda_ = flt32(rho * (cp * cp - 2.0 * cs * cs))
lambdaplus2mu = flt32(rho * cp * cp)

# Numero total de passos de tempo
NSTEP = configs["simul_params"]["time_steps"]

# Passo de tempo em microssegundos
dt = flt32(configs["simul_params"]["dt"])

# Numero de iteracoes de tempo para apresentar e armazenar informacoes
IT_DISPLAY = configs["simul_params"]["it_display"]

# Define a posicao das fontes
i_src = simul_probe.get_points_roi(simul_roi, simul_type="2d")
ix_src = i_src[:, 0].astype(np.int32)
iy_src = i_src[:, 2].astype(np.int32)
NSRC = i_src.shape[0]

# Define a localizacao dos receptores
NREC = data_rec.shape[0]
i_rec = np.array([simul_roi.get_nearest_grid_idx(p[0:3]) for p in data_rec])
ix_rec = i_rec[:, 0].astype(np.int32)
iy_rec = i_rec[:, 2].astype(np.int32)

# for evolution of total energy in the medium
v_2 = np.zeros((nx, ny), dtype=flt32)
v_solid_norm = np.zeros(NSTEP, dtype=flt32)

# Arrays para as variaveis de memoria do calculo
memory_dvx_dx = np.zeros((nx, ny), dtype=flt32)
memory_dvx_dy = np.zeros((nx, ny), dtype=flt32)
memory_dvy_dx = np.zeros((nx, ny), dtype=flt32)
memory_dvy_dy = np.zeros((nx, ny), dtype=flt32)
memory_dsigmaxx_dx = np.zeros((nx, ny), dtype=flt32)
memory_dsigmayy_dy = np.zeros((nx, ny), dtype=flt32)
memory_dsigmaxy_dx = np.zeros((nx, ny), dtype=flt32)
memory_dsigmaxy_dy = np.zeros((nx, ny), dtype=flt32)

value_dvx_dx = np.zeros((nx, ny), dtype=flt32)
value_dvx_dy = np.zeros((nx, ny), dtype=flt32)
value_dvy_dx = np.zeros((nx, ny), dtype=flt32)
value_dvy_dy = np.zeros((nx, ny), dtype=flt32)
value_dsigmaxx_dx = np.zeros((nx, ny), dtype=flt32)
value_dsigmayy_dy = np.zeros((nx, ny), dtype=flt32)
value_dsigmaxy_dx = np.zeros((nx, ny), dtype=flt32)
value_dsigmaxy_dy = np.zeros((nx, ny), dtype=flt32)

# Arrays dos campos de velocidade e tensoes
vx = np.zeros((nx, ny), dtype=flt32)
vy = np.zeros((nx, ny), dtype=flt32)
sigmaxx = np.zeros((nx, ny), dtype=flt32)
sigmayy = np.zeros((nx, ny), dtype=flt32)
sigmaxy = np.zeros((nx, ny), dtype=flt32)

# Total de arrays
N_ARRAYS = 5 + 2 * 4

print(f'2D elastic finite-difference code in velocity and stress formulation with C-PML')
print(f'NX = {nx}')
print(f'NY = {ny}')
print(f'Total de pontos no grid = {nx * ny}')
print(f'Number of points of all the arrays = {nx * ny * N_ARRAYS}')
print(f'Size in GB of all the arrays = {nx * ny * N_ARRAYS * 4 / (1024 * 1024 * 1024)}\n')

# Valor da potencia para calcular "d0"
NPOWER = flt32(configs["simul_params"]["npower"])
if NPOWER < 1:
    raise ValueError('NPOWER deve ser maior que 1')

# Coeficiente de reflexao e calculo de d0 do relatorio da INRIA section 6.1
# http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
rcoef = flt32(configs["simul_params"]["rcoef"])
d0_x = flt32(-(NPOWER + 1) * cp * np.log(rcoef) / simul_roi.get_pml_thickness_x())
d0_y = flt32(-(NPOWER + 1) * cp * np.log(rcoef) / simul_roi.get_pml_thickness_z())

print(f'd0_x = {d0_x}')
print(f'd0_y = {d0_y}')

# Calculo dos coeficientes de amortecimento para a PML
# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = flt32(configs["simul_params"]["k_max_pml"])
ALPHA_MAX_PML = flt32(2.0 * PI * (simul_probe.get_freq() / 2.0))  # from Festa and Vilotte

# Perfil de amortecimento na direcao "x" dentro do grid
a_x, b_x, k_x = simul_roi.calc_pml_array(axis='x', grid='f', dt=dt, d0=d0_x,
                                         npower=NPOWER, k_max=K_MAX_PML, alpha_max=ALPHA_MAX_PML)
a_x = np.expand_dims(a_x.astype(flt32), axis=1)
b_x = np.expand_dims(b_x.astype(flt32), axis=1)
k_x = np.expand_dims(k_x.astype(flt32), axis=1)

# Perfil de amortecimento na direcao "x" dentro do meio grid (staggered grid)
a_x_half, b_x_half, k_x_half = simul_roi.calc_pml_array(axis='x', grid='h', dt=dt, d0=d0_x,
                                                        npower=NPOWER, k_max=K_MAX_PML, alpha_max=ALPHA_MAX_PML)
a_x_half = np.expand_dims(a_x_half.astype(flt32), axis=1)
b_x_half = np.expand_dims(b_x_half.astype(flt32), axis=1)
k_x_half = np.expand_dims(k_x_half.astype(flt32), axis=1)

# Perfil de amortecimento na direcao "y" dentro do grid
a_y, b_y, k_y = simul_roi.calc_pml_array(axis='z', grid='f', dt=dt, d0=d0_y,
                                         npower=NPOWER, k_max=K_MAX_PML, alpha_max=ALPHA_MAX_PML)
a_y = np.expand_dims(a_y.astype(flt32), axis=0)
b_y = np.expand_dims(b_y.astype(flt32), axis=0)
k_y = np.expand_dims(k_y.astype(flt32), axis=0)

# Perfil de amortecimento na direcao "y" dentro do meio grid (staggered grid)
a_y_half, b_y_half, k_y_half = simul_roi.calc_pml_array(axis='z', grid='h', dt=dt, d0=d0_y,
                                                        npower=NPOWER, k_max=K_MAX_PML, alpha_max=ALPHA_MAX_PML)
a_y_half = np.expand_dims(a_y_half.astype(flt32), axis=0)
b_y_half = np.expand_dims(b_y_half.astype(flt32), axis=0)
k_y_half = np.expand_dims(k_y_half.astype(flt32), axis=0)

# Imprime a quantidade de fontes e receptores
print(f'Existem {NSRC} fontes')
print(f'Existem {NREC} receptores')

# Arrays para armazenamento dos sinais dos sensores
sisvx = np.zeros((NSTEP, NREC), dtype=flt32)
sisvy = np.zeros((NSTEP, NREC), dtype=flt32)

# Verifica a condicao de estabilidade de Courant
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
courant_number = flt32(cp * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2))
print(f'\nNumero de Courant e {courant_number}')
if courant_number > 1:
    print("O passo de tempo e muito longo, a simulacao sera instavel")
    exit(1)

# Listas para armazenamento de resultados (tempos de execucao e sinais nos sensores)
times_gpu = list()
times_cpu = list()
sensor_gpu_result = list()
sensor_cpu_result = list()

# Configuracao e inicializacao da janela de exibicao
if show_anim:
    App = pg.QtWidgets.QApplication([])
    if do_sim_cpu:
        x_pos = 200 + np.arange(3) * (nx + 50)
        y_pos = 500 + np.arange(3) * (ny + 50)
        windows_cpu_data = [
            {"title": "Vx [CPU]", "geometry": (x_pos[0], y_pos[0],
                                               simul_roi.get_nx(), simul_roi.get_nz())},
            {"title": "Vy [CPU]", "geometry": (x_pos[1], y_pos[0],
                                               simul_roi.get_nx(), simul_roi.get_nz())},
            {"title": "Vx + Vy [CPU]", "geometry": (x_pos[2], y_pos[0],
                                                    simul_roi.get_nx(), simul_roi.get_nz())},
        ]
        windows_cpu = [Window(title=data["title"], geometry=data["geometry"]) for data in windows_cpu_data]

    if do_sim_gpu:
        x_pos = 200 + np.arange(3) * (nx + 50)
        y_pos = 100 + np.arange(3) * (ny + 50)
        windows_gpu_data = [
            {"title": "Vx [GPU]", "geometry": (x_pos[0], y_pos[0],
                                               simul_roi.get_nx(), simul_roi.get_nz())},
            {"title": "Vy [GPU]", "geometry": (x_pos[1], y_pos[0],
                                               simul_roi.get_nx(), simul_roi.get_nz())},
            {"title": "Vx + Vy [GPU]", "geometry": (x_pos[2], y_pos[0],
                                                    simul_roi.get_nx(), simul_roi.get_nz())},
        ]
        windows_gpu = [Window(title=data["title"], geometry=data["geometry"]) for data in windows_gpu_data]
else:
    App = None

# WebGPU
if do_sim_gpu:
    for n in range(n_iter_gpu):
        print(f'Simulacao WEBGPU')
        print(f'wsx = {wsx}, wsy = {wsy}')
        print(f'Iteracao {n}')
        t_gpu = time()
        vx_gpu, vy_gpu, sensor_vx_gpu, sensor_vy_gpu, v_solid_norm_gpu, gpu_str = sim_webgpu(device_gpu)
        times_gpu.append(time() - t_gpu)
        print(gpu_str)
        print(f'{times_gpu[-1]:.3}s')

        # Plota as velocidades tomadas no sensores
        if plot_results and plot_sensors:
            for r in range(NREC):
                fig, ax = plt.subplots(3, sharex=True, sharey=True)
                fig.suptitle(f'Receptor {r + 1} [GPU]')
                ax[0].plot(sensor_vx_gpu[:, r])
                ax[0].set_title(r'$V_x$')
                ax[1].plot(sensor_vy_gpu[:, r])
                ax[1].set_title(r'$V_y$')
                ax[2].plot(sensor_vx_gpu[:, r] + sensor_vy_gpu[:, r], 'tab:orange')
                ax[2].set_title(r'$V_x + V_y$')
                sensor_gpu_result.append(fig)

            if show_results:
                plt.show(block=False)

# CPU
if do_sim_cpu:
    for n in range(n_iter_cpu):
        print(f'SIMULACAO CPU')
        print(f'Iteracao {n}')
        t_cpu = time()
        sim_cpu()
        times_cpu.append(time() - t_cpu)
        print(f'{times_cpu[-1]:.3}s')

        # Plota as velocidades tomadas no sensores
        if plot_results and plot_sensors:
            for r in range(NREC):
                sensor_cpu_result, ax = plt.subplots(3, sharex=True, sharey=True)
                sensor_cpu_result.suptitle(f'Receptor {r + 1} [CPU]')
                ax[0].plot(sisvx[:, r])
                ax[0].set_title(r'$V_x$')
                ax[1].plot(sisvy[:, r])
                ax[1].set_title(r'$V_y$')
                ax[2].plot(sisvx[:, r] + sisvy[:, r], 'tab:orange')
                ax[2].set_title(r'$V_x + V_y$')

            if show_results:
                plt.show(block=False)

if show_anim and App:
    App.exit()

times_gpu = np.array(times_gpu)
times_cpu = np.array(times_cpu)
if do_sim_gpu:
    print(f'workgroups X: {wsx}; workgroups Y: {wsy}')

print(f'TEMPO - {NSTEP} pontos de tempo')
if do_sim_gpu and n_iter_gpu > 5:
    print(f'GPU: {times_gpu[5:].mean():.3}s (std = {times_gpu[5:].std()})')

if do_sim_cpu and n_iter_cpu > 5:
    print(f'CPU: {times_cpu[5:].mean():.3}s (std = {times_cpu[5:].std()})')

if do_sim_gpu and do_sim_cpu:
    print(f'MSE entre as simulacoes [Vx]: {np.sum((vx_gpu - vx) ** 2) / vx.size}')
    print(f'MSE entre as simulacoes [Vy]: {np.sum((vy_gpu - vy) ** 2) / vy.size}')

if plot_results:
    if do_sim_gpu:
        vx_gpu_sim_result = plt.figure()
        plt.title(f'GPU simulation Vx\n[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
        plt.imshow(vx_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                   aspect='auto', cmap='gray',
                   extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                           simul_roi.h_points[-1], simul_roi.h_points[0]))
        plt.colorbar()

        vy_gpu_sim_result = plt.figure()
        plt.title(f'GPU simulation Vy\n[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
        plt.imshow(vy_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                   aspect='auto', cmap='gray',
                   extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                           simul_roi.h_points[-1], simul_roi.h_points[0])
                   )
        plt.colorbar()

    if do_sim_cpu:
        vx_cpu_sim_result = plt.figure()
        plt.title(f'CPU simulation Vx ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
        plt.imshow(vx[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                   aspect='auto', cmap='gray',
                   extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                           simul_roi.h_points[-1], simul_roi.h_points[0])
                   )
        plt.colorbar()

        vy_cpu_sim_result = plt.figure()
        plt.title(f'CPU simulation Vy ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
        plt.imshow(vy[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                   aspect='auto', cmap='gray',
                   extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                           simul_roi.h_points[-1], simul_roi.h_points[0]))
        plt.colorbar()

    if do_comp_fig_cpu_gpu and do_sim_cpu and do_sim_gpu:
        vx_comp_sim_result = plt.figure()
        plt.title(f'CPU vs GPU Vx ({gpu_type}) error simulation ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
        plt.imshow(vx[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T -
                   vx_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                   aspect='auto', cmap='gray',
                   extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                           simul_roi.h_points[-1], simul_roi.h_points[0]))
        plt.colorbar()

        vy_comp_sim_result = plt.figure()
        plt.title(f'CPU vs GPU Vy ({gpu_type}) error simulation ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
        plt.imshow(vy[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T -
                   vy_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                   simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                   aspect='auto', cmap='gray',
                   extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                           simul_roi.h_points[-1], simul_roi.h_points[0]))
        plt.colorbar()

if save_results:
    now = datetime.now()
    name = (f'results/result_2D_elast_CPML_{now.strftime("%Y%m%d-%H%M%S")}_'
            f'{simul_roi.get_len_x()}x{simul_roi.get_len_z()}_{NSTEP}_iter_')
    if plot_results:
        if do_sim_gpu:
            vx_gpu_sim_result.savefig(name + 'Vx_gpu_' + gpu_type + '.png')
            vy_gpu_sim_result.savefig(name + 'Vy_gpu_' + gpu_type + '.png')
            for s in range(NREC):
                sensor_gpu_result[s].savefig(name + f'_sensor_{s}_' + gpu_type + '.png')

        if do_sim_cpu:
            vx_cpu_sim_result.savefig(name + 'Vx_cpu.png')
            vy_cpu_sim_result.savefig(name + 'Vy_cpu.png')
            for s in range(NREC):
                sensor_cpu_result[s].savefig(name + f'_sensor_{s}_CPU.png')

        if do_comp_fig_cpu_gpu and do_sim_cpu and do_sim_gpu:
            vx_comp_sim_result.savefig(name + 'Vx_XY_comp_cpu_gpu_' + gpu_type + '.png')
            vy_comp_sim_result.savefig(name + 'Vy_XY_comp_cpu_gpu_' + gpu_type + '.png')

    np.savetxt(name + '_GPU_' + gpu_type + '.csv', times_gpu, '%10.3f', delimiter=',')
    np.savetxt(name + '_CPU.csv', times_cpu, '%10.3f', delimiter=',')
    with open(name + '_desc.txt', 'w') as f:
        f.write('Parametros do ensaio\n')
        f.write('--------------------\n')
        f.write('\n')
        f.write(f'Quantidade de iteracoes no tempo: {NSTEP}\n')
        f.write(f'Tamanho da ROI: {simul_roi.get_len_x()}x{simul_roi.get_len_z()}\n')
        f.write(f'Refletores na ROI: {"Sim" if use_refletors else "Nao"}\n')
        f.write(f'Simulacao GPU: {"Sim" if do_sim_gpu else "Nao"}\n')
        if do_sim_gpu:
            f.write(f'GPU: {gpu_str}\n')
            f.write(f'Numero de simulacoes GPU: {n_iter_gpu}\n')
            if n_iter_gpu > 5:
                f.write(f'Tempo medio de execucao: {times_gpu[5:].mean():.3}s\n')
                f.write(f'Desvio padrao: {times_gpu[5:].std()}\n')
            else:
                f.write(f'Tempo execucao: {times_gpu[0]:.3}s\n')

        f.write(f'Simulacao CPU: {"Sim" if do_sim_cpu else "Nao"}\n')
        if do_sim_cpu:
            f.write(f'Numero de simulacoes CPU: {n_iter_cpu}\n')
            if n_iter_cpu > 5:
                f.write(f'Tempo medio de execucao: {times_cpu[5:].mean():.3}s\n')
                f.write(f'Desvio padrao: {times_cpu[5:].std()}\n')
            else:
                f.write(f'Tempo execucao: {times_cpu[0]:.3}s\n')

        if do_sim_gpu and do_sim_cpu:
            f.write(f'MSE entre as simulacoes [Vx]: {np.sum((vx_gpu - vx) ** 2) / vx.size}\n')
            f.write(f'MSE entre as simulacoes [Vy]: {np.sum((vy_gpu - vy) ** 2) / vy.size}\n')

if show_results:
    plt.show()
