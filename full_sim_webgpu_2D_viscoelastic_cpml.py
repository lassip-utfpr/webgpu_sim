import wgpu

if wgpu.version_info[1] > 11:
    import wgpu.backends.wgpu_native  # Select backend 0.13.X
else:
    import wgpu.backends.rs  # Select backend 0.9.5

from datetime import datetime
import numpy as np
import argparse
import ast
import matplotlib.pyplot as plt
from time import time
from PyQt6.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from simul_utils import SimulationROI, SimulationProbeLinearArray, SimulationProbePoint
import os.path
import file_law

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
    global simul_probes, coefs
    global a_x, a_x_half, b_x, b_x_half, k_x, k_x_half
    global a_y, a_y_half, b_y, b_y_half, k_y, k_y_half
    global vx, vy, sigmaxx, sigmayy, sigmaxy
    global r_xx, r_yy, r_xy, tau_epsilon_p, tau_sigma_p, tau_epsilon_s, tau_sigma_s, alpha_p, alpha_s
    global memory_dvx_dx, memory_dvx_dy
    global memory_dvy_dx, memory_dvy_dy
    global memory_dsigmaxx_dx, memory_dsigmayy_dy
    global memory_dsigmaxy_dx, memory_dsigmaxy_dy
    global value_dvx_dx, value_dvy_dy
    global value_dsigmaxx_dx, value_dsigmaxy_dy
    global value_dsigmaxy_dx, value_dsigmayy_dy
    global sisvx, sisvy
    global ix_src, iy_src, ix_rec, iy_rec
    global simul_roi, rho_grid_vx, cp_grid_vx, cs_grid_vx
    global windows_cpu

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

    # Obtem fontes e receptores dos transdutores
    source_term = list()
    idx_src = list()
    idx_rec = list()
    idx_src_offset = 0
    idx_rec_offset = 0
    for _pr in simul_probes:
        if source_env:
            st = _pr.get_source_term(samples=NSTEP, dt=dt, out='e')
            _, i_src = _pr.get_points_roi(sim_roi=simul_roi, simul_type="2d")
        else:
            st = _pr.get_source_term(samples=NSTEP, dt=dt)
            _, i_src = _pr.get_points_roi(sim_roi=simul_roi, simul_type="2d")
        if len(i_src) > 0:
            source_term.append(st)
            idx_src += [np.array(_s) + idx_src_offset for _s in i_src]
            idx_src_offset += _pr.num_elem

        i_rec = _pr.get_idx_rec(sim_roi=simul_roi, simul_type="2D")
        if len(i_rec) > 0:
            idx_rec += [np.array(_r) + idx_rec_offset for _r in i_rec]
            idx_rec_offset += _pr.num_elem

    # Source terms
    if len(source_term) > 1:
        source_term = np.concatenate(source_term, axis=1)
    else:
        source_term = source_term[0][:, np.newaxis]

    if save_sources:
        np.save(f'ensaios/teste_viscoelastico/results/sources_2D_viscoelast_CPML_{
            datetime.now().strftime("%Y%m%d-%H%M%S")}_CPU', source_term)

    idx_src = np.array(idx_src).astype(np.int32).flatten()
    idx_rec = np.array(idx_rec).astype(np.int32).flatten()

    # rho_grid_vy e a matriz de densidade calculada no ponto medio do grid de vx (grid de vy)
    rho_grid_vy = rho_grid_vx
    rho_grid_vy[:-1, :-1] = flt32(0.25) * (
            rho_grid_vx[:-1, :-1] + rho_grid_vx[1:, :-1] + rho_grid_vx[1:, 1:] + rho_grid_vx[:-1, 1:])

    # Inicializa os mapas dos parametros de Lame
    mu_grid_vx = mu_grid_sig_norm = mu_grid_sig_trans = rho_grid_vx * cs_grid_vx * cs_grid_vx
    lambda_grid_vx = lambda_grid_sig_norm = rho_grid_vx * (cp_grid_vx * cp_grid_vx - 2.0 * cs_grid_vx * cs_grid_vx)

    mu_grid_sig_norm[:-1, :-1] = flt32(0.5) * (mu_grid_vx[1:, :-1] + mu_grid_vx[:-1, :-1])
    lambda_grid_sig_norm[:-1, :-1] = flt32(0.5) * (lambda_grid_vx[1:, :-1] + lambda_grid_vx[:-1, :-1])
    lambdaplus2mu_grid_sig_norm = lambda_grid_sig_norm + flt32(2.0) * mu_grid_sig_norm
    mu_grid_sig_trans[:-1, :-1] = flt32(0.5) * (mu_grid_vx[:-1, 1:] + mu_grid_vx[:-1, :-1])
    lambdaplusmu_grid_sig_norm = lambda_grid_sig_norm + mu_grid_sig_norm

    alpha_p = tau_epsilon_p / tau_sigma_p
    alpha_s = tau_epsilon_s / tau_sigma_s
    deltat_phi_p = dt * (1.0 - alpha_p) / tau_sigma_p / np.sum(alpha_p)
    deltat_phi_s = dt * (1.0 - alpha_s) / tau_sigma_s / np.sum(alpha_s)
    half_deltat_overtau_sigma_p = np.float32(0.5) * dt / tau_sigma_p
    half_deltat_overtau_sigma_s = np.float32(0.5) * dt / tau_sigma_s
    mult_factor_tau_sigma_p = np.float32(1.0) / (1.0 + half_deltat_overtau_sigma_p)
    mult_factor_tau_sigma_s = np.float32(1.0) / (1.0 + half_deltat_overtau_sigma_s)

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
        r_xx_old = r_xx
        r_yy_old = r_yy
        sum_r_xx = np.zeros_like(sigmaxx)
        sum_r_yy = np.zeros_like(sigmayy)
        if viscoelastic_attn:
            for _l in range(N_SLS):
                # r_xx[:, :, _l] += -(r_xx[:, :, _l] +
                #                     alpha_p[_l] * lambdaplus2mu_grid_sig_norm * (value_dvx_dx + value_dvy_dy)
                #                     - np.float32(2.0) * alpha_s[_l] * mu_grid_sig_norm *
                #                     value_dvy_dy) * dt / tau_sigma_p[_l]
                #
                # r_yy[:, :, _l] += -(r_yy[:, :, _l] +
                #                     alpha_p[_l] * lambdaplus2mu_grid_sig_norm * (value_dvx_dx + value_dvy_dy)
                #                     - np.float32(2.0) * alpha_s[_l] * mu_grid_sig_norm *
                #                     value_dvx_dx) * dt / tau_sigma_p[_l]
                r_xx[:, :, _l] = ((r_xx_old[:, :, _l] +
                                  (value_dvx_dx + value_dvy_dy) * deltat_phi_p[_l] -
                                  r_xx_old[:, :, _l] * half_deltat_overtau_sigma_p[_l]) *
                                  mult_factor_tau_sigma_p[_l])

                r_yy[:, :, _l] = ((r_yy_old[:, :, _l] +
                                  0.5 * (value_dvx_dx - value_dvy_dy) * deltat_phi_s[_l] -
                                  r_yy_old[:, :, _l] * half_deltat_overtau_sigma_s[_l]) *
                                  mult_factor_tau_sigma_s[_l])

                sum_r_xx += r_xx[:, :, _l] + r_xx_old[:, :, _l]
                sum_r_yy += r_yy[:, :, _l] + r_yy_old[:, :, _l]

            # sigmaxx += (lambdaplus2mu_grid_sig_norm * np.mean(alpha_p) * (value_dvx_dx + value_dvy_dy) -
            #             2.0 * mu_grid_sig_norm * np.mean(alpha_s) * value_dvy_dy + np.sum(r_xx, axis=2)) * dt
            # sigmayy += (lambdaplus2mu_grid_sig_norm * np.mean(alpha_p) * (value_dvx_dx + value_dvy_dy) -
            #             2.0 * mu_grid_sig_norm * np.mean(alpha_s) * value_dvx_dx + np.sum(r_yy, axis=2)) * dt

            sigmaxx += (lambdaplus2mu_grid_sig_norm * value_dvx_dx + lambda_grid_sig_norm * value_dvy_dy +
                        0.5 * lambdaplusmu_grid_sig_norm * sum_r_xx + mu_grid_sig_norm * sum_r_yy) * dt
            sigmayy += (lambda_grid_sig_norm * value_dvx_dx + lambdaplus2mu_grid_sig_norm * value_dvy_dy +
                        0.5 * lambdaplusmu_grid_sig_norm * sum_r_xx - mu_grid_sig_norm * sum_r_yy) * dt
        else:
            sigmaxx += (lambdaplus2mu_grid_sig_norm * value_dvx_dx + lambda_grid_sig_norm * value_dvy_dy) * dt
            sigmayy += (lambda_grid_sig_norm * value_dvx_dx + lambdaplus2mu_grid_sig_norm * value_dvy_dy) * dt

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
        r_xy_old = r_xy
        sum_r_xy = np.zeros_like(sigmaxy)
        if viscoelastic_attn:
            for _l in range(N_SLS):
                # r_xy[:, :, _l] += - dt * (r_xy[:, :, _l] + alpha_s[_l] * mu_grid_sig_trans *
                #                           (value_dvy_dx + value_dvx_dy)) / tau_sigma_s[_l]
                r_xy[:, :, _l] = ((r_xy_old[:, :, _l] +
                                   (value_dvy_dx + value_dvx_dy) * deltat_phi_s[_l] -
                                   r_xy_old[:, :, _l] * half_deltat_overtau_sigma_s[_l]) *
                                  mult_factor_tau_sigma_s[_l])

                sum_r_xy += r_xy[:, :, _l] + r_xy_old[:, :, _l]

            # sigmaxy += (mu_grid_sig_trans * np.mean(alpha_s) * (value_dvx_dy + value_dvy_dx) + np.sum(r_xy, axis=2))
            sigmaxy += (mu_grid_sig_trans * (value_dvx_dy + value_dvy_dx) + 0.5 * sum_r_xy) * dt
        else:
            sigmaxy += mu_grid_sig_trans * (value_dvx_dy + value_dvy_dx) * dt

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

        vx = dt * (value_dsigmaxx_dx + value_dsigmaxy_dy) / rho_grid_vx + vx

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

        vy = dt * (value_dsigmaxy_dx + value_dsigmayy_dy) / rho_grid_vy + vy

        # add the source (force vector located at a given grid point)
        for _isrc in range(NSRC):
            val_src = source_term[it - 1, idx_src[_isrc]]
            vy[ix_src[_isrc], iy_src[_isrc]] += val_src * dt / rho

        # implement Dirichlet boundary conditions on the six edges of the grid
        # which is the right condition to implement in order for C-PML to remain stable at long times
        # xmin
        qwe = vy[300:321, 300:321]
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
        for _i in range(idx_rec.shape[0]):
            _irec = idx_rec[_i]
            if it >= delay_recv[_irec]:
                _x = ix_rec[_i]
                _y = iy_rec[_i]
                sisvx[it - 1, _irec] += vx[_x, _y]
                sisvy[it - 1, _irec] += vy[_x, _y]

        vsn2 = np.sqrt(np.max(vx[:, :] ** 2 + vy[:, :] ** 2))
        if (it % IT_DISPLAY) == 0 or it == 5 or it == 1:
            if show_debug:
                print(f'Time step # {it} out of {NSTEP}')
                print(f'Max norm velocity vector V (m/s) = {vsn2}')
                print(f'Max Vx = {np.max(vx)}, Vy = {np.max(vy)}')
                print(f'Min Vx = {np.min(vx)}, Vy = {np.min(vy)}')

            if show_anim:
                windows_cpu[0].imv.setImage(vx[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                windows_cpu[1].imv.setImage(vy[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                App.processEvents()

        # Verifica a estabilidade da simulacao
        if vsn2 > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instavel")
            exit(2)


# ----------------------------------------
# Funcao do simulador em WebGPU
# ----------------------------------------
def sim_webgpu(device):

    global simul_probes, coefs
    global a_x, a_x_half, b_x, b_x_half, k_x, k_x_half
    global a_y, a_y_half, b_y, b_y_half, k_y, k_y_half
    global vx, vy, sigmaxx, sigmayy, sigmaxy
    global r_xx, r_yy, r_xy, tau_epsilon_p, tau_sigma_p, tau_epsilon_s, tau_sigma_s, sum_alpha_p, sum_alpha_s
    global r_xx_old, r_yy_old, r_xy_old
    global memory_dvx_dx, memory_dvx_dy
    global memory_dvy_dx, memory_dvy_dy
    global memory_dsigmaxx_dx, memory_dsigmayy_dy
    global memory_dsigmaxy_dx, memory_dsigmaxy_dy
    global value_dvx_dx, value_dvy_dy
    global value_dsigmaxx_dx, value_dsigmaxy_dy
    global value_dsigmaxy_dx, value_dsigmayy_dy
    global sisvx, sisvy
    global ix_src, iy_src, ix_rec, iy_rec
    global simul_roi, rho_grid_vx, cp_grid_vx, cs_grid_vx
    global windows_gpu

    # Obtem fontes e receptores dos transdutores
    source_term = list()
    idx_src = list()
    idx_rec = list()
    idx_src_offset = 0
    idx_rec_offset = 0
    for _pr in simul_probes:
        if source_env:
            st = _pr.get_source_term(samples=NSTEP, dt=dt, out='e')
            _, i_src = _pr.get_points_roi(sim_roi=simul_roi, simul_type="2d")
        else:
            st = _pr.get_source_term(samples=NSTEP, dt=dt)
            _, i_src = _pr.get_points_roi(sim_roi=simul_roi, simul_type="2d")
        if len(i_src) > 0:
            source_term.append(st)
            idx_src += [np.array(_s) + idx_src_offset for _s in i_src]
            idx_src_offset += _pr.num_elem

        i_rec = _pr.get_idx_rec(sim_roi=simul_roi, simul_type="2D")
        if len(i_rec) > 0:
            idx_rec += [np.array(_r) + idx_rec_offset for _r in i_rec]
            idx_rec_offset += _pr.num_elem

    # Source terms
    if len(source_term) > 1:
        source_term = np.concatenate(source_term, axis=1)
    else:
        source_term = source_term[0][:, np.newaxis]

    if save_sources:
        np.save(f'ensaios/teste_viscoelastico/results/sources_2D_viscoelast_CPML_{
            datetime.now().strftime("%Y%m%d-%H%M%S")}_GPU', source_term)

    pos_sources = -np.ones((nx, ny), dtype=np.int32)
    pos_sources[ix_src, iy_src] = np.array(idx_src).astype(np.int32).flatten()

    # Receivers
    info_rec_pt = np.column_stack((ix_rec, iy_rec, np.array(idx_rec).flatten())).astype(np.int32)
    numbers = list(np.array(idx_rec, dtype=np.int32).flatten())
    offset_sensors = [numbers[0]]
    for i in range(1, len(numbers)):
        if numbers[i] != numbers[i - 1]:
            offset_sensors.append(np.int32(i))
    offset_sensors = np.array(offset_sensors, dtype=np.int32)
    n_pto_rec = np.int32(len(numbers))

    # Arrays com parametros inteiros (i32) e ponto flutuante (f32) para rodar o simulador
    _ord = coefs.shape[0]
    params_i32 = np.array([nx, ny, NSTEP, source_term.shape[1], sisvx.shape[1], n_pto_rec, _ord, 0, N_SLS,
                           1 if viscoelastic_attn else 0],
                          dtype=np.int32)
    params_f32 = np.array([dx, dy, dt], dtype=flt32)

    alpha_sum = np.array([sum_alpha_p, sum_alpha_s], dtype=flt32)

    tau_attenuation = np.array([tau_epsilon_p, tau_sigma_p, tau_epsilon_s, tau_sigma_s], dtype=flt32)

    # Cria o shader para calculo contido no arquivo ``shader_2D_elast_cpml.wgsl''
    with open('shader_2D_viscoelastic.wgsl') as shader_file:
        cshader_string = shader_file.read()
        cshader_string = cshader_string.replace('wsx', f'{wsx}')
        cshader_string = cshader_string.replace('wsy', f'{wsy}')
        cshader_string = cshader_string.replace('idx_rec_offset', f'{idx_rec_offset}')
        cshader = device.create_shader_module(code=cshader_string)

    # Definicao dos buffers que terao informacoes compartilhadas entre CPU e GPU
    # ------- Buffers para o binding de parametros -------------
    # Buffer de parametros com valores em ponto flutuante
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_param_flt32 = device.create_buffer_with_data(data=params_f32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)

    # Forcas da fonte
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_force = device.create_buffer_with_data(data=source_term,
                                             usage=wgpu.BufferUsage.STORAGE |
                                                   wgpu.BufferUsage.COPY_SRC)

    # Indices das fontes na ROI
    b_idx_src = device.create_buffer_with_data(data=pos_sources, usage=wgpu.BufferUsage.STORAGE |
                                                                       wgpu.BufferUsage.COPY_SRC)

    # Coeficientes de absorcao
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_a_x = device.create_buffer_with_data(data=a_x.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_b_x = device.create_buffer_with_data(data=b_x.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_k_x = device.create_buffer_with_data(data=k_x.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_a_x_h = device.create_buffer_with_data(data=a_x_half.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                            wgpu.BufferUsage.COPY_SRC)
    b_b_x_h = device.create_buffer_with_data(data=b_x_half.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                            wgpu.BufferUsage.COPY_SRC)
    b_k_x_h = device.create_buffer_with_data(data=k_x_half.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                            wgpu.BufferUsage.COPY_SRC)

    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_a_y = device.create_buffer_with_data(data=a_y.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_b_y = device.create_buffer_with_data(data=b_y.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_k_y = device.create_buffer_with_data(data=k_y.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_a_y_h = device.create_buffer_with_data(data=a_y_half.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                            wgpu.BufferUsage.COPY_SRC)
    b_b_y_h = device.create_buffer_with_data(data=b_y_half.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                            wgpu.BufferUsage.COPY_SRC)
    b_k_y_h = device.create_buffer_with_data(data=k_y_half.flatten(), usage=wgpu.BufferUsage.STORAGE |
                                                                            wgpu.BufferUsage.COPY_SRC)

    # Parametros de absorcao
    b_r_xx = device.create_buffer_with_data(data=r_xx, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)

    b_r_yy = device.create_buffer_with_data(data=r_yy, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_DST |
                                                             wgpu.BufferUsage.COPY_SRC)

    b_r_xy = device.create_buffer_with_data(data=r_xy, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_DST |
                                                             wgpu.BufferUsage.COPY_SRC)

    b_r_xx_old = device.create_buffer_with_data(data=r_xx_old, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)

    b_r_yy_old = device.create_buffer_with_data(data=r_yy_old, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_DST |
                                                             wgpu.BufferUsage.COPY_SRC)

    b_r_xy_old = device.create_buffer_with_data(data=r_xy_old, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_DST |
                                                             wgpu.BufferUsage.COPY_SRC)

    b_alpha_sum = device.create_buffer_with_data(data=alpha_sum, usage=wgpu.BufferUsage.STORAGE |
                                                                                       wgpu.BufferUsage.COPY_SRC)

    b_tau_attenuation = device.create_buffer_with_data(data=tau_attenuation, usage=wgpu.BufferUsage.STORAGE |
                                                                                   wgpu.BufferUsage.COPY_SRC)

    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_param_int32 = device.create_buffer_with_data(data=params_i32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC |
                                                                          wgpu.BufferUsage.COPY_DST)

    # Buffers com os indices para o calculo das derivadas com acuracia maior
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    idx_fd = np.array([[c + 1, c, -c, -c - 1] for c in range(_ord)], dtype=np.int32)
    b_idx_fd = device.create_buffer_with_data(data=idx_fd, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    # Buffer com os mapas de velocidade e densidade da ROI
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_rho_map = device.create_buffer_with_data(data=rho_grid_vx, usage=wgpu.BufferUsage.STORAGE |
                                                                       wgpu.BufferUsage.COPY_SRC)
    b_cp_map = device.create_buffer_with_data(data=cp_grid_vx, usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)
    b_cs_map = device.create_buffer_with_data(data=cs_grid_vx, usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)

    # Buffer com os coeficientes para ao calculo das derivadas
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_fd_coeffs = device.create_buffer_with_data(data=coefs, usage=wgpu.BufferUsage.STORAGE |
                                                                   wgpu.BufferUsage.COPY_SRC)

    # Buffers com os arrays de simulacao
    # Velocidades
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    b_vx = device.create_buffer_with_data(data=vx, usage=wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)
    b_vy = device.create_buffer_with_data(data=vy, usage=wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_DST |
                                                         wgpu.BufferUsage.COPY_SRC)
    b_v_2 = device.create_buffer_with_data(data=v_2, usage=wgpu.BufferUsage.STORAGE |
                                                           wgpu.BufferUsage.COPY_DST |
                                                           wgpu.BufferUsage.COPY_SRC)

    # Estresses
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    b_sigmaxx = device.create_buffer_with_data(data=sigmaxx, usage=wgpu.BufferUsage.STORAGE |
                                                                   wgpu.BufferUsage.COPY_DST |
                                                                   wgpu.BufferUsage.COPY_SRC)
    b_sigmayy = device.create_buffer_with_data(data=sigmayy, usage=wgpu.BufferUsage.STORAGE |
                                                                   wgpu.BufferUsage.COPY_DST |
                                                                   wgpu.BufferUsage.COPY_SRC)
    b_sigmaxy = device.create_buffer_with_data(data=sigmaxy, usage=wgpu.BufferUsage.STORAGE |
                                                                   wgpu.BufferUsage.COPY_DST |
                                                                   wgpu.BufferUsage.COPY_SRC)

    # Arrays de memoria do simulador
    # [STORAGE | COPY_SRC] pois sao valores passados para a GPU, mas nao necessitam retornar a CPU
    b_memory_dvx_dx = device.create_buffer_with_data(data=memory_dvx_dx, usage=wgpu.BufferUsage.STORAGE |
                                                                               wgpu.BufferUsage.COPY_SRC)
    b_memory_dvx_dy = device.create_buffer_with_data(data=memory_dvx_dy, usage=wgpu.BufferUsage.STORAGE |
                                                                               wgpu.BufferUsage.COPY_SRC)
    b_memory_dvy_dx = device.create_buffer_with_data(data=memory_dvy_dx, usage=wgpu.BufferUsage.STORAGE |
                                                                               wgpu.BufferUsage.COPY_SRC)
    b_memory_dvy_dy = device.create_buffer_with_data(data=memory_dvy_dy, usage=wgpu.BufferUsage.STORAGE |
                                                                               wgpu.BufferUsage.COPY_SRC)
    b_memory_dsigmaxx_dx = device.create_buffer_with_data(data=memory_dsigmaxx_dx, usage=wgpu.BufferUsage.STORAGE |
                                                                                         wgpu.BufferUsage.COPY_SRC)
    b_memory_dsigmayy_dy = device.create_buffer_with_data(data=memory_dsigmayy_dy, usage=wgpu.BufferUsage.STORAGE |
                                                                                         wgpu.BufferUsage.COPY_SRC)
    b_memory_dsigmaxy_dx = device.create_buffer_with_data(data=memory_dsigmaxy_dx, usage=wgpu.BufferUsage.STORAGE |
                                                                                         wgpu.BufferUsage.COPY_SRC)
    b_memory_dsigmaxy_dy = device.create_buffer_with_data(data=memory_dsigmaxy_dy, usage=wgpu.BufferUsage.STORAGE |
                                                                                         wgpu.BufferUsage.COPY_SRC)

    # Sinal do sensor
    # [STORAGE | COPY_DST | COPY_SRC] pois sao valores passados para a GPU e tambem retornam a CPU [COPY_DST]
    b_sens_x = device.create_buffer_with_data(data=sisvx, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)
    b_sens_y = device.create_buffer_with_data(data=sisvy, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)
    b_sens_sigxx = device.create_buffer_with_data(data=sisvy, usage=wgpu.BufferUsage.STORAGE |
                                                                    wgpu.BufferUsage.COPY_DST |
                                                                    wgpu.BufferUsage.COPY_SRC)
    b_sens_sigyy = device.create_buffer_with_data(data=sisvy, usage=wgpu.BufferUsage.STORAGE |
                                                                    wgpu.BufferUsage.COPY_DST |
                                                                    wgpu.BufferUsage.COPY_SRC)
    b_sens_sigxy = device.create_buffer_with_data(data=sisvy, usage=wgpu.BufferUsage.STORAGE |
                                                                    wgpu.BufferUsage.COPY_DST |
                                                                    wgpu.BufferUsage.COPY_SRC)

    # Tempo de espera para recepcao nos sensores
    b_delay_rec = device.create_buffer_with_data(data=delay_recv, usage=wgpu.BufferUsage.STORAGE |
                                                                        wgpu.BufferUsage.COPY_SRC)

    # Informacoes dos pontos receptores
    b_info_rec_pt = device.create_buffer_with_data(data=info_rec_pt, usage=wgpu.BufferUsage.STORAGE |
                                                                           wgpu.BufferUsage.COPY_SRC)
    b_offset_sensors = device.create_buffer_with_data(data=offset_sensors, usage=wgpu.BufferUsage.STORAGE |
                                                                                 wgpu.BufferUsage.COPY_SRC)

    # Esquema de amarracao dos parametros (binding layouts [bl])
    # Parametros
    bl_params = [
        {"binding": 0,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 21,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 22,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 23,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 24,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 25,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 26,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 27,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 28,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
    ]
    bl_params += [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         } for ii in range(1, 21)
    ]

    # Arrays da simulacao
    bl_sim_arrays = [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         } for ii in range(0, 14)
    ]

    # Sensores
    bl_sensors = [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         } for ii in [*range(0, 2), *range(5, 8)]
    ]
    bl_sensors += [
        {"binding": ii,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         } for ii in range(2, 5)
    ]

    # Configuracao das amarracoes (bindings)
    b_params = [
        {
            "binding": 0,
            "resource": {"buffer": b_param_int32, "offset": 0, "size": b_param_int32.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b_param_flt32, "offset": 0, "size": b_param_flt32.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": b_force, "offset": 0, "size": b_force.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b_idx_src, "offset": 0, "size": b_idx_src.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": b_a_x, "offset": 0, "size": b_a_x.size},
        },
        {
            "binding": 5,
            "resource": {"buffer": b_b_x, "offset": 0, "size": b_b_x.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": b_k_x, "offset": 0, "size": b_k_x.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b_a_x_h, "offset": 0, "size": b_a_x_h.size},
        },
        {
            "binding": 8,
            "resource": {"buffer": b_b_x_h, "offset": 0, "size": b_b_x_h.size},
        },
        {
            "binding": 9,
            "resource": {"buffer": b_k_x_h, "offset": 0, "size": b_k_x_h.size},
        },
        {
            "binding": 10,
            "resource": {"buffer": b_a_y, "offset": 0, "size": b_a_y.size},
        },
        {
            "binding": 11,
            "resource": {"buffer": b_b_y, "offset": 0, "size": b_b_y.size},
        },
        {
            "binding": 12,
            "resource": {"buffer": b_k_y, "offset": 0, "size": b_k_y.size},
        },
        {
            "binding": 13,
            "resource": {"buffer": b_a_y_h, "offset": 0, "size": b_a_y_h.size},
        },
        {
            "binding": 14,
            "resource": {"buffer": b_b_y_h, "offset": 0, "size": b_b_y_h.size},
        },
        {
            "binding": 15,
            "resource": {"buffer": b_k_y_h, "offset": 0, "size": b_k_y_h.size},
        },
        {
            "binding": 16,
            "resource": {"buffer": b_idx_fd, "offset": 0, "size": b_idx_fd.size},
        },
        {
            "binding": 17,
            "resource": {"buffer": b_fd_coeffs, "offset": 0, "size": b_fd_coeffs.size},
        },
        {
            "binding": 18,
            "resource": {"buffer": b_rho_map, "offset": 0, "size": b_rho_map.size},
        },
        {
            "binding": 19,
            "resource": {"buffer": b_cp_map, "offset": 0, "size": b_cp_map.size},
        },
        {
            "binding": 20,
            "resource": {"buffer": b_cs_map, "offset": 0, "size": b_cs_map.size},
        },
        {
            "binding": 21,
            "resource": {"buffer": b_r_xx, "offset": 0, "size": b_r_xx.size},
        },
        {
            "binding": 22,
            "resource": {"buffer": b_alpha_sum, "offset": 0, "size": b_alpha_sum.size},
        },
        {
            "binding": 23,
            "resource": {"buffer": b_tau_attenuation, "offset": 0, "size": b_tau_attenuation.size},
        },
        {
            "binding": 24,
            "resource": {"buffer": b_r_yy, "offset": 0, "size": b_r_yy.size},
        },
        {
            "binding": 25,
            "resource": {"buffer": b_r_xy, "offset": 0, "size": b_r_xy.size},
        },
        {
            "binding": 26,
            "resource": {"buffer": b_r_xx_old, "offset": 0, "size": b_r_xx_old.size},
        },
        {
            "binding": 27,
            "resource": {"buffer": b_r_yy_old, "offset": 0, "size": b_r_yy_old.size},
        },
        {
            "binding": 28,
            "resource": {"buffer": b_r_xy_old, "offset": 0, "size": b_r_xy_old.size},
        },
    ]
    b_sim_arrays = [
        {
            "binding": 0,
            "resource": {"buffer": b_vx, "offset": 0, "size": b_vx.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b_vy, "offset": 0, "size": b_vy.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": b_v_2, "offset": 0, "size": b_v_2.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b_sigmaxx, "offset": 0, "size": b_sigmaxx.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": b_sigmayy, "offset": 0, "size": b_sigmayy.size},
        },
        {
            "binding": 5,
            "resource": {"buffer": b_sigmaxy, "offset": 0, "size": b_sigmaxy.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": b_memory_dvx_dx, "offset": 0, "size": b_memory_dvx_dx.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b_memory_dvx_dy, "offset": 0, "size": b_memory_dvx_dy.size},
        },
        {
            "binding": 8,
            "resource": {"buffer": b_memory_dvy_dx, "offset": 0, "size": b_memory_dvy_dx.size},
        },
        {
            "binding": 9,
            "resource": {"buffer": b_memory_dvy_dy, "offset": 0, "size": b_memory_dvy_dy.size},
        },
        {
            "binding": 10,
            "resource": {"buffer": b_memory_dsigmaxx_dx, "offset": 0, "size": b_memory_dsigmaxx_dx.size},
        },
        {
            "binding": 11,
            "resource": {"buffer": b_memory_dsigmayy_dy, "offset": 0, "size": b_memory_dsigmayy_dy.size},
        },
        {
            "binding": 12,
            "resource": {"buffer": b_memory_dsigmaxy_dx, "offset": 0, "size": b_memory_dsigmaxy_dx.size},
        },
        {
            "binding": 13,
            "resource": {"buffer": b_memory_dsigmaxy_dy, "offset": 0, "size": b_memory_dsigmaxy_dy.size},
        },
    ]
    b_sensors = [
        {
            "binding": 0,
            "resource": {"buffer": b_sens_x, "offset": 0, "size": b_sens_x.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b_sens_y, "offset": 0, "size": b_sens_y.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": b_delay_rec, "offset": 0, "size": b_delay_rec.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b_info_rec_pt, "offset": 0, "size": b_info_rec_pt.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": b_offset_sensors, "offset": 0, "size": b_offset_sensors.size},
        },
        {
            "binding": 5,
            "resource": {"buffer": b_sens_sigxx, "offset": 0, "size": b_sens_sigxx.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": b_sens_sigyy, "offset": 0, "size": b_sens_sigyy.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b_sens_sigxy, "offset": 0, "size": b_sens_sigxy.size},
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
    compute_store_sensors_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                                  compute={"module": cshader,
                                                                           "entry_point": "store_sensors_kernel"})
    compute_incr_it_kernel = device.create_compute_pipeline(layout=pipeline_layout,
                                                            compute={"module": cshader,
                                                                     "entry_point": "incr_it_kernel"})

    v_max = 10.0
    v_min = - v_max
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

        # # Ativa o pipeline de execucao do calculo dos estresses
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

        # Ativa o pipeline de execucao do armazenamento dos sensores
        compute_pass.set_pipeline(compute_store_sensors_kernel)
        compute_pass.dispatch_workgroups(1)

        # Ativa o pipeline de atualizacao da amostra de tempo
        compute_pass.set_pipeline(compute_incr_it_kernel)
        compute_pass.dispatch_workgroups(1)

        # Termina o passo de execucao
        compute_pass.end()

        # Efetua a execucao dos comandos na GPU
        device.queue.submit([command_encoder.finish()])

        # Leitura da GPU para sincronismo
        vsn2 = np.sqrt(device.queue.read_buffer(b_v_2, buffer_offset=0, size=b_v_2.size).cast("f")[0])
        if (it % IT_DISPLAY) == 0 or it == 5:
            if show_debug:
                print(f'Time step # {it} out of {NSTEP}')
                print(f'Max norm velocity vector V (m/s) = {vsn2}')

            if show_anim:
                vxgpu = np.asarray(device.queue.read_buffer(b_vx, buffer_offset=0).cast("f")).reshape((nx, ny))
                vygpu = np.asarray(device.queue.read_buffer(b_vy, buffer_offset=0).cast("f")).reshape((nx, ny))

                windows_gpu[0].imv.setImage(vxgpu[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                windows_gpu[1].imv.setImage(vygpu[ix_min:ix_max, iy_min:iy_max], levels=[v_min, v_max])
                App.processEvents()

                if show_debug:
                    print(f'Max Vx = {np.max(vxgpu)}, Vy = {np.max(vygpu)}')
                    print(f'Min Vx = {np.min(vxgpu)}, Vy = {np.min(vygpu)}')

        # Verifica a estabilidade da simulacao
        if vsn2 > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instavel")
            exit(2)

    # Pega os resultados da simulacao
    vxgpu = np.asarray(device.queue.read_buffer(b_vx, buffer_offset=0).cast("f")).reshape((nx, ny))
    vygpu = np.asarray(device.queue.read_buffer(b_vy, buffer_offset=0).cast("f")).reshape((nx, ny))
    sigxx_gpu = np.asarray(device.queue.read_buffer(b_sigmaxx, buffer_offset=0).cast("f")).reshape((nx, ny))
    sigyy_gpu = np.asarray(device.queue.read_buffer(b_sigmayy, buffer_offset=0).cast("f")).reshape((nx, ny))
    sigxy_gpu = np.asarray(device.queue.read_buffer(b_sigmaxy, buffer_offset=0).cast("f")).reshape((nx, ny))
    sens_vx = np.array(device.queue.read_buffer(b_sens_x).cast("f")).reshape((NSTEP, NREC))
    sens_vy = np.array(device.queue.read_buffer(b_sens_y).cast("f")).reshape((NSTEP, NREC))
    sens_sigxx = np.array(device.queue.read_buffer(b_sens_sigxx).cast("f")).reshape((NSTEP, NREC))
    sens_sigyy = np.array(device.queue.read_buffer(b_sens_sigyy).cast("f")).reshape((NSTEP, NREC))
    sens_sigxy = np.array(device.queue.read_buffer(b_sens_sigxy).cast("f")).reshape((NSTEP, NREC))

    return (vxgpu, vygpu, sigxx_gpu, sigyy_gpu, sigxy_gpu, sens_vx, sens_vy, sens_sigxx, sens_sigyy, sens_sigxy,
            device.adapter.info["device"])


# ----------------------------------------------------------
# Aqui comeca o codigo principal de execucao dos simuladores
# ----------------------------------------------------------
# ----------------------------------------------------------
# Definicao de constantes
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

# Parametro de atenuacao
N_SLS = 3

# Qp approximately equal to 13, Qkappa approximately to 20 and Qmu / Qs approximately to 10
q_kappa_att = 20.0
q_mu_att = 10.0
f0_attenuation = 16  # in Hz


# Portar essa funçãobbbbbbbbbbb
def compute_attenuation_coeffs(is_kappa):  # n, q_kappa, f0, f_min,f_max):
    if is_kappa:
        return (np.array([0.024081581857536852, 0.0046996089908613505, 0.00095679978724359251], dtype=flt32),
                np.array([0.022560146386368083, 0.0045084712797122525, 0.00089378764037688395], dtype=flt32))

    return (np.array([0.024305444805272164, 0.0047281078292263955, 0.00096672526958635019], dtype=flt32),
            np.array([0.022509197794294895, 0.0045013880073380965, 0.00089173320953691182], dtype=flt32))


# ----------------------------------------------------------
# Avaliacao dos parametros na linha de comando
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Configuration file', default='config.json')
args = parser.parse_args()

# -----------------------
# Leitura da configuracao no formato JSON
# -----------------------
with open(args.config, 'r') as f:
    configs = ast.literal_eval(f.read())

deriv_ord = configs["simul_params"]["ord"] if "ord" in configs["simul_params"] else 2
try:
    coefs = np.array(coefs_Lui[deriv_ord - 2], dtype=flt32)
except IndexError:
    print(f"Acurácia das derivadas {deriv_ord} não suportada. Usando o maior valor permitido (6).")
    coefs = np.array(coefs_Lui[-1], dtype=flt32)

# Configuracao do corpo de prova
cp = flt32(5.9)
if "cp" in configs["specimen_params"]:
    cp = flt32(configs["specimen_params"]["cp"])  # [mm/us]

cp_map = None
if "cp_map" in configs["specimen_params"]:
    cp_map = np.load(configs["specimen_params"]["cp_map"]).astype(np.float32)

cs = flt32(3.23)
if "cs" in configs["specimen_params"]:
    cs = flt32(configs["specimen_params"]["cs"])  # [mm/us]

cs_map = None
if "cs_map" in configs["specimen_params"]:
    cs_map = np.load(configs["specimen_params"]["cs_map"]).astype(np.float32)

rho = flt32(7800.0)
if "rho" in configs["specimen_params"]:
    rho = flt32(configs["specimen_params"]["rho"])

rho_map = None
if "rho_map" in configs["specimen_params"]:
    rho_map = np.load(configs["specimen_params"]["rho_map"]).astype(np.float32)

# Configuracao da ROI
simul_roi = SimulationROI(**configs["roi"], pad=coefs.shape[0] - 1, rho_map=rho_map)

# Configuracao dos transdutores
# TODO: Incluir essa parte em simul_utils, para ficar generico os tipos de transdutores suportados
simul_probes = list()
probes_cfg = configs["probes"]
for p in probes_cfg:
    if "linear" in p:
        simul_probes.append(SimulationProbeLinearArray(**p["linear"]))
    elif "point" in p:
        simul_probes.append(SimulationProbePoint(**p["point"]))
print(f'Ordem da acuracia: {coefs.shape[0] * 2}')

# Configuracao geral dos ensaios
n_iter_gpu = configs["simul_configs"]["n_iter_gpu"] if "n_iter_gpu" in configs["simul_configs"] else 1
n_iter_cpu = configs["simul_configs"]["n_iter_cpu"] if "n_iter_cpu" in configs["simul_configs"] else 1
do_sim_gpu = bool(configs["simul_configs"]["do_sim_gpu"]) if "do_sim_gpu" in configs["simul_configs"] else False
do_sim_cpu = bool(configs["simul_configs"]["do_sim_cpu"]) if "do_sim_cpu" in configs["simul_configs"] else False
show_anim = bool(configs["simul_configs"]["show_anim"]) if "show_anim" in configs["simul_configs"] else False
show_debug = bool(configs["simul_configs"]["show_debug"]) if "show_debug" in configs["simul_configs"] else False
plot_results = bool(configs["simul_configs"]["plot_results"]) if "plot_results" in configs["simul_configs"] else False
plot_sensors = bool(configs["simul_configs"]["plot_sensors"]) if "plot_sensors" in configs["simul_configs"] else False
plot_bscan = bool(configs["simul_configs"]["plot_bscan"]) if "plot_bscan" in configs["simul_configs"] else False
save_bscan = bool(configs["simul_configs"]["save_bscan"]) if "save_bscan" in configs["simul_configs"] else False
save_sources = bool(configs["simul_configs"]["save_sources"]) if "save_sources" in configs["simul_configs"] else False
show_results = bool(configs["simul_configs"]["show_results"]) if "show_results" in configs["simul_configs"] else False
save_results = bool(configs["simul_configs"]["save_results"]) if "save_results" in configs["simul_configs"] else False
viscoelastic_attn = bool(configs["simul_configs"]["viscoelastic_attn"]) \
    if "viscoelastic_attn" in configs["simul_configs"] else False
gpu_type = configs["simul_configs"]["gpu_type"] if "gpu_type" in configs["simul_configs"] else "high-perf"
source_env = bool(configs["simul_configs"]["source_env"]) if "source_env" in configs["simul_configs"] else False
if "emission_laws" in configs["simul_configs"] and os.path.isfile(configs["simul_configs"]["emission_laws"]):
    emission_laws, _ = file_law.read(configs["simul_configs"]["emission_laws"])
else:
    emission_laws = None

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

# Inicializa os mapas de densidade do meio
# rho_grid_vx e a matriz das densidades no mesmo grid de vx
rho_grid_vx = np.ones((nx, ny), dtype=flt32) * rho
if rho_map is not None:
    if rho_map.shape[0] < nx and rho_map.shape[1] < ny:
        rho_grid_vx[simul_roi.get_ix_min(): simul_roi.get_ix_max(),
        simul_roi.get_iz_min(): simul_roi.get_iz_max()] = rho_map
    elif rho_map.shape[0] > nx and rho_map.shape[1] > ny:
        rho_grid_vx = rho_map[:nx, :ny]
    elif rho_map.shape[0] == nx and rho_map.shape[1] == ny:
        rho_grid_vx = rho_map
    else:
        raise ValueError(f'rho_map shape {rho_map.shape} e incompativel com a ROI')

# cp_grid_vx e a matriz das velocidades longitudinais no mesmo grid de vx
cp_grid_vx = np.ones((nx, ny), dtype=flt32) * cp
if cp_map is not None:
    if cp_map.shape[0] < nx and cp_map.shape[1] < ny:
        cp_grid_vx[simul_roi.get_ix_min(): simul_roi.get_ix_max(),
        simul_roi.get_iz_min(): simul_roi.get_iz_max()] = cp_map
    elif cp_map.shape[0] > nx and cp_map.shape[1] > ny:
        cp_grid_vx = cp_map[:nx, :ny]
    elif cp_map.shape[0] == nx and cp_map.shape[1] == ny:
        cp_grid_vx = cp_map
    else:
        raise ValueError(f'cp_map shape {cp_map.shape} e incompativel com a ROI')

# cs_grid_vx e a matriz das velocidades transversais no mesmo grid de vx
cs_grid_vx = np.ones((nx, ny), dtype=flt32) * cs
if cs_map is not None:
    if cs_map.shape[0] < nx and cs_map.shape[1] < ny:
        cs_grid_vx[simul_roi.get_ix_min(): simul_roi.get_ix_max(),
        simul_roi.get_iz_min(): simul_roi.get_iz_max()] = cs_map
    elif cs_map.shape[0] > nx and cs_map.shape[1] > ny:
        cs_grid_vx = cs_map[:nx, :ny]
    elif cs_map.shape[0] == nx and cs_map.shape[1] == ny:
        cs_grid_vx = cs_map
    else:
        raise ValueError(f'cs_map shape {cs_map.shape} e incompativel com a ROI')

# Numero total de passos de tempo
NSTEP = configs["simul_params"]["time_steps"]

# Passo de tempo em microssegundos
dt = flt32(configs["simul_params"]["dt"])

# Numero de iteracoes de tempo para apresentar e armazenar informacoes
IT_DISPLAY = configs["simul_params"]["it_display"]

# Pega as listas de todos os pontos transmissores e receptores de todos os transdutores configurados
i_probe_tx_ptos = list()
i_probe_rx_ptos = list()
delay_recv = list()
NREC = 0
for pr in simul_probes:
    i_probe_tx_ptos += pr.get_points_roi(simul_roi, simul_type="2d", dir="e")[0]
    i_probe_rx_ptos += pr.get_points_roi(simul_roi, simul_type="2d", dir="r")[0]
    delay_recv += pr.get_delay_rx()
    NREC += pr.receivers.count(True)

# Define a posicao das fontes
i_probe_tx_ptos = np.array(i_probe_tx_ptos, dtype=np.int32).reshape(-1, 3)
ix_src = i_probe_tx_ptos[:, 0].astype(np.int32)
iy_src = i_probe_tx_ptos[:, 2].astype(np.int32)
NSRC = i_probe_tx_ptos.shape[0]

# Define a localizacao dos receptores
i_probe_rx_ptos = np.array(i_probe_rx_ptos, dtype=np.int32).reshape(-1, 3)
ix_rec = i_probe_rx_ptos[:, 0].astype(np.int32)
iy_rec = i_probe_rx_ptos[:, 2].astype(np.int32)

# Calcula o delay de recepcao dos receptores
delay_recv = (np.array(delay_recv) / dt + 1.0).astype(np.int32)

# for evolution of total energy in the medium
v_2 = np.float32(0.0)

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

# Arrays para armazenamento das variaveis de memoria para substituir a convolucao
r_xx = np.zeros((nx, ny, N_SLS), dtype=flt32)
r_yy = np.zeros((nx, ny, N_SLS), dtype=flt32)
r_xy = np.zeros((nx, ny, N_SLS), dtype=flt32)

r_xx_old = np.zeros((nx, ny, N_SLS), dtype=flt32)
r_yy_old = np.zeros((nx, ny, N_SLS), dtype=flt32)
r_xy_old = np.zeros((nx, ny, N_SLS), dtype=flt32)

# Calculo da faixa de atenuacao em frequencia: f_max/f_min=12 and (log(f_min)+log(f_max))/2 = log(f0)
f_min_attenuation = np.exp(np.log(f0_attenuation) - np.log(12.0) / 2.0)
f_max_attenuation = 12.0 * f_min_attenuation

tau_epsilon_p, tau_sigma_p = compute_attenuation_coeffs(is_kappa=True)
tau_epsilon_s, tau_sigma_s = compute_attenuation_coeffs(is_kappa=False)

alpha_p = tau_epsilon_p / tau_sigma_p
alpha_s = tau_epsilon_s / tau_sigma_s
sum_alpha_p = sum(alpha_p)
sum_alpha_s = sum(alpha_s)


# Total de arrays
N_ARRAYS = 5 + 2 * 4

print(f'2D viscoelastic finite-difference code in velocity and stress formulation with C-PML')
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
if simul_roi.get_pml_thickness_x() != 0.0:
    d0_x = flt32(-(NPOWER + 1) * cp * np.log(rcoef) / simul_roi.get_pml_thickness_x())
else:
    d0_x = flt32(0.0)

if simul_roi.get_pml_thickness_z() != 0.0:
    d0_y = flt32(-(NPOWER + 1) * cp * np.log(rcoef) / simul_roi.get_pml_thickness_z())
else:
    d0_y = flt32(0.0)

print(f'd0_x = {d0_x}')
print(f'd0_y = {d0_y}')

# Calculo dos coeficientes de amortecimento para a PML
# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = flt32(configs["simul_params"]["k_max_pml"])
ALPHA_MAX_PML = flt32(2.0 * PI * (simul_probes[0].get_freq() / 2.0))  # from Festa and Vilotte

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
cp_max = max(cp_grid_vx.max(), cp)
courant_number = flt32(cp_max * dt * np.sqrt(1.0 / dx ** 2 + 1.0 / dy ** 2))
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
        ]
        windows_gpu = [Window(title=data["title"], geometry=data["geometry"]) for data in windows_gpu_data]
else:
    App = None

# WebGPU
now = datetime.now()
if do_sim_gpu:
    for n in range(n_iter_gpu):
        print(f'Simulacao WEBGPU')
        print(f'wsx = {wsx}, wsy = {wsy}')
        print(f'Iteracao {n}')

        n_laws = emission_laws.shape[0] if emission_laws is not None else 1
        for law in range(n_laws):
            print(f'\tLaw {law} of {n_laws}')

            if emission_laws is not None:
                for p in simul_probes:
                    p.set_t0(emission_laws[law])

            t_gpu = time()
            (vx_gpu, vy_gpu, sigxx_gpu, sigyy_gpu, sigxy_gpu,
             sensor_vx_gpu, sensor_vy_gpu, sensor_sigxx_gpu, sensor_sigyy_gpu, sensor_sigxy_gpu,
             gpu_str) = sim_webgpu(device_gpu)
            times_gpu.append(time() - t_gpu)
            print(gpu_str)
            print(f'{times_gpu[-1]:.3}s')
            name = (f'results/result_2D_elast_CPML_{now.strftime("%Y%m%d-%H%M%S")}_'
                    f'{simul_roi.get_len_x()}x{simul_roi.get_len_z()}_{NSTEP}_iter_{n}_law_{law}')

            # Plota os mapas de velocidades
            if plot_results:
                vx_gpu_sim_result = plt.figure()
                plt.title(f'GPU simulation Vx - law ({law})\n'
                          f'[{gpu_type}]({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(vx_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0]))
                plt.colorbar()

                vy_gpu_sim_result = plt.figure()
                plt.title(f'GPU simulation Vy - law ({law})\n'
                          f'[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(vy_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0])
                           )
                plt.colorbar()

                sigxx_gpu_sim_result = plt.figure()
                plt.title(f'GPU simulation SigXX - law ({law})\n'
                          f'[{gpu_type}]({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sigxx_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0]))
                plt.colorbar()

                sigyy_gpu_sim_result = plt.figure()
                plt.title(f'GPU simulation SigYY - law ({law})\n'
                          f'[{gpu_type}]({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sigyy_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0]))
                plt.colorbar()

                sigxy_gpu_sim_result = plt.figure()
                plt.title(f'GPU simulation SigXY - law ({law})\n'
                          f'[{gpu_type}]({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sigxy_gpu[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0]))
                plt.colorbar()

                if show_results:
                    plt.show(block=False)

                if save_results:
                    vx_gpu_sim_result.savefig(name + '_Vx_gpu_' + gpu_type + '.png')
                    vy_gpu_sim_result.savefig(name + '_Vy_gpu_' + gpu_type + '.png')
                    sigxx_gpu_sim_result.savefig(name + '_SigXX_gpu_' + gpu_type + '.png')
                    sigyy_gpu_sim_result.savefig(name + '_SigYY_gpu_' + gpu_type + '.png')
                    sigxy_gpu_sim_result.savefig(name + '_SigXY_gpu_' + gpu_type + '.png')

            # Plota as velocidades tomadas no sensores
            if plot_results and plot_sensors:
                for r in range(NREC):
                    fig, ax = plt.subplots(3, sharex=True, sharey=True)
                    fig.suptitle(f'Receptor {r + 1} [GPU] - law ({law})')
                    ax[0].plot(sensor_vx_gpu[:, r])
                    ax[0].set_title(r'$V_x$')
                    ax[1].plot(sensor_vy_gpu[:, r])
                    ax[1].set_title(r'$V_y$')
                    ax[2].plot(sensor_vx_gpu[:, r] + sensor_vy_gpu[:, r], 'tab:orange')
                    ax[2].set_title(r'$V_x + V_y$')
                    sensor_gpu_result.append(fig)

                if show_results:
                    plt.show(block=False)

                if save_results:
                    for s in range(NREC):
                        try:
                            sensor_gpu_result[s].savefig(name + f'_sensor_{s}_' + gpu_type + '.png')
                        except IndexError:
                            pass

            if plot_results and plot_bscan:
                gpu_bscan_vx_sim_result = plt.figure()
                plt.title(f'GPU simulation B-scan Vx - law({law})\n'
                          f'[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sensor_vx_gpu, aspect='auto', cmap='viridis')
                plt.colorbar()

                gpu_bscan_vy_sim_result = plt.figure()
                plt.title(f'GPU simulation B-scan Vy - law({law})\n'
                          f'[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sensor_vy_gpu, aspect='auto', cmap='viridis')
                plt.colorbar()

                gpu_bscan_sigxx_sim_result = plt.figure()
                plt.title(f'GPU simulation B-scan SigXX - law({law})\n'
                          f'[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sensor_sigxx_gpu, aspect='auto', cmap='viridis')
                plt.colorbar()

                gpu_bscan_sigyy_sim_result = plt.figure()
                plt.title(f'GPU simulation B-scan SigYY - law({law})\n'
                          f'[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sensor_sigyy_gpu, aspect='auto', cmap='viridis')
                plt.colorbar()

                gpu_bscan_sigxy_sim_result = plt.figure()
                plt.title(f'GPU simulation B-scan SigXY - law({law})\n'
                          f'[{gpu_type}] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sensor_sigxy_gpu, aspect='auto', cmap='viridis')
                plt.colorbar()

                if show_results:
                    plt.show(block=False)

            if save_bscan:
                name = f'results/bscan_2D_elast_CPML_{datetime.now().strftime("%Y%m%d-%H%M%S")}_law_{law}'
                np.save(name + '_Vx_GPU', sensor_vx_gpu)
                np.save(name + '_Vy_GPU', sensor_vy_gpu)
                np.save(name + '_SigXX_GPU', sensor_sigxx_gpu)
                np.save(name + '_SigYY_GPU', sensor_sigyy_gpu)
                np.save(name + '_SigXY_GPU', sensor_sigxy_gpu)

# CPU
if do_sim_cpu:
    for n in range(n_iter_cpu):
        print(f'SIMULACAO CPU')
        print(f'Iteracao {n}')

        n_laws = emission_laws.shape[0] if emission_laws is not None else 1
        for law in range(n_laws):
            print(f'\tLaw {law} of {n_laws}')

            if emission_laws is not None:
                for p in simul_probes:
                    p.set_t0(emission_laws[law])

            t_cpu = time()
            sim_cpu()
            times_cpu.append(time() - t_cpu)
            print(f'{times_cpu[-1]:.3}s')
            name = (f'results/result_2D_elast_CPML_{now.strftime("%Y%m%d-%H%M%S")}_'
                    f'{simul_roi.get_len_x()}x{simul_roi.get_len_z()}_{NSTEP}_iter_{n}_law_{law}')

            # Plota os mapas de velocidade
            if plot_results:
                vx_cpu_sim_result = plt.figure()
                plt.title(f'CPU simulation Vx - law ({law})\n'
                          f'({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(vx[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0])
                           )
                plt.colorbar()

                vy_cpu_sim_result = plt.figure()
                plt.title(f'CPU simulation Vy - law ({law})\n'
                          f'({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(vy[simul_roi.get_ix_min():simul_roi.get_ix_max(),
                           simul_roi.get_iz_min():simul_roi.get_iz_max()].T,
                           aspect='auto', cmap='gray',
                           extent=(simul_roi.w_points[0], simul_roi.w_points[-1],
                                   simul_roi.h_points[-1], simul_roi.h_points[0]))
                plt.colorbar()

                if show_results:
                    plt.show(block=False)

                if save_results:
                    vx_cpu_sim_result.savefig(name + '_Vx_cpu.png')
                    vy_cpu_sim_result.savefig(name + '_Vy_cpu.png')

            # Plota as velocidades tomadas no sensores
            if plot_results and plot_sensors:
                for r in range(NREC):
                    sensor_cpu_result, ax = plt.subplots(3, sharex=True, sharey=True)
                    sensor_cpu_result.suptitle(f'Receptor {r + 1} [CPU] - law ({law})')
                    ax[0].plot(sisvx[:, r])
                    ax[0].set_title(r'$V_x$')
                    ax[1].plot(sisvy[:, r])
                    ax[1].set_title(r'$V_y$')
                    ax[2].plot(sisvx[:, r] + sisvy[:, r], 'tab:orange')
                    ax[2].set_title(r'$V_x + V_y$')

                if show_results:
                    plt.show(block=False)

                if save_results:
                    for s in range(NREC):
                        try:
                            sensor_cpu_result[s].savefig(name + f'_sensor_{s}_cpu.png')
                        except IndexError:
                            pass

            if plot_results and plot_bscan:
                cpu_bscan_sim_result = plt.figure()
                plt.title(f'CPU simulation B-scan - law({law})\n'
                          f'[CPU] ({simul_roi.get_len_x()}x{simul_roi.get_len_z()})')
                plt.imshow(sisvx + sisvy, aspect='auto', cmap='viridis')
                plt.colorbar()

                if show_results:
                    plt.show(block=False)

            if save_bscan:
                name = f'results/bscan_2D_elast_CPML_{datetime.now().strftime("%Y%m%d-%H%M%S")}_law{law}'
                np.save(name + '_Vx_CPU', sisvx)
                np.save(name + '_Vy_CPU', sisvy)

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

if save_results:
    name = (f'results/result_2D_elast_CPML_{now.strftime("%Y%m%d-%H%M%S")}_'
            f'{simul_roi.get_len_x()}x{simul_roi.get_len_z()}_{NSTEP}_iter_')

    if do_sim_gpu:
        np.savetxt(name + 'GPU_' + gpu_type + '.csv', times_gpu, '%10.3f', delimiter=',')

    if do_sim_cpu:
        np.savetxt(name + 'CPU.csv', times_cpu, '%10.3f', delimiter=',')

    with open(name + '_desc.txt', 'w') as f:
        f.write('Parametros do ensaio\n')
        f.write('--------------------\n')
        f.write('\n')
        f.write(f'Quantidade de iteracoes no tempo: {NSTEP}\n')
        f.write(f'Tamanho da ROI: {simul_roi.get_len_x()}x{simul_roi.get_len_z()}\n')
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

if show_results:
    plt.show()
