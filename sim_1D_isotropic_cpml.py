import numpy as np
import ast
from scipy.signal import gausspulse
import matplotlib.pyplot as plt
from simul_utils import SimulationROI


def sim_cpu():
    global source_term, coefs
    global a_x, a_x_half, b_x, b_x_half, k_x, k_x_half
    global vx, sigmaxx
    global memory_dvx_dx
    global memory_dsigmaxx_dx
    global value_dvx_dx
    global value_dsigmaxx_dx
    global sisvx
    global v_2, v_solid_norm

    DELTAT_over_rho = dt / rho
    _ord = coefs.shape[0]
    idx_i = np.array([[c + 1, -c] for c in range(_ord)]) + (_ord - 1)
    idx_f = np.array([[c + 1, -c] for c in range(_ord)]) - _ord

    # Inicio do laco de tempo
    for it in range(1, NSTEP + 1):
        # Calculo da tensao [stress] - {sigma} (equivalente a pressao nos gases-liquidos)
        # sigma_ii -> tensoes normais; sigma_ij -> tensoes cisalhantes
        # Staggered grid da velocidade vx - 1/2: 1,NX-1
        idx_id = idx_i[0, 1]
        idx_fd = idx_f[0, 1]
        for c in range(_ord):
            idx_ia = None if idx_i[c, 0] == 0 else idx_i[c, 0]
            idx_fa = None if idx_f[c, 0] == 0 else idx_f[c, 0]
            idx_ib = None if idx_i[c, 1] == 0 else idx_i[c, 1]
            idx_fb = None if idx_f[c, 1] == 0 else idx_f[c, 1]
            if c:
                value_dvx_dx[idx_id:idx_fd] += coefs[c] * (vx[idx_ia:idx_fa] - vx[idx_ib:idx_fb]) * one_dx
            else:
                value_dvx_dx[idx_id:idx_fd] = coefs[c] * (vx[idx_ia:idx_fa] - vx[idx_ib:idx_fb]) * one_dx

        memory_dvx_dx[idx_id:idx_fd] = (b_x_half[:-1] * memory_dvx_dx[idx_id:idx_fd] +
                                        a_x_half[:-1] * value_dvx_dx[idx_id:idx_fd])

        value_dvx_dx[idx_id:idx_fd] = (value_dvx_dx[idx_id:idx_fd] / k_x_half[:-1] + memory_dvx_dx[idx_id:idx_fd])

        # compute the stress using the Lame parameters
        sigmaxx = sigmaxx + (lambdaplus2mu * value_dvx_dx) * dt

        # Calculo da velocidade
        # # Staggered grid da tensao sigmaxx - 1: 2,NX
        idx_id = idx_i[0, 0]
        idx_fd = idx_f[0, 0]
        for c in range(_ord):
            idx_ia = None if idx_i[c, 0] == 0 else idx_i[c, 0]
            idx_fa = None if idx_f[c, 0] == 0 else idx_f[c, 0]
            idx_ib = None if idx_i[c, 1] == 0 else idx_i[c, 1]
            idx_fb = None if idx_f[c, 1] == 0 else idx_f[c, 1]
            if c:
                value_dsigmaxx_dx[idx_id:idx_fd] += coefs[c] * (sigmaxx[idx_ia: idx_fa] - sigmaxx[idx_ib:idx_fb]) * one_dx
            else:
                value_dsigmaxx_dx[idx_id:idx_fd] = coefs[c] * (sigmaxx[idx_ia: idx_fa] - sigmaxx[idx_ib:idx_fb]) * one_dx

        memory_dsigmaxx_dx[idx_id:idx_fd] = (b_x[1:] * memory_dsigmaxx_dx[idx_id:idx_fd] +
                                             a_x[1:] * value_dsigmaxx_dx[idx_id:idx_fd])

        value_dsigmaxx_dx[idx_id:idx_fd] = (value_dsigmaxx_dx[idx_id:idx_fd] / k_x[1:] +
                                            memory_dsigmaxx_dx[idx_id:idx_fd])

        vx = DELTAT_over_rho * value_dsigmaxx_dx + vx

        # add the source (force vector located at a given grid point)
        for _isrc in range(NSRC):
            vx[ix_src[_isrc]] += source_term[it - 1, _isrc] * dt / rho

        # implement Dirichlet boundary conditions on the six edges of the grid
        # which is the right condition to implement in order for C-PML to remain stable at long times
        # xmin
        vx[:2] = np.float32(0.0)

        # xmax
        vx[-2:] = np.float32(0.0)

        # Store seismograms
        for _irec in range(NREC):
            sisvx[it - 1, _irec] = vx[ix_rec[_irec]]

        v_2 = vx[:] ** 2
        v_solid_norm[it - 1] = np.sqrt(np.max(v_2))
        if (it % IT_DISPLAY) == 0 or it == 5:
            if show_debug:
                print(f'Time step # {it} out of {NSTEP}')
                print(f'Max Vx = {np.max(vx)}')
                print(f'Min Vx = {np.min(vx)}')
                print(f'Max norm velocity vector V (m/s) = {v_solid_norm[it - 1]}')

            if show_anim:
                ax.clear()
                ax.plot(simul_roi.w_points, vx[ix_min:ix_max])
                ax.set_xlim(0.0, dx * nx)
                ax.set_ylim(vx_min, vx_max)
                plt.show(block=False)
                plt.pause(0.001)

        # Verifica a estabilidade da simulacao
        if v_solid_norm[it - 1] > STABILITY_THRESHOLD:
            print("Simulacao tornando-se instavel")
            exit(2)


# ----------------------------------------------------------
# Aqui comeca o codigo principal de execucao dos simuladores
# ----------------------------------------------------------
# Constantes
PI = np.pi
STABILITY_THRESHOLD = 1.0e25  # Limite para considerar que a simulacao esta instavel

# Parametros dos ensaios
flt32 = np.float32
show_debug = False
show_anim = True
vx_min = -100000.0
vx_max = 100000.0

# Definicao das constantes para a o calculo das derivadas, seguindo Lui 2009 (10.1111/j.1365-246X.2009.04305.x)
coefs_Lui = [
    [9.0/8.0, -1.0/24.0],
    [75.0/64.0, -25.0/384.0, 3.0/640.0],
    [1225.0/1024.0, -245.0/3072.0, 49.0/5120.0, -5.0/7168.0],
    [19845.0/16384.0, -735.0/8192.0, 567.0/40960.0, -405.0/229376.0, 35.0/294912.0],
    [160083.0/131072.0, -12705.0/131072.0, 22869.0/1310720.0, -5445.0/1835008.0, 847.0/2359296.0, -63.0/2883584.0]
]

# -----------------------
# Leitura da configuracao no formato JSON
# -----------------------
with open('config.json', 'r') as f:
    configs = ast.literal_eval(f.read())
    data_src = np.array(configs["sources"])
    data_rec = np.array(configs["receivers"])
    source_term_cfg = configs["source_term"]
    coefs = np.array(coefs_Lui[configs["simul_params"]["ord"] - 2], dtype=np.float32)
    simul_roi = SimulationROI(**configs["roi"], pad=coefs.shape[0] - 1)
    print(f'Ordem da acuracia: {coefs.shape[0] * 2}')

# Espessura da PML in pixels
npoints_pml = configs["roi"]["len_pml_xmin"]  # pegando temporariamente

# Parametros da simulacao
nx = simul_roi.get_nx()

# Escala do grid (valor do passo no espaco em milimetros)
dx = simul_roi.w_step
one_dx = 1.0 / dx

# Velocidades do som e densidade do meio
cp = configs["specimen_params"]["cp"]  # [mm/us]
cs = configs["specimen_params"]["cs"]  # [mm/us]
rho = configs["specimen_params"]["rho"]
mu = rho * cs * cs
lambda_ = rho * (cp * cp - 2.0 * cs * cs)
lambdaplus2mu = rho * cp * cp

# Numero total de passos de tempo
NSTEP = configs["simul_params"]["time_steps"]

# Passo de tempo em microssegundos
dt = configs["simul_params"]["dt"]

# Numero de iteracoes de tempo para apresentar e armazenar informacoes
IT_DISPLAY = 10

# Define a posicao das fontes
NSRC = data_src.shape[0]
i_src = np.array([simul_roi.get_nearest_grid_idx(p[0:3]) for p in data_src])
ix_src = i_src[:, 0].astype(np.int32)

# Parametros da fonte
f0 = source_term_cfg["freq"]  # frequencia [MHz]
t0 = data_src[:, 3].reshape((1, NSRC))
factor = flt32(source_term_cfg["gain"])
t = np.expand_dims(np.arange(NSTEP) * dt, axis=1)

# Gauss pulse
source_term = factor * flt32(gausspulse((t - t0), fc=f0, bw=source_term_cfg["bw"]))

# Define a localizacao dos receptores
NREC = data_rec.shape[0]
i_rec = np.array([simul_roi.get_nearest_grid_idx(p[0:3]) for p in data_rec])
ix_rec = i_rec[:, 0].astype(np.int32)

# for evolution of total energy in the medium
v_2 = np.zeros(nx, dtype=flt32)
v_solid_norm = np.zeros(NSTEP, dtype=flt32)

# Valor da potencia para calcular "d0"
NPOWER = 2.0

# from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-11
K_MAX_PML = 1.0
ALPHA_MAX_PML = 2.0 * PI * (f0 / 2.0)  # from Festa and Vilotte

# Arrays para as variaveis de memoria do calculo
memory_dvx_dx = np.zeros(nx, dtype=flt32)
memory_dsigmaxx_dx = np.zeros(nx, dtype=flt32)

vx = np.zeros(nx, dtype=flt32)
vx_pr = np.zeros(nx, dtype=flt32)
vx_nx = np.zeros(nx, dtype=flt32)
sigmaxx = np.zeros(nx, dtype=flt32)
sigmaxx_pr = np.zeros(nx, dtype=flt32)
sigmaxx_nx = np.zeros(nx, dtype=flt32)
value_dvx_dx = np.zeros(nx, dtype=flt32)
value_dsigmaxx_dx = np.zeros(nx, dtype=flt32)

# Inicializacao dos parametros da PML (definicao dos perfis de absorcao na regiao da PML)
thickness_pml_x = npoints_pml * dx

# Coeficiente de reflexao (INRIA report section 6.1) http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
rcoef = 0.0001

print(f'1D elastic finite-difference code in velocity and stress formulation with C-PML')
print(f'NX = {nx}')

if NPOWER < 1:
    raise ValueError('NPOWER deve ser maior que 1')

# Calculo de d0 do relatorio da INRIA section 6.1 http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf
d0_x = -(NPOWER + 1) * cp * np.log(rcoef) / (2.0 * thickness_pml_x)
print(f'd0_x = {d0_x}')

pad_deriv = 2*(coefs.shape[0] - 1)
# Amortecimento na direcao "x" (horizontal)
# Origem da PML (posicao das bordas direita e esquerda menos a espessura, em unidades de distancia)
x_orig_left = thickness_pml_x
x_orig_right = (nx - pad_deriv - 1) * dx - thickness_pml_x

# Perfil de amortecimento na direcao "x" dentro do grid
i = np.arange(nx - pad_deriv)
xval = dx * i
xval_pml_left = x_orig_left - xval
xval_pml_right = xval - x_orig_right
x_pml_mask_left = np.where(xval_pml_left < 0.0, False, True)
x_pml_mask_right = np.where(xval_pml_right < 0.0, False, True)
x_mask = np.logical_or(x_pml_mask_left, x_pml_mask_right)
x_pml = np.zeros(nx - pad_deriv)
x_pml[x_pml_mask_left] = xval_pml_left[x_pml_mask_left]
x_pml[x_pml_mask_right] = xval_pml_right[x_pml_mask_right]
x_norm = x_pml / thickness_pml_x
d_x = (d0_x * x_norm ** NPOWER).astype(flt32)
k_x = (1.0 + (K_MAX_PML - 1.0) * x_norm ** NPOWER).astype(flt32)
alpha_x = (ALPHA_MAX_PML * (1.0 - np.where(x_mask, x_norm, 1.0))).astype(flt32)
b_x = np.exp(-(d_x / k_x + alpha_x) * dt).astype(flt32)
a_x = np.zeros(nx - pad_deriv, dtype=flt32)
i = np.where(d_x > 1e-6)
a_x[i] = d_x[i] * (b_x[i] - 1.0) / (k_x[i] * (d_x[i] + k_x[i] * alpha_x[i]))

# Perfil de amortecimento na direcao "x" dentro do meio grid (staggered grid)
xval_pml_left = x_orig_left - (xval + dx / 2.0)
xval_pml_right = (xval + dx / 2.0) - x_orig_right
x_pml_mask_left = np.where(xval_pml_left < 0.0, False, True)
x_pml_mask_right = np.where(xval_pml_right < 0.0, False, True)
x_mask_half = np.logical_or(x_pml_mask_left, x_pml_mask_right)
x_pml = np.zeros(nx - pad_deriv)
x_pml[x_pml_mask_left] = xval_pml_left[x_pml_mask_left]
x_pml[x_pml_mask_right] = xval_pml_right[x_pml_mask_right]
x_norm = x_pml / thickness_pml_x
d_x_half = (d0_x * x_norm ** NPOWER).astype(flt32)
k_x_half = (1.0 + (K_MAX_PML - 1.0) * x_norm ** NPOWER).astype(flt32)
alpha_x_half = (ALPHA_MAX_PML * (1.0 - np.where(x_mask_half, x_norm, 1.0))).astype(flt32)
b_x_half = np.exp(-(d_x_half / k_x_half + alpha_x_half) * dt).astype(flt32)
a_x_half = np.zeros(nx - pad_deriv, dtype=flt32)
i = np.where(d_x_half > 1e-6)
a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (k_x_half[i] * (d_x_half[i] + k_x_half[i] * alpha_x_half[i]))

# Imprime a posicao das fontes e dos receptores
print(f'Existem {NSRC} fontes')
for isrc in range(NSRC):
    print(f'Fonte {isrc}: i = {ix_src[isrc]:3}')

print(f'\nExistem {NREC} receptores')
for irec in range(NREC):
    print(f'Receptor {irec}: i = {ix_rec[irec]:3}')

# Arrays para armazenamento dos sinais dos sensores
sisvx = np.zeros((NSTEP, NREC), dtype=flt32)

# Verifica a condicao de estabilidade de Courant
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
courant_number = cp * dt / dx
print(f'\nNumero de Courant e {courant_number}')
if courant_number > 1:
    print("O passo de tempo e muito longo, a simulacao sera instavel")
    exit(1)

# Inicializa a figura que apresenta a simulacao em tempo real
ix_min = simul_roi.get_ix_min()
ix_max = simul_roi.get_ix_max()
fig, ax = plt.subplots()
ax.plot(simul_roi.w_points, vx[ix_min:ix_max])
ax.set_xlim(0.0, dx * nx)
ax.set_ylim(vx_min, vx_max)

# Roda a simulacao
sim_cpu()
print("Fim da simulação !!!!!")

# Plota o sinal da fonte
plt.figure()
plt.plot(t, source_term)
plt.title('Sinal da fonte')

# Plota as velocidades tomadas no sensores
for irec in range(NREC):
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    fig.suptitle(f'Receptor {irec + 1} [CPU]')
    ax.plot(sisvx[:, irec])
    ax.set_title(r'$V_x$')

plt.show()

