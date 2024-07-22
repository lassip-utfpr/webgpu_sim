import wgpu
import numpy as np
from numpy import pi
import ast
from scipy.signal import gausspulse
from simul_utils import SimulationROI, SimulationProbeLinearArray
import matplotlib.pyplot as plt
import json

# Definicao das constantes para a o calculo das derivadas, seguindo Lui 2009 (10.1111/j.1365-246X.2009.04305.x)
coefs_Lui = [
    [9.0 / 8.0, -1.0 / 24.0],
    [75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0],
    [1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0],
    [19845.0 / 16384.0, -735.0 / 8192.0, 567.0 / 40960.0, -405.0 / 229376.0, 35.0 / 294912.0],
    [160083.0 / 131072.0, -12705.0 / 131072.0, 22869.0 / 1310720.0, -5445.0 / 1835008.0, 847.0 / 2359296.0,
     -63.0 / 2883584.0]
]

# Configurações Iniciais
flt32 = np.float32
device_gpu = wgpu.utils.get_default_device()
IT_DISPLAY = 10

with open('config1D.json', 'r') as d:
    configs2 = json.load(d)

ord = configs2["simul_params"]["ord"]
coefs = np.array(coefs_Lui[configs2["simul_params"]["ord"] - 2], dtype=flt32)
data_src = configs2["probes"][0]["linear"]["coord_center"][:]
t_src = configs2["probes"][0]["linear"]["t0_emmition"]*1e-6
data_src.append(t_src)

data_rec = configs2["probes"][0]["linear"]["coord_center"][:]
t_rec= configs2["probes"][0]["linear"]["t0_reception"]*1e-6
data_rec.append(t_rec)

source_term_cfg = [configs2["probes"][0]["linear"]["pulse_type"]]
source_term_cfg.append(configs2["probes"][0]["linear"]["freq"])
source_term_cfg.append(configs2["probes"][0]["linear"]["bw"])
source_term_cfg.append(configs2["probes"][0]["linear"]["gain"])

data_src = np.array([data_src])
data_rec = np.array([data_rec])

with open('config1D.json', 'r') as f:
    configs = ast.literal_eval(f.read())




    simul_roi = SimulationROI(**configs["roi"], pad=coefs.shape[0])
    wavenumber_x = (2.0 * pi * source_term_cfg[1]) / configs["specimen_params"]["cp"]




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

# Define a posicao das fontes

NSRC = data_src.shape[0]

i_src = np.array([simul_roi.get_nearest_grid_idx(d[0:3]) for d in data_src])
ix_src = i_src[:, 0].astype(np.int32)
aux_src = ix_src[0]

# Parametros da fonte

f0 = source_term_cfg[1]  # frequencia [MHz]
t0 = data_src[:, 3].reshape((1, NSRC))
factor = flt32(source_term_cfg[3])
t = np.expand_dims(np.arange(NSTEP) * dt, axis=1)

# Gauss pulse

source_term = factor * flt32(gausspulse((t - t0), fc=f0, bw=source_term_cfg[2])).astype(flt32)

# Define a localizacao dos receptores
NREC = data_rec.shape[0]

i_rec = np.array([simul_roi.get_nearest_grid_idx(p[0:3]) for p in data_rec])
ix_rec = i_rec[:, 0].astype(np.int32)

# Valor da potencia para calcular "d0"
NPOWER = 2.0

vx = np.zeros(nx, dtype=flt32)
sigmaxx = np.zeros(nx, dtype=flt32)
sisvx = np.zeros((NSTEP, NREC), dtype=flt32)

wsx = nx

def sim_1D_wgpu(device):
    global source_term
    global vx, sigmaxx
    global ix_rec, ix_src
    global sisvx
    global coefs, ord

    # Arrays com parametros
    params_i32 = np.array([nx, aux_src, NSTEP, NREC, 0, ord], dtype=np.int32)
    params_f32 = np.array([dx, dt, rho, lambda_, mu, lambdaplus2mu], dtype=flt32)

    # Rodar shader
    with open('shader_1D_elast.wgsl') as shader_file:
        cshader_string = shader_file.read()
        cshader_string = cshader_string.replace('wsx', f'{wsx}')
        cshader = device.create_shader_module(code=cshader_string)


    # Buffers

    # Binding 0
    bf_param_flt32 = device.create_buffer_with_data(data=params_f32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)

    # Binding 1
    bf_src = device.create_buffer_with_data(data=np.column_stack((source_term)),
                                             usage=wgpu.BufferUsage.STORAGE |
                                                   wgpu.BufferUsage.COPY_SRC)

    # Binding 2
    bf_param_int32 = device.create_buffer_with_data(data=params_i32, usage=wgpu.BufferUsage.STORAGE |
                                                                          wgpu.BufferUsage.COPY_SRC)

    # Binding 3
    bf_vx = device.create_buffer_with_data(data=np.vstack(vx), usage=wgpu.BufferUsage.STORAGE |
                                                                                wgpu.BufferUsage.COPY_DST |
                                                                                wgpu.BufferUsage.COPY_SRC)

    # Binding 4
    bf_sigma = device.create_buffer_with_data(data=np.vstack(sigmaxx),
                                           usage=wgpu.BufferUsage.STORAGE |
                                                 wgpu.BufferUsage.COPY_DST |
                                                 wgpu.BufferUsage.COPY_SRC)

    # Binding 5
    bf_sensx = device.create_buffer_with_data(data=sisvx, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_DST |
                                                                wgpu.BufferUsage.COPY_SRC)

    # Binding 6
    bf_pos_x = device.create_buffer_with_data(data=ix_rec, usage=wgpu.BufferUsage.STORAGE |
                                                                     wgpu.BufferUsage.COPY_SRC)

    # binding 7
    bf_coefs = device.create_buffer_with_data(data=coefs, usage=wgpu.BufferUsage.STORAGE |
                                                                wgpu.BufferUsage.COPY_SRC)




    # Binding Layouts

    bl_params = [
        {"binding": 0,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {"binding": 1,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         },
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,}
        },
        {
            "binding": 7,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage, }
        }
    ]

    bl_sim = [
        {"binding": 3,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 4,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         }
    ]

    bl_sensors = [
        {"binding": 5,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.storage}
         },
        {"binding": 6,
         "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {
             "type": wgpu.BufferBindingType.read_only_storage}
         }
    ]


    # Bindings

    b_params = [
        {
            "binding": 0,
            "resource": {"buffer": bf_param_flt32, "offset": 0, "size": bf_param_flt32.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": bf_src, "offset": 0, "size": bf_src.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": bf_param_int32, "offset": 0, "size": bf_param_int32.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": bf_coefs, "offset": 0, "size": bf_coefs.size},
        }
    ]
    b_sim = [
        {
            "binding": 3,
            "resource": {"buffer": bf_vx, "offset": 0, "size": bf_vx.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": bf_sigma, "offset": 0, "size": bf_sigma.size},
        },
    ]
    b_sensors = [
        {
            "binding": 5,
            "resource": {"buffer": bf_sensx, "offset": 0, "size": bf_sensx.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": bf_pos_x, "offset": 0, "size": bf_pos_x.size},
        },
    ]

    # B Layouts + Bindings + Pipeline Layout
    bgl_0 = device.create_bind_group_layout(entries=bl_params)
    bgl_1 = device.create_bind_group_layout(entries=bl_sim)
    bgl_2 = device.create_bind_group_layout(entries=bl_sensors)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_0, bgl_1, bgl_2])
    bg_0 = device.create_bind_group(layout=bgl_0, entries=b_params)
    bg_1 = device.create_bind_group(layout=bgl_1, entries=b_sim)
    bg_2 = device.create_bind_group(layout=bgl_2, entries=b_sensors)

    # Pipeline config
    compute_sigmax = device.create_compute_pipeline(layout=pipeline_layout,
                                                          compute={"module": cshader, "entry_point": "sigmax"})
    compute_velx = device.create_compute_pipeline(layout=pipeline_layout,
                                                             compute={"module": cshader,
                                                                      "entry_point": "velx"})
    compute_sensx = device.create_compute_pipeline(layout=pipeline_layout,
                                                              compute={"module": cshader,
                                                                       "entry_point": "sensx"})
    compute_incr_it = device.create_compute_pipeline(layout=pipeline_layout,
                                                            compute={"module": cshader,
                                                                     "entry_point": "incr_it"})



    for it in range(1, NSTEP + 1):

        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()

        compute_pass.set_bind_group(0, bg_0, [], 0, 999999)
        compute_pass.set_bind_group(1, bg_1, [], 0, 999999)
        compute_pass.set_bind_group(2, bg_2, [], 0, 999999)

        compute_pass.set_pipeline(compute_sigmax)
        compute_pass.dispatch_workgroups(wsx)

        compute_pass.set_pipeline(compute_velx)
        compute_pass.dispatch_workgroups(wsx)

        compute_pass.set_pipeline(compute_sensx)
        compute_pass.dispatch_workgroups(wsx)

        compute_pass.set_pipeline(compute_incr_it)
        compute_pass.dispatch_workgroups(1)

        compute_pass.end()
        device.queue.submit([command_encoder.finish()])

        vxgpu = np.asarray(device.queue.read_buffer(bf_vx, buffer_offset=0, size=vx.size * 4).cast("f")).reshape(nx)
        sens_vx = np.array(device.queue.read_buffer(bf_sensx).cast("f")).reshape((NSTEP, NREC))

        if (it % IT_DISPLAY) == 0 or it == 5:
            print(f'Time step # {it} out of {NSTEP}')
            # print(f'Max Vx = {np.max(vxgpu)}')
            # print(f'Min Vx = {np.min(vxgpu)}')
            ax.clear()
            ax.plot(simul_roi.w_points, vxgpu[ix_min:ix_max])
            ax.set_xlim(0.0, dx * nx)
            ax.set_ylim(vx_min, vx_max)
            plt.show(block=False)
            plt.pause(0.001)



    # Resultados

    return vxgpu, sens_vx




# Inicializa a figura que apresenta a simulacao em tempo real
vx_max = 100.0
vx_min = -vx_max
ix_min = simul_roi.get_ix_min()
ix_max = simul_roi.get_ix_max()
fig, ax = plt.subplots()
ax.plot(simul_roi.w_points, vx[ix_min:ix_max])
ax.set_xlim(0.0, dx * nx)
ax.set_ylim(vx_min, vx_max)


vx, sisvx = sim_1D_wgpu(device_gpu)

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