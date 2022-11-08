import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time

# ==========================================================
# Esse arquivo contém as simulações realizadas dentro da GPU.
# ==========================================================

dtype = np.float32

# Field config
nx = 256  # number of grid points in x-direction
nz = 256  # number of grid points in z-direction
nt = 1000  # number of time steps
c = np.zeros((nz, nx), dtype=dtype)  # wave velocity field
c[:, :] = .2
c[nz // 2 - 10:nz // 2 + 10, nx // 2 - 10:nx // 2 + 10] = 0.0  # add reflector in the center of the grid

# Escolha do valor de wsx
wsx = 1
for i in range(15, 0, -1):
    if (nz % i) == 0:
        wsx = i  # workgroup x size
        break

# Escolha do valor de wsy
wsy = 1
for i in range(15, 0, -1):
    if (nx % i) == 0:
        wsy = i  # workgroup x size
        break

# Source term
src_x = nx // 2  # source location in x-direction
src_z = 0  # source location in z-direction
t = np.arange(nt)
src = np.exp(-(t - nt / 10) ** 2 / 500, dtype=np.float32)
src[:-1] -= src[1:]

# Sensor signal
sens_x = nx // 2
sens_z = nz // 3
sensor = np.zeros(nt, dtype=np.float32)


def lap_pipa(u, cl):
    v = (2.0 * cl[0] * u).astype(dtype)
    for k in range(1, len(cl)):
        v[k:, :] += cl[k] * u[:-k, :]  # v[z + k, x] = v[z + k, x] + c[k] * u[z, x]  -- k acima
        v[:-k, :] += cl[k] * u[k:, :]  # v[z - k, x] = v[z - k, x] + c[k] * u[z, x]  -- k abaixo
        v[:, k:] += cl[k] * u[:, :-k]  # v[z, x + k] = v[z, x + k] + c[k] * u[z, x]  -- k a esquerda
        v[:, :-k] += cl[k] * u[:, k:]  # v[z, x - k] = v[z, x - k] + c[k] * u[z, x]  -- k a direita
    return v


# Simulação SERIAL
def sim_full():
    u = np.zeros((nz, nx, nt), dtype=dtype)

    for k in range(2, nt):
        u[:, :, k] = -u[:, :, k - 2] + 2.0 * u[:, :, k - 1] + c * c * lap_pipa(u[:, :, k - 1], c8)
        u[src_z, src_x, k] += src[k]

    return u[:, :, -1]


# Shader [kernel] para a simulação com Lap FOR
shader_test = f"""
    struct LapIntValues {{
        z_sz: i32,          // Z field size
        x_sz: i32,          // X field size
        z_src: i32,         // Z source
        x_src: i32,         // X source
        z_sens: i32,        // Z sensor
        x_sens: i32,        // X sensor
        num_coef: i32,      // num of discrete coefs
        k: i32              // iteraction
    }};

    // Group 0 - parameters
    @group(0) @binding(0)   // info_int buffer
    var<storage,read_write> liv: LapIntValues;
    
    @group(0) @binding(1) // info_float buffer
    var<storage,read> coef: array<f32>;
    
    @group(0) @binding(5) // source term
    var<storage,read> src: array<f32>;
    
    // Group 1 - simulation arrays
    @group(1) @binding(2) // pressure field k
    var<storage,read_write> PK: array<f32>;
    
    @group(1) @binding(3) // pressure field k-1 
    var<storage,read_write> PKm1: array<f32>;
    
    @group(1) @binding(4) // pressure field k-2
    var<storage,read_write> PKm2: array<f32>;
    
    @group(1) @binding(8) // laplacian matrix
    var<storage,read_write> lap: array<f32>;
    
    @group(1) @binding(7) // velocity map
    var<storage,read> c: array<f32>;
    
    // Group 2 - sensors arrays
    @group(2) @binding(6) // sensor signal
    var<storage,read_write> sensor: array<f32>;
    
    // function to convert 2D [z,x] index into 1D [zx] index
    fn zx(z: i32, x: i32) -> i32 {{
        let index = x + z * liv.x_sz;
        
        return select(-1, index, z >= 0 && z < liv.z_sz && x >= 0 && x < liv.x_sz);
    }}
    
    // function to get an PK array value
    fn getPK(z: i32, x: i32) -> f32 {{
        let index: i32 = zx(z, x);
        
        return select(0.0, PK[index], index != -1);
    }}
    
    // function to set a PK array value
    fn setPK(z: i32, x: i32, val : f32) {{
        let index: i32 = zx(z, x);
        
        if(index != -1) {{
            PK[index] = val;
        }}
    }}
    
    // function to get an PKm1 array value
    fn getPKm1(z: i32, x: i32) -> f32 {{
        let index: i32 = zx(z, x);
        
        return select(0.0, PKm1[index], index != -1);
    }}
    
    // function to set a PKm1 array value
    fn setPKm1(z: i32, x: i32, val : f32) {{
        let index: i32 = zx(z, x);
        
        if(index != -1) {{
            PKm1[index] = val;
        }}
    }} 

    // function to get an PKm2 array value
    fn getPKm2(z: i32, x: i32) -> f32 {{
        let index: i32 = zx(z, x);
        
        return select(0.0, PKm2[index], index != -1);
    }}
    
    // function to set a PKm2 array value
    fn setPKm2(z: i32, x: i32, val : f32) {{
        let index: i32 = zx(z, x);
        
        if(index != -1) {{
            PKm2[index] = val;
        }}
    }} 

    // function to calculate laplacian
    @stage(compute)
    @workgroup_size({wsx}, {wsy})
    fn laplacian(@builtin(global_invocation_id) index: vec3<u32>) {{
        let z: i32 = i32(index.x);          // z thread index
        let x: i32 = i32(index.y);          // x thread index
        let num_coef: i32 = liv.num_coef;   // num coefs
        let idx: i32 = zx(z, x);
             
        // central
        if(idx != -1) {{
            lap[idx] = 2.0 * coef[0] * getPKm1(z, x);
   
            for (var i = 1; i < num_coef; i = i + 1) {{
                lap[idx] += coef[i] * (getPKm1(z - i, x) +  // i acima
                                       getPKm1(z + i, x) +  // i abaixo
                                       getPKm1(z, x - i) +  // i a esquerda
                                       getPKm1(z, x + i));  // i a direita
            }}
        }}
    }}

    @stage(compute)
    @workgroup_size(1)
    fn incr_k() {{
        liv.k += 1;
    }}

    @stage(compute)
    @workgroup_size({wsx}, {wsy})
    fn sim(@builtin(global_invocation_id) index: vec3<u32>) {{
        var add_src: f32 = 0.0;             // Source term
        let z: i32 = i32(index.x);          // z thread index
        let x: i32 = i32(index.y);          // x thread index
        let z_src: i32 = liv.z_src;         // source term z position
        let x_src: i32 = liv.x_src;         // source term x position
        let idx: i32 = zx(z, x);

        // --------------------
        // Update pressure field
        add_src = select(0.0, src[liv.k], z == z_src && x == x_src);
        setPK(z, x, -1.0*getPKm2(z, x) + 2.0*getPKm1(z, x) + c[idx]*c[idx]*lap[idx] + add_src);
            
        // --------------------
        // Circular buffer
        setPKm2(z, x, getPKm1(z, x));
        setPKm1(z, x, getPK(z, x));
        
        if(z == liv.z_sens && x == liv.x_sens) {{
            sensor[liv.k] = getPK(z, x);
        }}
    }}
    """


# Simulação completa na GPU com LAP FOR
def sim_webgpu_for(coef):
    # pressure field
    PKm2 = np.zeros((nz, nx), dtype=dtype)  # k-2
    PKm1 = np.zeros((nz, nx), dtype=dtype)  # k-1
    PK = np.zeros((nz, nx), dtype=dtype)  # k
    lap = np.zeros((nz, nx), dtype=dtype)  # k

    # source term
    src_gpu = src.astype(dtype)

    # auxiliar variables
    info_i32 = np.array([nz, nx, src_z, src_x, sens_z, sens_x, len(coef), 0], dtype=np.int32)
    info_f32 = coef.astype(dtype)

    # =====================
    # webgpu configurations
    device = wgpu.utils.get_default_device()
    # adapter = wgpu.request_adapter(canvas=None, power_preference="low-power")
    # device = adapter.request_device()
    print(f'GPU: {device.adapter.properties["name"]}')
    cshader = device.create_shader_module(code=shader_test)

    # info integer buffer
    b0 = device.create_buffer_with_data(data=info_i32, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_SRC)
    # info float buffer
    b1 = device.create_buffer_with_data(data=info_f32, usage=wgpu.BufferUsage.STORAGE |
                                                             wgpu.BufferUsage.COPY_SRC)
    # field pressure at time k-2
    b2 = device.create_buffer_with_data(data=PKm2, usage=wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_SRC)
    # field pressure at time k-1
    b3 = device.create_buffer_with_data(data=PKm1, usage=wgpu.BufferUsage.STORAGE |
                                                         wgpu.BufferUsage.COPY_SRC)
    # field pressure at time k
    b4 = device.create_buffer_with_data(data=PK, usage=wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_DST |
                                                       wgpu.BufferUsage.COPY_SRC)
    # source term
    b5 = device.create_buffer_with_data(data=src_gpu, usage=wgpu.BufferUsage.STORAGE |
                                                            wgpu.BufferUsage.COPY_SRC)

    # sensor signal
    b6 = device.create_buffer_with_data(data=sensor, usage=wgpu.BufferUsage.STORAGE |
                                                           wgpu.BufferUsage.COPY_DST |
                                                           wgpu.BufferUsage.COPY_SRC)

    # velocity map
    b7 = device.create_buffer_with_data(data=c, usage=wgpu.BufferUsage.STORAGE |
                                                      wgpu.BufferUsage.COPY_SRC)

    # laplacian matrix
    b8 = device.create_buffer_with_data(data=lap, usage=wgpu.BufferUsage.STORAGE |
                                                        wgpu.BufferUsage.COPY_DST |
                                                        wgpu.BufferUsage.COPY_SRC)

    binding_layouts_params = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 5,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
    ]
    binding_layouts_sim_arrays = [
        {
            "binding": 2,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 3,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 4,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 8,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 7,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
    ]
    binding_layouts_sensors = [
        {
            "binding": 6,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]

    bindings_params = [
        {
            "binding": 0,
            "resource": {"buffer": b0, "offset": 0, "size": b0.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b1, "offset": 0, "size": b1.size},
        },
        {
            "binding": 5,
            "resource": {"buffer": b5, "offset": 0, "size": b5.size},
        },
    ]
    bindings_sim_arrays = [
        {
            "binding": 2,
            "resource": {"buffer": b2, "offset": 0, "size": b2.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b3, "offset": 0, "size": b3.size},
        },
        {
            "binding": 4,
            "resource": {"buffer": b4, "offset": 0, "size": b4.size},
        },
        {
            "binding": 8,
            "resource": {"buffer": b8, "offset": 0, "size": b8.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b7, "offset": 0, "size": b7.size},
        },
    ]
    bindings_sensors = [
        {
            "binding": 6,
            "resource": {"buffer": b6, "offset": 0, "size": b6.size},
        },
    ]

    # Put everything together
    bind_group_layout_0 = device.create_bind_group_layout(entries=binding_layouts_params)
    bind_group_layout_1 = device.create_bind_group_layout(entries=binding_layouts_sim_arrays)
    bind_group_layout_2 = device.create_bind_group_layout(entries=binding_layouts_sensors)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout_0,
                                                                        bind_group_layout_1,
                                                                        bind_group_layout_2])
    bind_group_0 = device.create_bind_group(layout=bind_group_layout_0, entries=bindings_params)
    bind_group_1 = device.create_bind_group(layout=bind_group_layout_1, entries=bindings_sim_arrays)
    bind_group_2 = device.create_bind_group(layout=bind_group_layout_2, entries=bindings_sensors)

    # Create and run the pipeline
    compute_sim = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "sim"},
    )
    compute_lap = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "laplacian"},
    )
    compute_incr_k = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "incr_k"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()

    compute_pass.set_bind_group(0, bind_group_0, [], 0, 999999)  # last 2 elements not used
    compute_pass.set_bind_group(1, bind_group_1, [], 0, 999999)  # last 2 elements not used
    compute_pass.set_bind_group(2, bind_group_2, [], 0, 999999)  # last 2 elements not used
    for i in range(nt):
        compute_pass.set_pipeline(compute_lap)
        compute_pass.dispatch_workgroups(nz // wsx, nx // wsy)

        compute_pass.set_pipeline(compute_sim)
        compute_pass.dispatch_workgroups(nz // wsx, nx // wsy)

        compute_pass.set_pipeline(compute_incr_k)
        compute_pass.dispatch_workgroups(1)

    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    out = device.queue.read_buffer(b3).cast("f")  # reads from buffer 3
    sens = np.array(device.queue.read_buffer(b6).cast("f"))
    return np.asarray(out).reshape((nz, nx)), sens


# --------------------------
# COEFFICIENTS
# finite difference coefficient
c2 = np.array([-2, 1], dtype=dtype)
c4 = np.array([-5 / 2, 4 / 3, -1 / 12], dtype=dtype)
c6 = np.array([-49 / 18, 3 / 2, -3 / 20, 1 / 90], dtype=dtype)
c8 = np.array([-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560], dtype=dtype)

# --------------------------
# webgpu
t_for = time()
u_for, sensor = sim_webgpu_for(c8)
t_for = time() - t_for
# serial
t_ser = time()
u_ser = sim_full()
t_ser = time() - t_ser

print(f'workgroups X: {wsx}; workgroups Y: {wsy}')
print(f'TEMPO - {nt} pontos de tempo:\nFor: {t_for:.3}s\nSerial: {t_ser:.3}s')
print(f'MSE entre as simulações: {mean_squared_error(u_ser, u_for)}')

plt.figure(1)
plt.title('Full sim. na GPU com lap for')
plt.imshow(u_for, aspect='auto', cmap='turbo_r')
plt.figure(2)
plt.title('Full sim. na CPU')
plt.imshow(u_ser, aspect='auto', cmap='turbo_r')
plt.figure(3)
plt.title(f'Sensor em z = {sens_z} e x = {sens_x}')
plt.plot(t, sensor)
plt.show()
