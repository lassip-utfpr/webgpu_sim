import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
import numpy as np
import matplotlib.pyplot as plt
from lap_webgpu import lap_pipa
from time import time

# ==========================================================
# Esse arquivo contém as simulações realizadas dentro da GPU.
# ==========================================================

dtype = np.float32

wsx = 20
wsy = 20
wsz = 2

# Field config
nx = 256  # number of grid points in x-direction
nz = 256  # number of grid points in z-direction
nt = 100  # number of time steps
c = .2  # wave velocity
# Source term
src_x = nx // 2  # source location in x-direction
src_z = nz // 2  # source location in z-direction
t = np.arange(nt)
src = np.exp(-(t - nt / 10) ** 2 / 500, dtype=np.float32)
src[:-1] -= src[1:]


# Simulação SERIAL
def sim_full():
    u = np.zeros((nz, nx, nt), dtype=dtype)

    for k in range(2, nt):
        u[:, :, k] = -u[:, :, k - 2] + 2.0 * u[:, :, k - 1] + c * c * lap_pipa(u[:, :, k - 1], c8)
        u[src_z, src_z, k] += src[k]

    return u


# Shader [kernel] para a simulação com Lap FOR
shader_test = f"""
    struct LapValues {{
        z_sz: f32,          // Z field size
        x_sz: f32,          // X field size
        num_coef: f32,      // num of discrete coefs
        c: f32,             // velocity
        k: f32,
        coef: array<f32>    // discrete coefs
    }};

    //@group(1) @binding(0)   // info buffer
    @group(0) @binding(0)   // info buffer
    var<storage,read_write> lv: LapValues;
    //var<storage,read> lv: LapValues;

    @group(0) @binding(1) // pressure field k
    var<storage,read_write> PK: array<f32>;
    
    @group(0) @binding(2) // pressure field k-1 
    var<storage,read_write> PKm1: array<f32>;
    
    @group(0) @binding(3) // pressure field k-2
    var<storage,read_write> PKm2: array<f32>;
    
    @group(0) @binding(4) // source term
    var<storage,read_write> src: array<f32>;
    
    @group(0) @binding(5) // source term
    var<storage,read_write> src_out: array<f32>;
    
    @group(0) @binding(6) // source term
    var<storage,read_write> diff: array<f32>;
    
    @group(0) @binding(7) // source term
    var<storage,read_write> b7: array<f32>;
    
    // function to convert 2D [z,x] index into 1D [zx] index
    fn zx(z: i32, x: i32) -> i32 {{
        let x_sz: i32 = i32(lv.x_sz);
        let z_sz: i32 = i32(lv.z_sz);
        let index = i32(x + z*x_sz);
        
        if(z >= 0 && z < z_sz && x >= 0 && x < x_sz) {{
            return index;
        }}
        else
        {{
            return -1;
        }}    
    }}
    
    // function to get an PK array value
    fn getPK(z: i32, x: i32) -> f32 {{
        let index = zx(z, x);
        
        if(index != -1) {{
            return PK[index];
        }}
        else
        {{
            return 0.0;
        }}    
    }}
    
    // function to set a PK array value
    fn setPK(z: i32, x: i32, val : f32) {{
        let index = zx(z, x);
        
        if(index != -1) {{
            PK[index] = val;
        }}
    }}
    
    // function to get an PKm1 array value
    fn getPKm1(z: i32, x: i32) -> f32 {{
        let index: i32 = zx(z, x);
        
        if(index != -1) {{
            return PKm1[index];
        }}
        else
        {{
            return 0.0;
        }}    
    }}
    
    // function to set a PKm1 array value
    fn setPKm1(z: i32, x: i32, val : f32) {{
        let index = zx(z, x);
        
        if(index != -1) {{
            PKm1[index] = val;
        }}
    }} 

    // function to get an PKm2 array value
    fn getPKm2(z: i32, x: i32) -> f32 {{
        let index = zx(z, x);
        
        if(index != -1) {{
            return PKm2[index];
        }}
        else
        {{
            return 0.0;
        }}    
    }}
    
    // function to set a PKm2 array value
    fn setPKm2(z: i32, x: i32, val : f32) {{
        let index = zx(z, x);
        
        if(index != -1) {{
            PKm2[index] = val;
        }}
    }} 

    
    // function to calculate laplacian
    fn laplacian(z: i32, x: i32) -> f32 {{
        let num_coef: i32 = i32(lv.num_coef);   // num coefs
             
        // central
        var lap: f32 = 2.0 * f32(lv.coef[0]) * getPKm1(z, x);
   
        for (var i = 1; i < num_coef; i = i + 1) {{
            let u_z_up:   f32 = getPKm1(z - i, x);     // i acima
            let u_z_down: f32 = getPKm1(z + i, x);     // i abaixo 
            let u_x_left:  f32 = getPKm1(z, x - i);    // i a esquerda
            let u_x_right: f32 = getPKm1(z, x + i);    // i a direita
            
            lap += lv.coef[i] * u_z_up;
            lap += lv.coef[i] * u_z_down;
            lap += lv.coef[i] * u_x_left;
            lap += lv.coef[i] * u_x_right;
        }}
        
        return lap;
    }}

    @stage(compute)
    @workgroup_size({nz}) // z --> num threads per block -- hardcoded
    fn main(@builtin(local_invocation_id) threadIdx: vec3<u32>,
            @builtin(workgroup_id) blockIdx: vec3<u32>) {{

        var add_src: f32 = 0.0;                 // 'boolean' to check if thread is in source position
        let c: f32 = lv.c;                      // velocity

        let z: i32 = i32(threadIdx.x);          // z thread index
        let x: i32 = i32(blockIdx.x);           // x thread index
        let z_src: i32 = i32(lv.z_sz)/2;        // source term z position
        let x_src: i32 = i32(lv.x_sz)/2;        // source term x position
        var lap: f32 = 0.0;                     // laplacian
        // let k: i32 = i32(lv.k);

        // --------------------
        // Calculate laplacian
        lap = laplacian(z, x);
            
        // --------------------
        // Update pressure field
        add_src = f32(!bool(zx(z,x)-zx(z_src,x_src)));
        //setPK(z, x, -1.0*getPKm2(z, x) + 2.0*getPKm1(z, x) + c*c*lap + b7[0]*add_src);
        //setPK(z, x, -1.0*getPKm2(z, x) + 2.0*getPKm1(z, x) + c*c*lap + src_out[0]*add_src);
        setPK(z, x, -1.0*getPKm2(z, x) + 2.0*getPKm1(z, x) + c*c*lap + src[i32(lv.k)]*add_src);
            
        // --------------------
        // Circular buffer
        setPKm2(z, x, getPKm1(z, x));
        setPKm1(z, x, getPK(z, x));
        
        if(z == 0 && x == 0) {{
            src_out[i32(lv.k)] = src[i32(lv.k)];
            //src_out[i32(lv.k)] = lv.k;
            //diff[i32(lv.k)] = b7[0];
            //lv.k += 1.0;
        }}
        //if(z == 0 && x == x_src) {{
            //src_out[i32(lv.k)] = getPK(z, x);
            //src_out[i32(lv.k)] = lv.k;
            //lv.k += 1.0;
        //}}
    }}
    """

shader_increment = f"""
struct LapValues {{
        z_sz: f32,          // Z field size
        x_sz: f32,          // X field size
        num_coef: f32,      // num of discrete coefs
        c: f32,             // velocity
        k: f32,
        coef: array<f32>    // discrete coefs
    }};

    @group(1) @binding(0)   // info buffer
    var<storage,read_write> lv: LapValues;

    @stage(compute)
    @workgroup_size(1)
    fn main_inc(@builtin(global_invocation_id) index: vec3<u32>) {{
        lv.k += 1.0;
    }}
"""

# Simulação completa na GPU com LAP FOR
def sim_webgpu_for(coef):
    # pressure field
    PKm2 = np.zeros((nz, nx), dtype=dtype)  # k-2
    PKm1 = np.zeros((nz, nx), dtype=dtype)  # k-1
    PK = np.zeros((nz, nx), dtype=dtype)  # k
    # source term
    src_gpu = src.astype(dtype)
    # auxiliar variables
    info = np.array([nz, nx, len(coef), c, 0], dtype=dtype)
    # info = np.array([nz, nx, len(coef), c], dtype=dtype)
    info = np.concatenate((info, coef))
    # =====================
    # webgpu configurations
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader_test)
    # cshader_inc = device.create_shader_module(code=shader_increment)
    # info buffer
    b0 = device.create_buffer_with_data(data=info, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_SRC)
    # b0 = device.create_buffer_with_data(data=info, usage=wgpu.BufferUsage.STORAGE)
    # field pressure at time k-2
    b1 = device.create_buffer_with_data(data=PKm2, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # field pressure at time k-1
    b2 = device.create_buffer_with_data(data=PKm1, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # field pressure at time k
    b3 = device.create_buffer_with_data(data=PK, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_SRC)
    # source term
    b4 = device.create_buffer_with_data(data=src_gpu, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # b4 = device.create_buffer(size=4, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    b5 = device.create_buffer(size=src.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_SRC)
    b6 = device.create_buffer(size=src.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                     wgpu.BufferUsage.COPY_SRC)
    b7 = device.create_buffer(size=4, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)

    # Setup layouts and bindings
    # binding_layouts_lap_values = [
    #     {
    #         "binding": 0,
    #         "visibility": wgpu.ShaderStage.COMPUTE,
    #         "buffer": {
    #             # "type": wgpu.BufferBindingType.read_only_storage,
    #             "type": wgpu.BufferBindingType.storage,
    #         },
    #     },
    # ]
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                # "type": wgpu.BufferBindingType.read_only_storage,
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
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
            "binding": 5,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 6,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
        {
            "binding": 7,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]
    # bindings_lap_values = [
    #     {
    #         "binding": 0,
    #         "resource": {"buffer": b0, "offset": 0, "size": b0.size},
    #     },
    # ]
    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": b0, "offset": 0, "size": b0.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b1, "offset": 0, "size": b1.size},
        },
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
            "binding": 5,
            "resource": {"buffer": b5, "offset": 0, "size": b5.size},
        },
        {
            "binding": 6,
            "resource": {"buffer": b6, "offset": 0, "size": b6.size},
        },
        {
            "binding": 7,
            "resource": {"buffer": b7, "offset": 0, "size": b7.size},
        },
    ]

    # Put everything together
    # bind_group_layout_lp = device.create_bind_group_layout(entries=binding_layouts_lap_values)
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    # pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout_lp, bind_group_layout])
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    # pipeline_layout_inc = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout_lp])
    # bind_group_lp = device.create_bind_group(layout=bind_group_layout_lp, entries=bindings_lap_values)
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    # compute_pipeline_inc = device.create_compute_pipeline(
    #     layout=pipeline_layout_inc,
    #     compute={"module": cshader_inc, "entry_point": "main_inc"},
    # )
    command_encoder = device.create_command_encoder()
    # compute_pass = command_encoder.begin_compute_pass()
    # compute_pass.set_pipeline(compute_pipeline)
    # compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    for i in range(nt):
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
        # compute_pass.set_bind_group(1, bind_group_lp, [], 0, 999999)  # last 2 elements not used
        compute_pass.dispatch_workgroups(nx)  # x y z -- num blocks
        compute_pass.end()

        # device.queue.write_buffer(b5, 0, src[i].tobytes())
        # compute_pass = command_encoder.begin_compute_pass()
        # compute_pass.set_pipeline(compute_pipeline_inc)
        # compute_pass.set_bind_group(1, bind_group_lp, [], 0, 999999)  # last 2 elements not used
        # compute_pass.dispatch_workgroups(1)  # x y z -- num blocks
        # compute_pass.end()

    # compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    out = device.queue.read_buffer(b3).cast("f")  # reads from buffer 3
    src_out = np.asarray(device.queue.read_buffer(b5).cast("f"))
    df = np.asarray(device.queue.read_buffer(b6).cast("f"))
    return np.asarray(out).reshape((nz, nx))


shader_test_index = f"""
    struct Params {{
        x: i32,
        y: i32,
        z: i32,
        c: f32
    }};

    @group(0) @binding(0)   // Param buffer
    var<storage,read> param: Params;
    
    @group(0) @binding(1)   // local_invocation_id x
    var<storage,read_write> lid_x: array<f32>;
    @group(0) @binding(2)   // local_invocation_id y
    var<storage,read_write> lid_y: array<f32>;
    @group(0) @binding(3)   // local_invocation_id z
    var<storage,read_write> lid_z: array<f32>;
   
    // function to convert 2D [x, y] index into 1D index
    fn idx2D(x: i32, y: i32) -> i32 {{
        let x_sz: i32 = param.x;
        let y_sz: i32 = param.y;
        let index = x + y*x_sz;
        
        if(x >= 0 && x < x_sz && y >= 0 && y < y_sz) {{
            return index;
        }}
        else
        {{
            return -1;
        }}    
    }}

    @stage(compute)
    @workgroup_size(1, {wsy})
    fn main(@builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(local_invocation_index) index: u32) {{
            let idx_x : i32 = i32(lid.x);
            let idx_y : i32 = i32(lid.y);
            let idx : i32  = idx2D(idx_x, idx_y);
            
            lid_x[index] = f32(lid.x);
            lid_y[index] = f32(lid.y);
            lid_z[index] = f32(param.z);
    }}
    """


# Teste do indices
def test_indexwebgpu():
    #
    lid_x = np.zeros((wsx, wsy), dtype=np.float32)
    lid_y = np.zeros((wsx, wsy), dtype=np.float32)
    lid_z = np.zeros((wsx, wsy), dtype=np.float32)
    # lid_z = np.float32(1000.0) * np.ones((wsx, wsy), dtype=np.float32)

    # auxiliar variables
    param = np.array([wsx, wsy, wsz], dtype=np.int32)

    # =====================
    # webgpu configurations
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader_test_index)

    # param buffer
    b0 = device.create_buffer_with_data(data=param, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)

    # local_invocation_id
    b1 = device.create_buffer_with_data(data=lid_x, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                          wgpu.BufferUsage.COPY_SRC)
    b2 = device.create_buffer_with_data(data=lid_y, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                          wgpu.BufferUsage.COPY_SRC)
    b3 = device.create_buffer_with_data(data=lid_z, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                          wgpu.BufferUsage.COPY_SRC)

    # Setup layouts and bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
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
    ]
    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": b0, "offset": 0, "size": b0.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": b1, "offset": 0, "size": b1.size},
        },
        {
            "binding": 2,
            "resource": {"buffer": b2, "offset": 0, "size": b2.size},
        },
        {
            "binding": 3,
            "resource": {"buffer": b3, "offset": 0, "size": b3.size},
        },
    ]

    # Put everything together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    for k in range(10 * wsz):
        param[2] += 1
        compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
        compute_pass.dispatch_workgroups(wsx, 1)  # x y z -- num blocks
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    lid_x = np.asarray(device.queue.read_buffer(b1).cast("f"))  # .reshape((wsx, wsy, wsz))
    lid_y = np.asarray(device.queue.read_buffer(b2).cast("f"))  # .reshape((wsx, wsy, wsz))
    lid_z = np.asarray(device.queue.read_buffer(b3).cast("f"))  # .reshape((wsx, wsy, wsz))
    return


# test_indexwebgpu()

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
u_for = sim_webgpu_for(c8)
t_for = time() - t_for
# serial
t_ser = time()
u_ser = sim_full()
t_ser = time() - t_ser

print(f'TEMPO - {nt} pontos de tempo:\nFor: {t_for:.3}s\nSerial: {t_ser:.3}s')

plt.figure(1)
plt.title('Full sim. na GPU com lap for')
plt.imshow(u_for, aspect='auto', cmap='turbo_r')
plt.figure(2)
plt.title('Full sim. na CPU')
plt.imshow(u_ser[:, :, -1], aspect='auto', cmap='turbo_r')
plt.show()
