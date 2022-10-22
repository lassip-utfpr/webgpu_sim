import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
import numpy as np
import matplotlib.pyplot as plt
# from sim_webgpu import sim_full
from sklearn.metrics import mean_squared_error as MSE
from lap_webgpu import lap_pipa
from time import time, sleep

# Esse arquivo contém as simulações realizadas dentro da GPU.

dtype = np.float32

# Field config
nx = 1024  # number of grid points in x-direction
nz = 256  # number of grid points in z-direction
# nx = 15  # number of grid points in x-direction
# nz = 15  # number of grid points in z-direction
nt = 1000  # number of time steps
c = .2  # wave velocity
# Source term
src_x = nx // 2  # source location in x-direction
src_z = nz // 2  # source location in z-direction
t = np.arange(nt)
src = np.exp(-(t - nt / 10) ** 2 / 500, dtype=np.float32)
src[:-1] -= src[1:]
#src[-10:] = np.arange(10)
# src[-1] = 1
# plt.plot(t, src)


shader_for = f"""
    struct LapValues {{
        z_sz: f32,          // Z field size
        x_sz: f32,          // X field size
        num_coef: f32,      // num of discrete coefs
        c: f32,             // velocity
        k: f32,
        coef: array<f32>,   // discrete coefs
    }};

    @group(0) @binding(0)   // info buffer
    var<storage,read> lv: LapValues;

    @group(0) @binding(1) // pressure field k-2 
    var<storage,read_write> PKm2: array<f32>;

    @group(0) @binding(2) // pressure field k-1 
    var<storage,read_write> PKm1: array<f32>;

    @group(0) @binding(3) // pressure field k 
    var<storage,read_write> PK: array<f32>;
    
    @group(0) @binding(4) // source term
    var<storage,read> src: array<f32>;
    
    @group(1) @binding(5) // source term
    var<storage,read_write> kk: array<f32>;

    // function to convert 2D [z,x] index into 1D [zx] index
    fn zx(z: i32, x: i32) -> u32 {{
        let x_sz: i32 = i32(lv.x_sz);
        return u32(x + z*x_sz);
    }}
    
    // function to calculate laplacian
    fn laplacian(z: i32, x: i32, num_coef: i32, stencil: i32, z_sz: i32, x_sz: i32) -> f32 {{
        var idx_coef: i32 = 0;                  
        
        // central
        var lap: f32 = 2.0 * lv.coef[0] * PKm1[zx(z,x)];
        
        // upper limit
        if(z < stencil) {{
            idx_coef = z + 1;
        }} else {{
            idx_coef = num_coef;
        }}
        for (var i = 1; i < idx_coef; i = i + 1) {{
            lap = lap + lv.coef[i] * PKm1[zx(z+(-i),x)]; 
        }}
        
        // left limit
        if(x < stencil) {{
            idx_coef = x + 1;
        }} else {{
            idx_coef = num_coef;
        }}
        for (var i = 1; i < idx_coef; i = i + 1) {{
            lap = lap + lv.coef[i] * PKm1[zx(z,x+(-i))]; 
        }}
        
        // lower limit
        if(z >= z_sz - stencil) {{
            idx_coef = z_sz - z;
        }} else {{
            idx_coef = num_coef;
        }}
        for (var i = 1; i < idx_coef; i = i + 1) {{
            lap = lap + lv.coef[i] * PKm1[zx(z+i,x)]; 
        }}
        
        // right limit
        if(x >= x_sz - stencil) {{
            idx_coef = x_sz - x;
        }} else {{
            idx_coef = num_coef;
        }}
        for (var i = 1; i < idx_coef; i = i + 1) {{
            lap = lap + lv.coef[i] * PKm1[zx(z,x+i)]; 
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
        //let x: i32 = i32(threadIdx.y);           // x thread index
        let z_sz: i32 = i32(lv.z_sz);           // size in z direction
        let x_sz: i32 = i32(lv.x_sz);           // size in x direction
        let z_src: i32 = 0;//z_sz/2;                // source term z position
        let x_src: i32 = x_sz/2;                // source term x position
        let num_coef: i32 = i32(lv.num_coef);   // num coefs
        let stencil: i32 = num_coef - 1;        // number of side coefs
        var lap: f32 = 0.0;                     // laplacian
        //let k: u32 = u32(lv.k);
        //let k: u32 = u32(kk[0]);
        
        //for (var t = 0; t < nt; t = t + 1) {{
            // --------------------
            // Calculate laplacian
            lap = laplacian(z, x, num_coef, stencil, z_sz, x_sz);
            //storageBarrier();
            //workgroupBarrier();
            
            // --------------------
            // Update pressure field
            //add_src = f32(!bool(zx(z,x)-zx(z_src,x_src)));
            //PK[zx(z,x)] = -1.0*PKm2[zx(z,x)] + 2.0*PKm1[zx(z,x)] + c*c*lap + src[k]*add_src;
            PK[zx(z,x)] = -1.0*PKm2[zx(z,x)] + 2.0*PKm1[zx(z,x)] + c*c*lap;
            if(zx(z,x)==zx(z_src,x_src)) {{
                PK[zx(z,x)] = src[i32(kk[0])];
            }}
            //storageBarrier();
            //workgroupBarrier();
            
            // --------------------
            // Circular buffer
            PKm2[zx(z,x)] = PKm1[zx(z,x)];
            //storageBarrier();
            //workgroupBarrier();
            PKm1[zx(z,x)] = PK[zx(z,x)];
            //storageBarrier();
            //workgroupBarrier();
            //PK[zx(z,x)] = f32(k);
            //PK[zx(z,x)] = src[k];
            //storageBarrier();
            //workgroupBarrier();
        //}}
        //if(zx(z,x)==zx(z_src,x_src)) {{
        if(z == 0 && x == 0) {{
            kk[0] += 1.0;
        }}
        //storageBarrier();
        //workgroupBarrier();
    }}
    """



def sim_webgpu_for(coef):
    kk = np.array([0.0], dtype=dtype)
    # pressure field
    PKm2 = np.zeros((nz, nx), dtype=dtype)  # k-2
    PKm1 = np.zeros((nz, nx), dtype=dtype)  # k-1
    PK = np.zeros((nz, nx), dtype=dtype)  # k
    k = 0.0
    # source term
    src_gpu = src.astype(dtype)
    # auxiliar variables
    info = np.array([nz, nx, len(coef), c, k], dtype=dtype)
    info = np.concatenate((info, coef))
    # =====================
    # webgpu configurations
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader_for)
    # info buffer
    b0 = device.create_buffer_with_data(data=info, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # field pressure at time k-2
    b1 = device.create_buffer_with_data(data=PKm2, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # field pressure at time k-1
    b2 = device.create_buffer_with_data(data=PKm1, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # field pressure at time k
    b3 = device.create_buffer_with_data(data=PK, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_SRC)
    # source term
    b4 = device.create_buffer_with_data(data=src_gpu, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC |
                                                            wgpu.BufferUsage.STORAGE)
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
        {
            "binding": 4,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
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
        {
            "binding": 4,
            "resource": {"buffer": b4, "offset": 0, "size": b4.size},
        },
    ]

    #b5 = device.create_buffer_with_data(data=kk, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.MAP_READ)
    b5 = device.create_buffer(size=kk.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    binding_layouts2 = [
        {
            "binding": 5,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]
    bindings2 = [
        {
            "binding": 5,
            "resource": {"buffer": b5, "offset": 0, "size": b5.size},
        },
    ]

    # Put everything together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    bind_group_layout2 = device.create_bind_group_layout(entries=binding_layouts2)
    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout, bind_group_layout2])
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)
    bind_group2 = device.create_bind_group(layout=bind_group_layout2, entries=bindings2)

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    compute_pass.set_bind_group(1, bind_group2, [], 0, 999999)  # last 2 elements not used
    device.queue.write_buffer(b5, 0, kk)
    for i in range(nt):
        #kk = np.asarray(device.queue.read_buffer(b5).cast("f"))
        #kk[0] += 1.0
        #device.queue.write_buffer(b5, 0, kk)
        # compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
        #compute_pass.set_bind_group(1, bind_group2, [], 0, 999999)  # last 2 elements not used
        #info = np.array([nz, nx, len(coef), c, k], dtype=dtype)
        #info = np.concatenate((info, coef))
        compute_pass.dispatch_workgroups(nx)  # x y z -- num blocks
        #k += 1


    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    out = device.queue.read_buffer(b3).cast("f")  # reads from buffer 3
    return np.asarray(out).reshape((nz, nx))


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
# u_ser = sim_full()
t_ser = time() - t_ser

# print(f'MSE entre For e Ref: {MSE(u_for[:, :, -1], u_ser[:, :, -1])}')
print(f'TEMPO - {nt} pontos de tempo:\nFor: {t_for:.3}s\nSerial: {t_ser:.3}s')

plt.figure(1)
plt.imshow(u_for, aspect='auto', cmap='turbo_r')
plt.colorbar()
# plt.figure(2)
# plt.imshow(u_ser[:, :, -1], aspect='auto', cmap='turbo_r')
# plt.show()

