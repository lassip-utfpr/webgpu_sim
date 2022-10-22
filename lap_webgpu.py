import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from time import time

# ==========================================================
# Esse arquivo contém as versões do laplaciano implementadas em Webgpu.
# ==========================================================

dtype = np.float32

# Webgpu laplacian with padding
def lap_pad(img, coef, it):
    # padding
    pad = len(coef) - 1
    # sizes
    z_sz, x_sz = img.shape[0], img.shape[1]
    Z_sz, X_sz = z_sz + pad * 2, x_sz + pad * 2
    # laplacian field
    u = np.zeros((Z_sz, X_sz), dtype=dtype)
    u[pad:-pad, pad:-pad] = img.astype(dtype)
    # other infos
    info = np.array([Z_sz, X_sz, pad, len(coef), it], dtype=dtype)
    info = np.concatenate((info, coef))  # struct LapValues
    # =====================
    # WEBGPU CONFIGURATIONS
    shader = f"""
    struct LapValues {{
        z_sz: f32,          // Z field size
        x_sz: f32,          // X field size
        pad: f32,           // padding
        num_coef: f32,      // num of discrete coefs
        num_iter: f32,      // num of iterations
        coef: array<f32>,   // discrete coefs
    }};

    @group(0) @binding(0)   // info buffer
    var<storage,read> lv: LapValues;

    @group(0) @binding(1)
    var<storage,read_write> u: array<f32>;

    @group(0) @binding(2)
    var<storage,read_write> v: array<f32>;

    // function to convert 2D [z,x] index into 1D [zx] index
    fn zx(z: i32, x: i32) -> u32 {{
        let x_sz: i32 = i32(lv.x_sz);
        return u32(x + z*x_sz);
    }}

    @stage(compute)
    @workgroup_size({z_sz}) // z --> num threads per block -- hardcoded
    fn main(@builtin(local_invocation_id) threadIdx: vec3<u32>,
            @builtin(workgroup_id) blockIdx: vec3<u32>) {{

        let pad: i32 = i32(lv.pad);             // padding
        let num_coef: i32 = i32(lv.num_coef);   // num coefs
        let num_iter: i32 = i32(lv.num_iter);   // num iterations
        let z: i32 = i32(threadIdx.x) + pad;    // z thread index
        let x: i32 = i32(blockIdx.x) + pad;     // x thread index
        var lap: f32 = 0.0;                     // lap accumulative variable
        var buffer: u32 = 1u;                   // buffer flag

        for (var t = 0; t < num_iter; t = t + 1) {{
            // reads from buffer 'u' and writes into buffer 'v'
            if (buffer == 1u) {{
                lap = 2.0 * lv.coef[0] * u[zx(z,x)];
                for (var k = 1; k < num_coef; k = k + 1)
                {{
                    lap = lap + lv.coef[k] * (u[zx(z+(-k),x)] + u[zx(z+k,x)] + u[zx(z,x+(-k))] + u[zx(z,x+k)]);
                }}
                // flag to change buffer writing
                buffer = 0u;
                v[zx(z,x)] = lap;
                storageBarrier();
            }} 
            // reads from buffer 'v' and writes into buffer 'u'
            else {{
                lap = 2.0 * lv.coef[0] * v[zx(z,x)];
                for (var k = 1; k < num_coef; k = k + 1)
                {{
                    lap = lap + lv.coef[k] * (v[zx(z+(-k),x)] + v[zx(z+k,x)] + v[zx(z,x+(-k))] + v[zx(z,x+k)]);
                }}
                // flag to change buffer writing
                buffer = 1u;
                u[zx(z,x)] = lap;
                storageBarrier();
            }}
        }}

        // 'v' buffer is the cpu output
        if (buffer == 1u) {{
            v[zx(z,x)] = u[zx(z,x)];
            storageBarrier();
        }}
    }}
    """

    # Create device and shader object
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader)

    # Create buffers
    # Info buffer
    b0 = device.create_buffer_with_data(data=info, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # Lap buffer 1 (u)
    b1 = device.create_buffer_with_data(data=u.ravel(), usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # Lap buffer 2 (v)
    b2 = device.create_buffer(size=u.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
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
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    compute_pass.dispatch_workgroups(x_sz, 1, 1)  # x y z -- num blocks
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    out = device.queue.read_buffer(b2).cast("f")  # v is the output buffer / cast f32
    return np.asarray(out).reshape((Z_sz, X_sz))[pad:-pad, pad:-pad]


# Webgpu laplacian with for
def lap_for(img, coef, it):
    # sizes
    z_sz, x_sz = img.shape[0], img.shape[1]
    # laplacian field
    u = img.astype(dtype)
    # other infos
    info = np.array([z_sz, x_sz, len(coef), it], dtype=dtype)
    info = np.concatenate((info, coef))  # struct LapValues
    # =====================
    # WEBGPU CONFIGURATIONS
    shader = f"""
    struct LapValues {{
        z_sz: f32,          // X field size
        x_sz: f32,          // Z field size
        num_coef: f32,      // num of discrete coefs
        num_iter: f32,      // num of iterations
        coef: array<f32>,   // discrete coefs
    }};

    @group(0) @binding(0)   // info buffer
    var<storage,read> lv: LapValues;

    @group(0) @binding(1)
    var<storage,read_write> u: array<f32>;

    @group(0) @binding(2)
    var<storage,read_write> v: array<f32>;

    // function to convert 2D [z,x] index into 1D [zx] index
    fn zx(z: i32, x: i32) -> u32 {{
        let x_sz: i32 = i32(lv.x_sz);
        return u32(x + z*x_sz);
    }}

    @stage(compute)
    @workgroup_size({z_sz}) // z --> num threads per block -- hardcoded
    fn main(@builtin(local_invocation_id) threadIdx: vec3<u32>,
            @builtin(workgroup_id) blockIdx: vec3<u32>) {{

        let num_coef: i32 = i32(lv.num_coef);       // num coefs
        let stencil: i32 = num_coef - 1;            // number of side coefs
        let num_iter: i32 = i32(lv.num_iter);       // num iterations
        let z_sz: i32 = i32(lv.z_sz);               // size in z direction
        let x_sz: i32 = i32(lv.x_sz);               // size in x direction
        let z: i32 = i32(threadIdx.x);              // z thread index
        let x: i32 = i32(blockIdx.x);               // x thread index
        var lap: f32 = 0.0;                         // lap accumulative variable
        var buffer: u32 = 1u;                       // buffer flag
        var idx_coef: i32 = 0;                      // index of side coefficients

        for (var t = 0; t < num_iter; t = t + 1) {{
            // reads from buffer 'u' and writes into buffer 'v'
            if (buffer == 1u) {{
                lap = 0.0;

                // upper limit
                if(z < stencil) {{
                    idx_coef = z + 1;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * u[zx(z+(-i), x)]; 
                }}

                // left limit
                if(x < stencil) {{
                    idx_coef = x + 1;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * u[zx(z, x+(-i))]; 
                }}

                // lower limit
                if(z >= z_sz - stencil) {{
                    idx_coef = z_sz - z;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * u[zx(z+i, x)]; 
                }}

                // right limit
                if(x >= x_sz - stencil) {{
                    idx_coef = x_sz - x;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * u[zx(z, x+i)]; 
                }}

                // central
                lap = lap + 2.0 * lv.coef[0] * u[zx(z, x)];

                v[zx(z,x)] = lap;
                storageBarrier();
                // flag to change buffer writing
                buffer = 0u;
            }} 
            // reads from buffer 'v' and writes into buffer 'u'
            else {{
                lap = 0.0;

                // upper limit
                if(z < stencil) {{
                    idx_coef = z + 1;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * v[zx(z+(-i), x)]; 
                }}

                // left limit
                if(x < stencil) {{
                    idx_coef = x + 1;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * v[zx(z, x+(-i))]; 
                }}

                // lower limit
                if(z >= z_sz - stencil) {{
                    idx_coef = z_sz - z;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * v[zx(z+i, x)]; 
                }}

                // right limit
                if(x >= x_sz - stencil) {{
                    idx_coef = x_sz - x;
                }} else {{
                    idx_coef = num_coef;
                }}
                for (var i = 1; i < idx_coef; i = i + 1) {{
                    lap = lap + lv.coef[i] * v[zx(z, x+i)]; 
                }}

                // central
                lap = lap + 2.0 * lv.coef[0] * v[zx(z, x)];

                u[zx(z,x)] = lap;
                storageBarrier();
                // flag to change buffer writing
                buffer = 1u;
            }}
        }}

        // 'v' buffer is the cpu output
        if (buffer == 1u) {{
            v[zx(z,x)] = u[zx(z,x)];
            storageBarrier();
        }}

    }}
    """

    # Create device and shader object
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader)

    # Create buffers
    # Info buffer - copy from cpu and read only
    b0 = device.create_buffer_with_data(data=info, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # Lap buffer 1 (u) - copy from cpu->gpu, read and write
    b1 = device.create_buffer_with_data(data=u.ravel(), usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE)
    # Lap buffer 2 (v) - copy from cpu->gpu, read and write, copy from gpu->cpu (output)
    b2 = device.create_buffer(size=u.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
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
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
    compute_pass.dispatch_workgroups(x_sz, 1, 1)  # x y z -- num blocks
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    out = device.queue.read_buffer(b2).cast("f")  # v is the output buffer / cast f32
    return np.asarray(out).reshape((z_sz, x_sz))


def lap_pipa(u, c):
    v = (2.0 * c[0] * u).astype(dtype)
    for k in range(1, len(c)):
        v[k:, :] += c[k] * u[:-k, :]    # v[z + k, x] = v[z + k, x] + c[k] * u[z, x]  -- k acima
        v[:-k, :] += c[k] * u[k:, :]    # v[z - k, x] = v[z - k, x] + c[k] * u[z, x]  -- k abaixo
        v[:, k:] += c[k] * u[:, :-k]    # v[z, x + k] = v[z, x + k] + c[k] * u[z, x]  -- k a esquerda
        v[:, :-k] += c[k] * u[:, k:]    # v[z, x - k] = v[z, x - k] + c[k] * u[z, x]  -- k a direita
    return v


# Caso queira testar os laplacianos nesse arquivo, descomentar as linhas de código abaixo.
# Entretanto, vale ressaltar que essas funções são chamadas em outros arquivos.
# Logo, no momento que for usar esses outros arquivos, deixar as linhas abaixo comentadas.

# # --------------------------
# # COEFFICIENTS
# # finite difference coefficient
# c2 = np.array([-2, 1], dtype=dtype)
# c4 = np.array([-5 / 2, 4 / 3, -1 / 12], dtype=dtype)
# c6 = np.array([-49 / 18, 3 / 2, -3 / 20, 1 / 90], dtype=dtype)
# c8 = np.array([-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560], dtype=dtype)
# 
# # --------------------------
# # IMAGE FOR LAPLACIAN
# N = 50
# # random image
# image = np.random.rand(N, N).astype(dtype)
#
# # --------------------------
# # TESTING
# im1 = image.copy()
# im2 = image.copy()
# im3 = image.copy()
# im4 = image.copy()
#
# # num de iterations
# it = 4
#
# t1 = time()

# # Não executar o teste 'inside' e 'outside' simultaneamente

# # For outside gpu
# for i in range(it):
#     v1 = lap_pipa(im1, c8)
#     im1 = v1
    # -------------
    # uncomment for outside gpu testing
    # v2 = lap_pad(im2, c8, 1)
    # v3 = lap_for(im3, c8, 1)
    # im1, im2, im3 = v1, v2, v3
    # print(f'MSE It. [{i}]: {MSE(v1, v3)}')
# t1 = time() - t1
#
# # For inside GPU [a versão serial v1 deve ser executada ainda dentro do for anterior]
# t2 = time()
# v2 = lap_pad(im2, c8, it)
# t2 = time() - t2
# t3 = time()
# v3 = lap_for(image, c8, it)
# t3 = time() - t3
# print(f'MSE It. [{i}]: {MSE(v1, v2)}')
# print(f'\nTEMPO - {it} loops:\nRef: {t1:.3}s\nPadding: {t2:.3}s\nFor: {t3:.3}s')
