"""
Example compute shader that does ... nothing but copy a value from one
buffer into another.
"""

import wgpu
if wgpu.version_info[1] > 11:
    import wgpu.backends.wgpu_native  # Select backend 0.13.X
else:
    import wgpu.backends.rs  # Select backend 0.9.5

import numpy as np

# Create input data as a memoryview
n = 20
j = 5
out = np.zeros((j, n), dtype=np.int32)
data = memoryview(bytearray(n * 4)).cast("i")
for i in range(n):
    data[i] = i

# %% Shader and data
shader_source = f"""
    @group(0) @binding(0)
    var<storage,read> data1: array<i32>;
    
    @group(0) @binding(1)
    var<storage,read_write> data2: array<i32>;
    
    @group(0) @binding(2)
    var<storage,read_write> k: array<i32>;
    
    @group(0) @binding(3)
    var<storage,read_write> out: array<i32>;

    // function to convert 2D [j,n] index into 1D index
    fn idx(j: i32, n: i32) -> i32 {{
        let j_sz: i32 = {j};
        let n_sz: i32 = {n};
        let index = i32(n + j*n_sz);
        
        return select(-1, index, j >= 0 && j < j_sz && n >= 0 && n < n_sz);
    }}
 
    @compute
    @workgroup_size(1)
    fn main(@builtin(global_invocation_id) index: vec3<u32>) {{
        let i: i32 = i32(index.y);
        let j: i32 = k[0];
        let ix: i32 = idx(j, i);
        
        data2[i] = data1[i] + data2[i] + k[0];
        //data2[i] = data1[i] + data2[i] * k[0];
        out[ix] = data2[i];
    }}
"""

# %% The long version using the wgpu API

# Create device and shader object
device = wgpu.utils.get_default_device()
# adapter = wgpu.request_adapter(canvas=None, power_preference="low-power")
# device = adapter.request_device()
cshader = device.create_shader_module(code=shader_source)

# Create buffer objects, input buffer is mapped.
buffer1 = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
buffer2 = device.create_buffer(size=data.nbytes, usage=wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_SRC |
                                                       wgpu.BufferUsage.COPY_DST)
buffer3 = device.create_buffer(size=4, usage=wgpu.BufferUsage.STORAGE |
                                             wgpu.BufferUsage.COPY_DST |
                                             wgpu.BufferUsage.COPY_SRC)
buffer4 = device.create_buffer_with_data(data=out, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

# Setup layout and bindings
bl_0 = [
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
bind_0 = [
    {
        "binding": 0,
        "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
    },
    {
        "binding": 2,
        "resource": {"buffer": buffer3, "offset": 0, "size": buffer3.size},
    },
    {
        "binding": 3,
        "resource": {"buffer": buffer4, "offset": 0, "size": buffer4.size},
    },
]

# Put everything together
bgl_0 = device.create_bind_group_layout(entries=bl_0)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_0])
bg_0 = device.create_bind_group(layout=bgl_0, entries=bind_0)

# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

for i in range(j):
    command_encoder = device.create_command_encoder()
    device.queue.write_buffer(buffer3, 0, i.to_bytes(4, "little"))
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_bind_group(0, bg_0, [], 0, 999999)  # last 2 elements not used
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.dispatch_workgroups(j, n)  # x y z
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

# Read result
result = device.queue.read_buffer(buffer2).cast("i").tolist()
k = device.queue.read_buffer(buffer3).cast("i").tolist()
out = np.asarray(device.queue.read_buffer(buffer4).cast("i")).reshape((j, n))
print(result)
print(out)
