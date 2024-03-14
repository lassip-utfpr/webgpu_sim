import numpy as np
import wgpu


data = np.array([1,1,1,1])
datatest= np.array([1])
t_out = np.zeros(shape=data.shape, dtype= np.int32)
device = wgpu.utils.get_default_device()

with open('test_max.wgsl') as shader_file:
    cshader_str = shader_file.read()
    cshader = device.create_shader_module(code=cshader_str)

bf00 = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE |
                                                       wgpu.BufferUsage.COPY_DST |
                                                       wgpu.BufferUsage.COPY_SRC)

bf01 = device.create_buffer_with_data(data= datatest, usage= wgpu.BufferUsage.STORAGE |
                                                            wgpu.BufferUsage.COPY_DST |
                                                            wgpu.BufferUsage.COPY_SRC)

bg_layout = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

bg1_layout = [
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer":{
            "type": wgpu.BufferBindingType.storage,
        },
    },
]

bg1_config = [
    {
        "binding": 1,
        "resource":{"buffer": bf01, "offset":0, "size":bf01.size},
    },

]

bg_config = [
    {
        "binding": 0,
        "resource":{"buffer":bf00, "offset":0, "size":bf00.size},
    },
]

bgl= device.create_bind_group_layout(entries=bg_layout)
bgl1= device.create_bind_group_layout(entries=bg1_layout)

pipeline= device.create_pipeline_layout(bind_group_layouts=[bgl,bgl1])

bgf= device.create_bind_group(layout=bgl, entries=bg_config)
bgf1= device.create_bind_group(layout= bgl1, entries= bg1_config)

compute_incr_h = device.create_compute_pipeline(
    layout = pipeline,
    compute = {"module": cshader, "entry_point": "incr_h"},
)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()

compute_pass.set_bind_group(0,bgf,[],0,999999)
compute_pass.set_bind_group(1,bgf1,[],0,999999)

compute_pass.set_pipeline(compute_incr_h)
compute_pass.dispatch_workgroups(1)
compute_pass.end()
device.queue.submit([command_encoder.finish()])
t_out= device.queue.read_buffer(bf00).cast("i").tolist()
print(t_out)