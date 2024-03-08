
@group(0) @binding(0)
var<storage,read_write> data0 : array<i32>;

@group(1) @binding(0)
var<storage,read_write> data1 : array<i32>;

@group(2) @binding(0)
var<storage,read_write> data2 : array<i32>;

@group(3) @binding(0)
var<storage,read_write> data3 : array<i32>;

@group(4) @binding(0)
var<storage,read_write> data4 : array<i32>;

@group(5) @binding(0)
var<storage,read_write> data5 : array<i32>;

@group(6) @binding(0)
var<storage,read_write> data6 : array<i32>;

@group(7) @binding(0)
var<storage,read_write> data7 : array<i32>;


//var i: i32 = 0;
//loop{
//if i>=1000 {break;}
//@group(7) @binding(i)
//    var<storage,read_write> i : array<i32>;
//i++
//}

// NAO ESQUECER DE CONITNUAR DAQUI
//  In wgpuDeviceCreateComputePipeline
   //    Error matching shader requirements against the pipeline
   //    Shader entry point's workgroup size [1024, 1024, 64] (67108864 total invocations) must be less or equal to the per-dimension limit [1024, 1024, 64] and the total invocation limit 1024

@compute
@workgroup_size(1024, 1024, 64) // o MAX é 16384
fn big_t() {
    data0[0] = 1;
    data0[1] = 2;
    data0[2] = 3;
    data0[3] = 4;
    data0[4] = 5;
    data0[5] = 6;
    data0[6] = 7;
    data0[7] = 8;
}

// Caso seja adicionado um oitavo grupo, será impossível computar esse arquivo
// @group(8) @binding(0)
//var<storage,read_write> data8 : array<i32>;
