
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

@compute
@workgroup_size(16384,16384,16384) // o MAX é 16384
fn incr_k() {
    data0[0] = 1;
}

// Caso seja adicionado um oitavo grupo, será impossível computar esse arquivo
// @group(8) @binding(0)
//var<storage,read_write> data8 : array<i32>;
