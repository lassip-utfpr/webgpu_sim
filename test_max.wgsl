
@group(0) @binding(0)
var<storage,read_write> data0 : array<i32>;

@group(1) @binding(1)
var<storage,read_write> data1 : array<i32>;

@group(2) @binding(2)
var<storage,read_write> data2 : array<i32>;

@group(3) @binding(3)
var<storage,read_write> data3 : array<i32>;

@group(4) @binding(4)
var<storage,read_write> data4 : array<i32>;

@group(5) @binding(5)
var<storage,read_write> data5 : array<i32>;

@group(6) @binding(6)
var<storage,read_write> data6 : array<i32>;

@group(7) @binding(7)
var<storage,read_write> data7 : array<i32>;

@compute
@workgroup_size(16,1,64)
fn incr_h(){

data7[0]=1;
}

@compute
@workgroup_size(16384) // o MAX é 16384 quando programado em wgsl, entretanto quando utilizado como shader no py seu limite é 1024,1024,64 MAS o numero total no workgroup tem que ser menor que 1024
fn big_t() {
//    data3[0]=1;
//    data3[3] = 4;
//    data3[4] = 5;
//    data3[5] = 6;
//    data3[6] = 7;
//    data3[999999999] = 8;
//    data4[0] = 1;
//    data4[1] = 2;
//    data4[2] = 3;
//    data4[3] = 4;
//    data4[4] = 5;
//    data4[5] = 6;
//    data4[6] = 7;
//    data4[999999999] = 8;
//    data5[0] = 1;
//    data5[1] = 2;
//    data5[2] = 3;
//    data5[3] = 4;
//    data5[4] = 5;
//    data5[5] = 6;
//    data5[6] = 7;
//    data5[999999999] = 8;
//    data6[0] = 1;
//    data6[1] = 2;
//    data6[2] = 3;
//    data6[3] = 4;
//    data6[4] = 5;
//    data6[5] = 6;
//    data6[6] = 7;
//    data6[999999999] = 8;
    data7[0] = 1;
//    data7[1] = 2;
//    data7[2] = 3;
//    data7[3] = 4;
//    data7[4] = 5;
//    data7[5] = 6;
//    data7[6] = 7;
//    data7[999999999] = 8;

}

// Caso seja adicionado um oitavo grupo, será impossível computar esse arquivo
// @group(8) @binding(0)
//var<storage,read_write> data8 : array<i32>;
