
@group(0) @binding(0)
var<storage,read_write> data0 : array<i32>;

@group(1) @binding(1)
var<storage,read_write> data1 : array<i32>;

@compute
@workgroup_size(1)
fn incr_h(){

data1[0]=1;

}
