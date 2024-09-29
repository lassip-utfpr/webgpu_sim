struct SimIntValues {
    x_sz: i32,
    x_source: i32,
    n_iter: i32,
    n_rec: i32,
    it: i32,
    ord: i32
};

struct SimFltValues {
    dx: f32,
    dt: f32,
    rho: f32,
    lambda: f32,
    mu : f32,
    lambdaplus2mu: f32
};

// Parametros
@group(0) @binding(0)
var<storage,read> sim_flt: SimFltValues;

@group(0) @binding(1)
var<storage,read> force: array<f32>;

@group(0) @binding(2)
var<storage,read_write> sim_int: SimIntValues;

// Valores da simulação
@group(1) @binding(3)
var<storage,read_write> vx: array<f32>;

@group(1) @binding(4)
var<storage,read_write> sigma: array<f32>;

// Receptor
@group(2) @binding(5)
var<storage,read_write> sensors_vx: array<f32>;

@group(2) @binding(6)
var<storage,read> sensors_pos_x: array<i32>;

@group(0) @binding(15)
var<storage,read> coefs:array<f32>;

@group(3) @binding(7)
var<storage,read_write> mdvx_dx: array<f32>;

@group(3) @binding(8)
var<storage,read_write> m_dsigxx_dx: array<f32>;

@group(3) @binding(9)
var<storage,read> a_x: array<f32>;

@group(3) @binding(10)
var<storage,read> b_x: array<f32>;

@group(3) @binding(11)
var<storage,read> k_x: array<f32>;

@group(3) @binding(12)
var<storage,read> a_x_h: array<f32>;

@group(3) @binding(13)
var<storage,read> b_x_h: array<f32>;

@group(3) @binding(14)
var<storage,read> k_x_h: array<f32>;

@compute
@workgroup_size(wsx)
fn sigmax(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);
    let lambda: f32 = sim_flt.lambda;
    let mu: f32 = sim_flt.mu;
    let lambdaplus2mu: f32 = sim_flt.lambdaplus2mu;
    let id: i32 = sim_int.ord - 1;
    var id_x_f: i32 = sim_int.x_sz - sim_int.ord;

    // Cálculo das derivadas da velocidade e do valor de sigma
    if(x >= id && x < id_x_f) {

       var dvx_dx: f32 = 0.0;

        for(var c: i32 = 0; c < sim_int.ord; c++) {
            dvx_dx += coefs[c]*(vx[x+c+1]-vx[x-c])/sim_flt.dx;
        }

        let mdvx_dx_new: f32 = b_x_h[x - id] * mdvx_dx[x] + a_x_h[x - id] * dvx_dx;
        dvx_dx = dvx_dx/k_x_h[x - id] + mdvx_dx_new;

        mdvx_dx[x] = mdvx_dx_new;

        sigma[x] = sigma[x] + (lambdaplus2mu * dvx_dx) * sim_flt.dt;
    }
}


@compute
@workgroup_size(wsx)
fn velx(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);
    let dt_rho: f32 = sim_flt.dt / sim_flt.rho;
    let id: i32 = sim_int.ord - 1;
    var id_x_f: i32 = sim_int.x_sz - id;

    // Cálculo das derivadas do sigma e do valor da velocidade
    if(x >= sim_int.ord && x < id_x_f) {

        var dsig_dx: f32 = 0.0;

        for(var c: i32 = 0; c < sim_int.ord; c++) {
            dsig_dx += coefs[c]*(sigma[x+c] - sigma[x - c - 1])/sim_flt.dx;
        }

        let mdssx_dx_new: f32 = b_x[x-id]*m_dsigxx_dx[x]+a_x[x-id]*dsig_dx;
        dsig_dx = dsig_dx/k_x[x-id] + mdssx_dx_new;
        m_dsigxx_dx[x]=mdssx_dx_new;

        vx[x] =  dt_rho * (dsig_dx) + vx[x];

    }
}


@compute
@workgroup_size(wsx)
fn sensx(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);
    let dt_over_rho: f32 = sim_flt.dt / sim_flt.rho;
    let x_source: i32 = sim_int.x_source;
    let it: i32 = sim_int.it;
    let id: i32 = sim_int.ord - 1;
    let id_x_f: i32 = sim_int.x_sz - sim_int.ord;

    // Fonte
    if(x == x_source) {
        vx[x] = vx[x] + force[it] * dt_over_rho;
    }

    // Condições de Dirichlet
    if(x <= id || x >= id_x_f) {
        vx[x]=0.0;

    }

    // Velocidade receptores
    for(var s: i32 = 0; s < sim_int.n_rec; s++) {
        if(x == sensors_pos_x[s]) {
            sensors_vx[it]=vx[x];

            break;
        }
    }
}


@compute
@workgroup_size(1)
fn incr_it() {
    sim_int.it += 1;
}
