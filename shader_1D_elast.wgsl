struct SimIntValues {
    x_sz: i32,
    x_source: i32,
    n_iter: i32,
    n_rec: i32,
    it: i32
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

@compute
@workgroup_size(wsx)
fn sigmax(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);
    let lambda: f32 = sim_flt.lambda;
    let mu: f32 = sim_flt.mu;
    let lambdaplus2mu: f32 = sim_flt.lambda + 2.0 * sim_flt.mu;

    // Cálculo das derivadas da velocidade e do valor de sigma
    if(x >= 1 && x < (sim_int.x_sz - 2)) {
        var vdvx_dx: f32 = (27*(vx[x + 1] - vx[x]) - vx[x + 2] + vx[x - 1])/(24 * sim_flt.dx);

        sigma[x] = sigma[x] + (lambdaplus2mu * vdvx_dx) * sim_flt.dt;
    }
}


@compute
@workgroup_size(wsx)
fn velx(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);
    let dt_rho: f32 = sim_flt.dt / sim_flt.rho;

    // Cálculo das derivadas do sigma e do valor da velocidade
    if(x >= 2 && x < (sim_int.x_sz - 1)) {
        var vdsigmaxx_dx: f32 = (27.0*(sigma[x] - sigma[x - 1]) - sigma[x + 1] + sigma[x - 2])/(24.0 * sim_flt.dx);

        vx[x] =  dt_rho * (vdsigmaxx_dx) + vx[x];
    }
}


@compute
@workgroup_size(wsx)
fn sensx(@builtin(global_invocation_id) index: vec3<u32>) {
    let x: i32 = i32(index.x);          // x thread index
    let y: i32 = i32(index.y);          // y thread index
    let dt_over_rho: f32 = sim_flt.dt / sim_flt.rho;
    let x_source: i32 = sim_int.x_source;
    let it: i32 = sim_int.it;

    // Fonte
    if(x == x_source) {
        vx[x] = vx[x] + force[it] * dt_over_rho;
    }

    // Condições de Dirichlet
    if(x <= 1 || x >= (sim_int.x_sz - 2)) {
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
