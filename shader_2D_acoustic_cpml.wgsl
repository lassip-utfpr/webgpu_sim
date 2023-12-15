    struct SimIntValues {
        x_sz: i32,          // x field size
        y_sz: i32,          // y field size
        x_sens: i32,        // x sensor
        y_sens: i32,        // y sensor
        k: i32              // iteraction
    };

    struct SimFltValues {
        cp_unrelaxed: f32,  // sound speed
        dx: f32,            // delta x
        dy: f32,            // delta y
        dt: f32             // delta t
    };

    // Group 0 - parameters
    @group(0) @binding(0)   // param_flt32
    var<storage,read> sim_flt_par: SimFltValues;

    @group(0) @binding(1) // source term
    var<storage,read> src: array<f32>;

    @group(0) @binding(2) // kronecker_src, rho_half_x, rho_half_y, kappa
    var<storage,read> img_params: array<f32>;

    @group(0) @binding(3) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
    var<storage,read> coef_x: array<f32>;

    @group(0) @binding(4) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
    var<storage,read> coef_y: array<f32>;

    @group(0) @binding(5) // param_int32
    var<storage,read_write> sim_int_par: SimIntValues;

    // Group 1 - simulation arrays
    @group(1) @binding(6) // pressure future (p_0)
    var<storage,read_write> pr_future: array<f32>;

    @group(1) @binding(7) // pressure fields p_1, p_2
    var<storage,read_write> pr_fields: array<f32>;

    @group(1) @binding(8) // derivative fields x (v_x, mdp_x, dp_x, dmdp_x)
    var<storage,read_write> der_x: array<f32>;

    @group(1) @binding(9) // derivative fields y (v_y, mdp_y, dp_y, dmdp_y)
    var<storage,read_write> der_y: array<f32>;

    // Group 2 - sensors arrays
    @group(2) @binding(10) // sensor signal
    var<storage,read_write> sensor: array<f32>;

    // function to convert 2D [x,y] index into 1D [xy] index
    fn xy(x: i32, y: i32) -> i32 {
        let index = y + x * sim_int_par.y_sz;

        return select(-1, index, x >= 0 && x < sim_int_par.x_sz && y >= 0 && y < sim_int_par.y_sz);
    }

    // function to convert 2D [i,j] index into 1D [] index
    fn ij(i: i32, j: i32, i_max: i32, j_max: i32) -> i32 {
        let index = j + i * j_max;

        return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max);
    }

    // function to convert 3D [i,j,k] index into 1D [] index
    fn ijk(i: i32, j: i32, k: i32, i_max: i32, j_max: i32, k_max: i32) -> i32 {
        let index = j + i * j_max + k * j_max * i_max;

        return select(-1, index, i >= 0 && i < i_max && j >= 0 && j < j_max && k >= 0 && k < k_max);
    }

    // function to get a kappa array value
    fn get_kappa(i: i32, j: i32) -> f32 {
        let index: i32 = ijk(i, j, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }

    // function to get a kronecker_src array value
    fn get_kronecker_src(i: i32, j: i32) -> f32 {
        let index: i32 = ijk(i, j, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }

    // function to get a rho_h_x array value
    fn get_rho_h_x(i: i32, j: i32) -> f32 {
        let index: i32 = ijk(i, j, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }

    // function to get a rho_h_y array value
    fn get_rho_h_y(i: i32, j: i32) -> f32 {
        let index: i32 = ijk(i, j, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, img_params[index], index != -1);
    }

    // function to get a src array value
    fn get_src(n: i32) -> f32 {
        return select(0.0, src[n], n >= 0);
    }

    // function to get a a_x array value
    fn get_a_x(n: i32) -> f32 {
        let index: i32 = ij(n, 0, sim_int_par.x_sz, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a b_x array value
    fn get_b_x(n: i32) -> f32 {
        let index: i32 = ij(n, 1, sim_int_par.x_sz, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a k_x array value
    fn get_k_x(n: i32) -> f32 {
        let index: i32 = ij(n, 2, sim_int_par.x_sz, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a a_y array value
    fn get_a_y(n: i32) -> f32 {
        let index: i32 = ij(0, n, 6, sim_int_par.y_sz);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a b_y array value
    fn get_b_y(n: i32) -> f32 {
        let index: i32 = ij(1, n, 6, sim_int_par.y_sz);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a k_y array value
    fn get_k_y(n: i32) -> f32 {
        let index: i32 = ij(2, n, 6, sim_int_par.y_sz);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a a_x_h array value
    fn get_a_x_h(n: i32) -> f32 {
        let index: i32 = ij(n, 3, sim_int_par.x_sz, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a b_x_h array value
    fn get_b_x_h(n: i32) -> f32 {
        let index: i32 = ij(n, 4, sim_int_par.x_sz, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a k_x_h array value
    fn get_k_x_h(n: i32) -> f32 {
        let index: i32 = ij(n, 5, sim_int_par.x_sz, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a a_y_h array value
    fn get_a_y_h(n: i32) -> f32 {
        let index: i32 = ij(3, n, 6, sim_int_par.y_sz);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a b_y_h array value
    fn get_b_y_h(n: i32) -> f32 {
        let index: i32 = ij(4, n, 6, sim_int_par.y_sz);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a k_y_h array value
    fn get_k_y_h(n: i32) -> f32 {
        let index: i32 = ij(5, n, 6, sim_int_par.y_sz);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get an p_0 (pr_future) array value
    fn get_p_0(x: i32, y: i32) -> f32 {
        let index: i32 = xy(x, y);

        return select(0.0, pr_future[index], index != -1);
    }

    // function to set a p_0 array value
    fn set_p_0(x: i32, y: i32, val : f32) {
        let index: i32 = xy(x, y);

        if(index != -1) {
            pr_future[index] = val;
        }
    }

    // function to get an p_1 array value
    fn get_p_1(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        return select(0.0, pr_fields[index], index != -1);
    }

    // function to set a p_1 array value
    fn set_p_1(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        if(index != -1) {
            pr_fields[index] = val;
        }
    }

    // function to get an p_2 array value
    fn get_p_2(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        return select(0.0, pr_fields[index], index != -1);
    }

    // function to set a p_2 array value
    fn set_p_2(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        if(index != -1) {
            pr_fields[index] = val;
        }
    }

    // function to get an v_x array value
    fn get_v_x(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }

    // function to set a v_x array value
    fn set_v_x(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_x[index] = val;
        }
    }

    // function to get an v_y array value
    fn get_v_y(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }

    // function to set a v_y array value
    fn set_v_y(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_y[index] = val;
        }
    }

    // function to get an mdp_x array value
    fn get_mdp_x(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }

    // function to set a mdp_x array value
    fn set_mdp_x(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_x[index] = val;
        }
    }

    // function to get an mdp_y array value
    fn get_mdp_y(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }

    // function to set a mdp_y array value
    fn set_mdp_y(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_y[index] = val;
        }
    }

    // function to get an dp_x array value
    fn get_dp_x(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }

    // function to set a dp_x array value
    fn set_dp_x(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_x[index] = val;
        }
    }

    // function to get an dp_y array value
    fn get_dp_y(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }

    // function to set a dp_y array value
    fn set_dp_y(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_y[index] = val;
        }
    }

    // function to get an dmdp_x array value
    fn get_dmdp_x(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_x[index], index != -1);
    }

    // function to set a dmdp_x array value
    fn set_dmdp_x(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_x[index] = val;
        }
    }

    // function to get an dmdp_y array value
    fn get_dmdp_y(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, der_y[index], index != -1);
    }

    // function to set a dmdp_y array value
    fn set_dmdp_y(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            der_y[index] = val;
        }
    }

    // function to calculate first derivatives
    @compute
    @workgroup_size(wsx, wsy)
    fn space_sim1(@builtin(global_invocation_id) index: vec3<u32>) {
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        var vdp_x: f32 = 0.0;
        var vdp_y: f32 = 0.0;

        // Calcula a primeira derivada espacial dividida pela densidade
        vdp_x = (get_p_1(x + 1, y) - get_p_1(x, y)) / sim_flt_par.dx;
        set_mdp_x(x, y, get_b_x_h(x)*get_mdp_x(x, y) + get_a_x_h(x)*vdp_x);
        vdp_y = (get_p_1(x, y + 1) - get_p_1(x, y)) / sim_flt_par.dy;
        set_mdp_y(x, y, get_b_y_h(y)*get_mdp_y(x, y) + get_a_y_h(y)*vdp_y);
        set_dp_x(x, y, (vdp_x / get_k_x_h(x) + get_mdp_x(x, y))/get_rho_h_x(x, y));
        set_dp_y(x, y, (vdp_y / get_k_y_h(y) + get_mdp_y(x, y))/get_rho_h_y(x, y));
    }

    // function to calculate second derivatives
    @compute
    @workgroup_size(wsx, wsy)
    fn space_sim2(@builtin(global_invocation_id) index: vec3<u32>) {
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        var vdp_xx: f32 = 0.0;
        var vdp_yy: f32 = 0.0;

        // Calcula a segunda derivada espacial
        vdp_xx = (get_dp_x(x, y) - get_dp_x(x - 1, y)) / sim_flt_par.dx;
        set_dmdp_x(x, y, get_b_x(x)*get_dmdp_x(x, y) + get_a_x(x)*vdp_xx);
        vdp_yy = (get_dp_y(x, y) - get_dp_y(x, y - 1)) / sim_flt_par.dy;
        set_dmdp_y(x, y, get_b_y(y)*get_dmdp_y(x, y) + get_a_y(y)*vdp_yy);
        set_v_x(x, y, vdp_xx / get_k_x(x) + get_dmdp_x(x, y));
        set_v_y(x, y, vdp_yy / get_k_y(y) + get_dmdp_y(x, y));
    }

    @compute
    @workgroup_size(1)
    fn incr_k() {
        sim_int_par.k += 1;
    }

    @compute
    @workgroup_size(wsx, wsy)
    fn time_sim(@builtin(global_invocation_id) index: vec3<u32>) {
        var add_src: f32 = 0.0;             // Source term
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dt: f32 = sim_flt_par.dt;
        let pi_4: f32 = 12.5663706144;

        // --------------------
        // Update pressure field
        add_src = pi_4*sim_flt_par.cp_unrelaxed*sim_flt_par.cp_unrelaxed*src[sim_int_par.k]*get_kronecker_src(x, y);
        set_p_0(x, y, -1.0*get_p_2(x, y) + 2.0*get_p_1(x, y) +
            dt*dt*((get_v_x(x, y) + get_v_y(x, y))*get_kappa(x, y) + add_src));

        // Aplly Dirichlet conditions
        if(x == 0 || x == (sim_int_par.x_sz - 1) || y == 0 || y == (sim_int_par.y_sz - 1)) {
            set_p_0(x, y, 0.0);
        }

        // --------------------
        // Circular buffer
        set_p_2(x, y, get_p_1(x, y));
        set_p_1(x, y, get_p_0(x, y));

        if(x == sim_int_par.x_sens && y == sim_int_par.y_sens) {
            sensor[sim_int_par.k] = get_p_0(x, y);
        }
    }
