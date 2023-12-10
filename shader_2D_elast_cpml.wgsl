    struct SimIntValues {
        x_sz: i32,          // x field size
        y_sz: i32,          // y field size
        x_source: i32,      // x source index
        y_source: i32,      // y source index
        x_sens: i32,        // x sensor
        y_sens: i32,        // y sensor
        np_pml: i32,        // PML size
        n_iter: i32,        // max iterations
        it: i32             // time iteraction
    };

    struct SimFltValues {
        cp: f32,            // longitudinal sound speed
        cs: f32,            // transverse sound speed
        dx: f32,            // delta x
        dy: f32,            // delta y
        dt: f32,            // delta t
        rho: f32,           // density
        lambda: f32,        // Lame parameter
        mu : f32,           // Lame parameter
        lambdaplus2mu: f32  // Lame parameter
    };

    // Group 0 - parameters
    @group(0) @binding(0)   // param_flt32
    var<storage,read> sim_flt_par: SimFltValues;

    @group(0) @binding(1) // force terms
    var<storage,read> force: array<f32>;

    @group(0) @binding(2) // a_x, b_x, k_x, a_x_h, b_x_h, k_x_h
    var<storage,read> coef_x: array<f32>;

    @group(0) @binding(3) // a_y, b_y, k_y, a_y_h, b_y_h, k_y_h
    var<storage,read> coef_y: array<f32>;

    @group(0) @binding(4) // param_int32
    var<storage,read_write> sim_int_par: SimIntValues;

    // Group 1 - simulation arrays
    @group(1) @binding(5) // velocity fields (vx, vy)
    var<storage,read_write> vel: array<f32>;

    @group(1) @binding(6) // stress fields (sigmaxx, sigmayy, sigmaxy)
    var<storage,read_write> sig: array<f32>;

    @group(1) @binding(7) // memory fields
                          // memory_dvx_dx, memory_dvx_dy, memory_dvy_dx, memory_dvy_dy,
                          // memory_dsigmaxx_dx, memory_dsigmayy_dy, memory_dsigmaxy_dx, memory_dsigmaxy_dy
    var<storage,read_write> memo: array<f32>;

    // Group 2 - sensors arrays and energies
    @group(2) @binding(8) // sensors signals (sisvx, sisvy)
    var<storage,read_write> sensors: array<f32>;

    @group(2) @binding(9) // epsilon fields
                          // epsilon_xx, epsilon_yy, epsilon_xy, vx_vy_2
    var<storage,read_write> eps: array<f32>;

    @group(2) @binding(10) // energy fields
                           // total_energy, total_energy_kinetic, total_energy_potential, v_solid_norm
    var<storage,read_write> energy: array<f32>;


    // -------------------------------
    // --- Index access functions ----
    // -------------------------------
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

    // ------------------------------------
    // --- Force array access funtions ---
    // ------------------------------------
    // function to get a force_x array value [force_x]
    fn get_force_x(n: i32) -> f32 {
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 2);

        return select(0.0, force[index], index != -1);
    }

    // function to get a force_y array value [force_y]
    fn get_force_y(n: i32) -> f32 {
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 2);

        return select(0.0, force[index], index != -1);
    }

    // -------------------------------------------------
    // --- CPML X coefficients array access funtions ---
    // -------------------------------------------------
    // function to get a a_x array value
    fn get_a_x(n: i32) -> f32 {
        let index: i32 = ij(n, 0, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a a_x_h array value
    fn get_a_x_h(n: i32) -> f32 {
        let index: i32 = ij(n, 3, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a b_x array value
    fn get_b_x(n: i32) -> f32 {
        let index: i32 = ij(n, 1, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a b_x_h array value
    fn get_b_x_h(n: i32) -> f32 {
        let index: i32 = ij(n, 4, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a k_x array value
    fn get_k_x(n: i32) -> f32 {
        let index: i32 = ij(n, 2, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // function to get a k_x_h array value
    fn get_k_x_h(n: i32) -> f32 {
        let index: i32 = ij(n, 5, sim_int_par.x_sz - 2, 6);

        return select(0.0, coef_x[index], index != -1);
    }

    // -------------------------------------------------
    // --- CPML Y coefficients array access funtions ---
    // -------------------------------------------------
    // function to get a a_y array value
    fn get_a_y(n: i32) -> f32 {
        let index: i32 = ij(0, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a a_y_h array value
    fn get_a_y_h(n: i32) -> f32 {
        let index: i32 = ij(3, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a b_y array value
    fn get_b_y(n: i32) -> f32 {
        let index: i32 = ij(1, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a b_y_h array value
    fn get_b_y_h(n: i32) -> f32 {
        let index: i32 = ij(4, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a k_y array value
    fn get_k_y(n: i32) -> f32 {
        let index: i32 = ij(2, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }

    // function to get a k_y_h array value
    fn get_k_y_h(n: i32) -> f32 {
        let index: i32 = ij(5, n, 6, sim_int_par.y_sz - 2);

        return select(0.0, coef_y[index], index != -1);
    }

    // ---------------------------------------
    // --- Velocity arrays access funtions ---
    // ---------------------------------------
    // function to get a vx array value
    fn get_vx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        return select(0.0, vel[index], index != -1);
    }

    // function to set a vx array value
    fn set_vx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        if(index != -1) {
            vel[index] = val;
        }
    }

    // function to get a vy array value
    fn get_vy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        return select(0.0, vel[index], index != -1);
    }

    // function to set a vy array value
    fn set_vy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 2);

        if(index != -1) {
            vel[index] = val;
        }
    }

    // -------------------------------------
    // --- Stress arrays access funtions ---
    // -------------------------------------
    // function to get a sigmaxx array value
    fn get_sigmaxx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        return select(0.0, sig[index], index != -1);
    }

    // function to set a sigmaxx array value
    fn set_sigmaxx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        if(index != -1) {
            sig[index] = val;
        }
    }

    // function to get a sigmayy array value
    fn get_sigmayy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        return select(0.0, sig[index], index != -1);
    }

    // function to set a sigmayy array value
    fn set_sigmayy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        if(index != -1) {
            sig[index] = val;
        }
    }

    // function to get a sigmaxy array value
    fn get_sigmaxy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        return select(0.0, sig[index], index != -1);
    }

    // function to set a sigmaxy array value
    fn set_sigmaxy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 3);

        if(index != -1) {
            sig[index] = val;
        }
    }

    // -------------------------------------
    // --- Memory arrays access funtions ---
    // -------------------------------------
    // function to get a memory_dvx_dx array value
    fn get_mdvx_dx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dvx_dx array value
    fn set_mdvx_dx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dvx_dy array value
    fn get_mdvx_dy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dvx_dy array value
    fn set_mdvx_dy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dvy_dx array value
    fn get_mdvy_dx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dvy_dx array value
    fn set_mdvy_dx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dvy_dy array value
    fn get_mdvy_dy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dvy_dy array value
    fn set_mdvy_dy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dsigmaxx_dx array value
    fn get_mdsxx_dx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 4, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dsigmaxx_dx array value
    fn set_mdsxx_dx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 4, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dsigmayy_dy array value
    fn get_mdsyy_dy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 5, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dsigmayy_dy array value
    fn set_mdsyy_dy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 5, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dsigmaxy_dx array value
    fn get_mdsxy_dx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 6, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dsigmaxy_dx array value
    fn set_mdsxy_dx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 6, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // function to get a memory_dsigmaxy_dy array value
    fn get_mdsxy_dy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 7, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        return select(0.0, memo[index], index != -1);
    }

    // function to set a memory_dsigmaxy_dy array value
    fn set_mdsxy_dy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 7, sim_int_par.x_sz, sim_int_par.y_sz, 8);

        if(index != -1) {
            memo[index] = val;
        }
    }

    // --------------------------------------
    // --- Sensors arrays access funtions ---
    // --------------------------------------
    // function to set a sens_vx array value
    fn set_sens_vx(n: i32, val : f32) {
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 2);

        if(index != -1) {
            sensors[index] = val;
        }
    }

    // function to set a sens_vy array value
    fn set_sens_vy(n: i32, val : f32) {
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 2);

        if(index != -1) {
            sensors[index] = val;
        }
    }

    // -------------------------------------
    // --- Epsilon arrays access funtions ---
    // -------------------------------------
    // function to get a epsilon_xx array value
    fn get_eps_xx(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }

    // function to set a epsilon_xx array value
    fn set_eps_xx(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 0, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            eps[index] = val;
        }
    }

    // function to get a epsilon_yy array value
    fn get_eps_yy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }

    // function to set a epsilon_yy array value
    fn set_eps_yy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 1, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            eps[index] = val;
        }
    }

    // function to get a epsilon_xy array value
    fn get_eps_xy(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }

    // function to set a epsilon_xy array value
    fn set_eps_xy(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 2, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            eps[index] = val;
        }
    }

    // function to get a vx_vy_2 array value
    fn get_vx_vy_2(x: i32, y: i32) -> f32 {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        return select(0.0, eps[index], index != -1);
    }

    // function to set a vx_vy_2 array value
    fn set_vx_vy_2(x: i32, y: i32, val : f32) {
        let index: i32 = ijk(x, y, 3, sim_int_par.x_sz, sim_int_par.y_sz, 4);

        if(index != -1) {
            eps[index] = val;
        }
    }

    // ------------------------------------
    // --- Energy array access funtions ---
    // ------------------------------------
    // function to get a total_energy array value [tot_en]
    fn get_tot_en(n: i32) -> f32 {
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }

    // function to set a total_energy array value
    fn set_tot_en(n: i32, val : f32) {
        let index: i32 = ij(n, 0, sim_int_par.n_iter, 4);

        if(index != -1) {
            energy[index] = val;
        }
    }

    // function to get a total_energy_kinetic array value [tot_en_k]
    fn get_tot_en_k(n: i32) -> f32 {
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }

    // function to set a total_energy_kinetic array value
    fn set_tot_en_k(n: i32, val : f32) {
        let index: i32 = ij(n, 1, sim_int_par.n_iter, 4);

        if(index != -1) {
            energy[index] = val;
        }
    }

    // function to get a total_energy_potencial array value [tot_en_p]
    fn get_tot_en_p(n: i32) -> f32 {
        let index: i32 = ij(n, 2, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }

    // function to set a total_energy_potencial array value
    fn set_tot_en_p(n: i32, val : f32) {
        let index: i32 = ij(n, 2, sim_int_par.n_iter, 4);

        if(index != -1) {
            energy[index] = val;
        }
    }

    // function to get a v_solid_norm array value [v_sol_n]
    fn get_v_sol_n(n: i32) -> f32 {
        let index: i32 = ij(n, 3, sim_int_par.n_iter, 4);

        return select(0.0, energy[index], index != -1);
    }

    // function to set a v_solid_norm array value
    fn set_v_sol_n(n: i32, val : f32) {
        let index: i32 = ij(n, 3, sim_int_par.n_iter, 4);

        if(index != -1) {
            energy[index] = val;
        }
    }

    // ---------------
    // --- Kernels ---
    // ---------------
    // Kernel to calculate stresses [sigmaxx, sigmayy, sigmaxy]
    @compute
    @workgroup_size(wsx, wsy)
    fn sigma_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dx: f32 = sim_flt_par.dx;
        let dy: f32 = sim_flt_par.dy;
        let dt: f32 = sim_flt_par.dt;
        let lambda: f32 = sim_flt_par.lambda;
        let mu: f32 = sim_flt_par.mu;
        let lambdaplus2mu: f32 = lambda + 2.0 * mu;

        // Normal stresses
        if(x >= 1 && x < (sim_int_par.x_sz - 2) && y >= 2 && y < (sim_int_par.y_sz - 1)) {
            var vdvx_dx: f32 = (27.0*(get_vx(x + 1, y) - get_vx(x, y)) - get_vx(x + 2, y) + get_vx(x - 1, y))/(24.0 * dx);
            var vdvy_dy: f32 = (27.0*(get_vy(x, y) - get_vy(x, y - 1)) - get_vy(x, y + 1) + get_vy(x, y - 2))/(24.0 * dy);

            var mdvx_dx_new: f32 = get_b_x_h(x - 1) * get_mdvx_dx(x, y) + get_a_x_h(x - 1) * vdvx_dx;
            var mdvy_dy_new: f32 = get_b_y(y - 1) * get_mdvy_dy(x, y) + get_a_y(y - 1) * vdvy_dy;

            vdvx_dx = vdvx_dx/get_k_x_h(x - 1) + mdvx_dx_new;
            vdvy_dy = vdvy_dy/get_k_y(y - 1)  + mdvy_dy_new;

            set_mdvx_dx(x, y, mdvx_dx_new);
            set_mdvy_dy(x, y, mdvy_dy_new);

            set_sigmaxx(x, y, get_sigmaxx(x, y) + (lambdaplus2mu * vdvx_dx + lambda        * vdvy_dy) * dt);
            set_sigmayy(x, y, get_sigmayy(x, y) + (lambda        * vdvx_dx + lambdaplus2mu * vdvy_dy) * dt);
        }

        // Shear stress
        if(x >= 2 && x < (sim_int_par.x_sz - 1) && y >= 1 && y < (sim_int_par.y_sz - 2)) {
            var vdvy_dx: f32 = (27.0*(get_vy(x, y) - get_vy(x - 1, y)) - get_vy(x + 1, y) + get_vy(x - 2, y))/(24.0 * dx);
            var vdvx_dy: f32 = (27.0*(get_vx(x, y + 1) - get_vx(x, y)) - get_vx(x, y + 2) + get_vx(x, y - 1))/(24.0 * dy);

            var mdvy_dx_new: f32 = get_b_x(x - 1) * get_mdvy_dx(x, y) + get_a_x(x - 1) * vdvy_dx;
            var mdvx_dy_new: f32 = get_b_y_h(y - 1) * get_mdvx_dy(x, y) + get_a_y_h(y - 1) * vdvx_dy;

            vdvy_dx = vdvy_dx/get_k_x(x - 1)   + mdvy_dx_new;
            vdvx_dy = vdvx_dy/get_k_y_h(y - 1) + mdvx_dy_new;

            set_mdvy_dx(x, y, mdvy_dx_new);
            set_mdvx_dy(x, y, mdvx_dy_new);

            set_sigmaxy(x, y, get_sigmaxy(x, y) + (vdvx_dy + vdvy_dx) * mu * dt);
        }
    }

    // Kernel to calculate velocities [vx, vy]
    @compute
    @workgroup_size(wsx, wsy)
    fn velocity_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
        let dx: f32 = sim_flt_par.dx;
        let dy: f32 = sim_flt_par.dy;

        // first step
        if(x >= 2 && x < (sim_int_par.x_sz - 1) && y >= 2 && y < (sim_int_par.y_sz - 1)) {
            var vdsigmaxx_dx: f32 = (27.0*(get_sigmaxx(x, y) - get_sigmaxx(x - 1, y)) - get_sigmaxx(x + 1, y) + get_sigmaxx(x - 2, y))/(24.0 * dx);
            var vdsigmaxy_dy: f32 = (27.0*(get_sigmaxy(x, y) - get_sigmaxy(x, y - 1)) - get_sigmaxy(x, y + 1) + get_sigmaxy(x, y - 2))/(24.0 * dy);

            var mdsxx_dx_new: f32 = get_b_x(x - 1) * get_mdsxx_dx(x, y) + get_a_x(x - 1) * vdsigmaxx_dx;
            var mdsxy_dy_new: f32 = get_b_y(y - 1) * get_mdsxy_dy(x, y) + get_a_y(y - 1) * vdsigmaxy_dy;

            vdsigmaxx_dx = vdsigmaxx_dx/get_k_x(x - 1) + mdsxx_dx_new;
            vdsigmaxy_dy = vdsigmaxy_dy/get_k_y(y - 1) + mdsxy_dy_new;

            set_mdsxx_dx(x, y, mdsxx_dx_new);
            set_mdsxy_dy(x, y, mdsxy_dy_new);

            set_vx(x, y, dt_over_rho * (vdsigmaxx_dx + vdsigmaxy_dy) + get_vx(x, y));
        }

        // second step
        if(x >= 1 && x < (sim_int_par.x_sz - 2) && y >= 1 && y < (sim_int_par.y_sz - 2)) {
            var vdsigmaxy_dx: f32 = (27.0*(get_sigmaxy(x + 1, y) - get_sigmaxy(x, y)) - get_sigmaxy(x + 2, y) + get_sigmaxy(x - 1, y))/(24.0 * dx);
            var vdsigmayy_dy: f32 = (27.0*(get_sigmayy(x, y + 1) - get_sigmayy(x, y)) - get_sigmayy(x, y + 2) + get_sigmayy(x, y - 1))/(24.0 * dy);

            var mdsxy_dx_new: f32 = get_b_x_h(x - 1) * get_mdsxy_dx(x, y) + get_a_x_h(x - 1) * vdsigmaxy_dx;
            var mdsyy_dy_new: f32 = get_b_y_h(y - 1) * get_mdsyy_dy(x, y) + get_a_y_h(y - 1) * vdsigmayy_dy;

            vdsigmaxy_dx = vdsigmaxy_dx/get_k_x_h(x - 1) + mdsxy_dx_new;
            vdsigmayy_dy = vdsigmayy_dy/get_k_y_h(y - 1) + mdsyy_dy_new;

            set_mdsxy_dx(x, y, mdsxy_dx_new);
            set_mdsyy_dy(x, y, mdsyy_dy_new);

            set_vy(x, y, dt_over_rho * (vdsigmaxy_dx + vdsigmayy_dy) + get_vy(x, y));
        }
    }

    // Kernel to finish iteration term
    @compute
    @workgroup_size(wsx, wsy)
    fn finish_it_kernel(@builtin(global_invocation_id) index: vec3<u32>) {
        let x: i32 = i32(index.x);          // x thread index
        let y: i32 = i32(index.y);          // y thread index
        let dt_over_rho: f32 = sim_flt_par.dt / sim_flt_par.rho;
        let x_source: i32 = sim_int_par.x_source;
        let y_source: i32 = sim_int_par.y_source;
        let x_sens: i32 = sim_int_par.x_sens;
        let y_sens: i32 = sim_int_par.y_sens;
        let it: i32 = sim_int_par.it;
        let xmin: i32 = sim_int_par.np_pml;
        let xmax: i32 = sim_int_par.x_sz - sim_int_par.np_pml;
        let ymin: i32 = sim_int_par.np_pml;
        let ymax: i32 = sim_int_par.y_sz - sim_int_par.np_pml;
        let rho_2: f32 = sim_flt_par.rho * 0.5;
        let lambda: f32 = sim_flt_par.lambda;
        let mu: f32 = sim_flt_par.mu;
        let lambdaplus2mu: f32 = lambda + 2.0 * mu;
        let denom: f32 = 4.0 * mu * (lambda + mu);
        let mu2: f32 = 2.0 * mu;

        // Add the source force
        if(x == x_source && y == y_source) {
            set_vx(x, y, get_vx(x, y) + get_force_x(it) * dt_over_rho);
            set_vy(x, y, get_vy(x, y) + get_force_y(it) * dt_over_rho);
        }

        // Apply Dirichlet conditions
        if(x <= 1 || x >= (sim_int_par.x_sz - 2) || y <= 1 || y >= (sim_int_par.y_sz - 2)) {
            set_vx(x, y, 0.0);
            set_vy(x, y, 0.0);
        }

        // Store sensor velocities
        if(x == x_sens && y == y_sens) {
            set_sens_vx(it, get_vx(x, y));
            set_sens_vy(it, get_vy(x, y));
        }

        // Compute total energy in the medium (without the PML layers)
        set_vx_vy_2(x, y, get_vx(x, y)*get_vx(x, y) + get_vy(x, y)*get_vy(x, y));
        if(x >= xmin && x < xmax && y >= ymin && y < ymax) {
            set_tot_en_k(it, rho_2 * get_vx_vy_2(x, y) + get_tot_en_k(it));

            set_eps_xx(x, y, (lambdaplus2mu*get_sigmaxx(x, y) - lambda*get_sigmayy(x, y))/denom);
            set_eps_yy(x, y, (lambdaplus2mu*get_sigmayy(x, y) - lambda*get_sigmaxx(x, y))/denom);
            set_eps_xy(x, y, get_sigmaxy(x, y)/mu2);

            set_tot_en_p(it, 0.5 * (get_eps_xx(x, y)*get_sigmaxx(x, y) +
                                    get_eps_yy(x, y)*get_sigmayy(x, y) +
                                    2.0 * get_eps_xy(x, y)* get_sigmaxy(x, y)));

            set_tot_en(it, get_tot_en_k(it) + get_tot_en_p(it));
        }
    }

    // Kernel to increase time iteraction [it]
    @compute
    @workgroup_size(1)
    fn incr_it_kernel() {
        sim_int_par.it += 1;
    }
