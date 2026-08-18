function [T_L1, T_L2, T_R1, T_R2] = Tractor_ASTSMC_Control( ...
    lambda_L1, lambda_L2, lambda_R1, lambda_R2, ...
    Fx_L1_in, Fx_L2_in, Fx_R1_in, Fx_R2_in, F_load_in)
%#codegen
% ==============================================================
% Tractor_ASTSMC_Control
%
% Boundary-layer adaptive super-twisting sliding-mode controller
% for four-wheel-independent tractor slip-ratio regulation.
%
% Wheel identifiers:
%   L1: front-left wheel
%   R1: front-right wheel
%   L2: rear-left wheel
%   R2: rear-right wheel
%
% Inputs:
%   lambda_L1, lambda_R1
%       Estimated front-wheel slip ratios [-]
%
%   lambda_L2, lambda_R2
%       Estimated rear-wheel slip ratios [-]
%
%   Fx_L1_in, Fx_R1_in
%       Front-wheel net longitudinal forces [N]
%
%   Fx_L2_in, Fx_R2_in
%       Rear-wheel net longitudinal forces [N]
%
%   F_load_in
%       External traction-load signal [N]
%       The upstream model uses a negative signal for the
%       resistive traction load.
%
% Outputs:
%   T_L1, T_R1
%       Front-wheel drive-torque commands [N*m]
%
%   T_L2, T_R2
%       Rear-wheel drive-torque commands [N*m]
%
% Controller implementation frequency:
%   f_s = 1000 Hz
%   T_s = 0.001 s
%
% Control law:
%
%   T_i,j = T_eq,i,j + T_st,i,j
%
%   T_eq,i,j =
%       F_x,i,j,net R_i
%       + J_i/[m R_i (1-lambda_d)]
%         (F_x,net-F_load)
%
%   T_st,i,j =
%       -k_1,i,j |s_i,j|^(1/2)
%        sat(s_i,j/phi_i,j) + z_i,j
%
%   dz_i,j/dt =
%       -k_2,i,j sat(s_i,j/phi_i,j)
%       -l_z,i,j z_i,j
%
%   s_i,j = lambda_i,j-lambda_d
%
% The upstream slip-ratio calculation should disable or
% regularize slip feedback when the wheel circumferential
% speed is below its prescribed low-speed threshold.
% ==============================================================

    %% =========================================================
    % 1. Sampling frequency and reference slip ratio
    % ==========================================================
    control_frequency = 1000.0;    % Controller frequency [Hz]
    Ts = 1.0 / control_frequency;  % Sampling interval [s]

    lambda_d = 0.20;               % Desired slip ratio [-]

    %% =========================================================
    % 2. Tractor parameters
    % ==========================================================
    m = 7750.0;                    % Tractor mass [kg]

    R_F = 0.680;                   % Front-wheel radius [m]
    R_R = 0.860;                   % Rear-wheel radius [m]

    J_F = 25.0;                    % Front-wheel inertia [kg*m^2]
    J_R = 85.0;                    % Rear-wheel inertia [kg*m^2]

    %% =========================================================
    % 3. Front-axle ASTSMC parameters
    % ==========================================================
    k1_0_F = 2850.0;               % Baseline first-order gain [N*m]
    k2_0_F = 700.0;                % Baseline second-order gain [N*m/s]

    a1_F = 700.0;                  % First-order adaptive coefficient [N*m]
    a2_F = 180.0;                  % Second-order adaptive coefficient [N*m/s]

    gamma_F = 3.8;                 % Gain-increase rate [1/s]
    eta_F = 38.0;                  % Gain-decay rate [1/s]

    rho_min_F = 0.16;              % Minimum adaptive variable [-]
    rho_max_F = 1.10;              % Maximum adaptive variable [-]

    mu_F = 0.0100;                 % Adaptive threshold [-]
    phi_F = 0.055;                 % Boundary-layer thickness [-]

    lz_F = 0.48;                   % Auxiliary-state leakage [1/s]

    %% =========================================================
    % 4. Rear-axle ASTSMC parameters
    % Rear-wheel torque response is moderately increased
    % ==========================================================
    k1_0_R = 3200.0;               % Increased from 3050 [N*m]
    k2_0_R = 560.0;                % Baseline second-order gain [N*m/s]

    a1_R = 600.0;                  % Increased from 570 [N*m]
    a2_R = 115.0;                  % Second-order adaptive coefficient [N*m/s]

    gamma_R = 2.20;                % Gain-increase rate [1/s]
    eta_R = 54.0;                  % Gain-decay rate [1/s]

    rho_min_R = 0.14;              % Minimum adaptive variable [-]
    rho_max_R = 0.92;              % Maximum adaptive variable [-]

    mu_R = 0.0100;                 % Adaptive threshold [-]
    phi_R = 0.050;                 % Boundary-layer thickness [-]

    lz_R = 0.82;                   % Auxiliary-state leakage [1/s]

    %% =========================================================
    % 5. Drive-torque constraints
    % ==========================================================
    T_min_F = 0.0;                 % Front minimum torque [N*m]
    T_max_F = 2450.0;              % Front maximum torque [N*m]

    T_min_R = 0.0;                 % Rear minimum torque [N*m]
    T_max_R = 2800.0;              % Increased from 2550 [N*m]

    %% =========================================================
    % 6. Persistent controller states
    % ==========================================================
    persistent z_L1 z_L2 z_R1 z_R2
    persistent rho_L1 rho_L2 rho_R1 rho_R2

    if isempty(z_L1)
        z_L1 = 0.0;
        z_R1 = 0.0;
        z_L2 = 0.0;
        z_R2 = 0.0;

        rho_L1 = rho_min_F;
        rho_R1 = rho_min_F;
        rho_L2 = rho_min_R;
        rho_R2 = rho_min_R;
    end

    %% =========================================================
    % 7. Input validation
    % ==========================================================
    lambda_L1_use = sanitize_slip_ratio(lambda_L1, lambda_d);
    lambda_R1_use = sanitize_slip_ratio(lambda_R1, lambda_d);
    lambda_L2_use = sanitize_slip_ratio(lambda_L2, lambda_d);
    lambda_R2_use = sanitize_slip_ratio(lambda_R2, lambda_d);

    % These inputs must correspond to the net longitudinal
    % wheel forces F_x,i,j,net used in the manuscript.
    Fx_net_L1 = sanitize_scalar(Fx_L1_in, 0.0);
    Fx_net_R1 = sanitize_scalar(Fx_R1_in, 0.0);
    Fx_net_L2 = sanitize_scalar(Fx_L2_in, 0.0);
    Fx_net_R2 = sanitize_scalar(Fx_R2_in, 0.0);

    % Convert the signed negative resistive-load signal used
    % by the Simulink model into the positive load magnitude
    % F_load used in the manuscript equations.
    F_load_signal = sanitize_scalar(F_load_in, 0.0);
    F_load = abs(F_load_signal);

    %% =========================================================
    % 8. Total four-wheel net longitudinal force
    % ==========================================================
    Fx_net_total = Fx_net_L1 + Fx_net_R1 + ...
                   Fx_net_L2 + Fx_net_R2;

    %% =========================================================
    % 9. Front-left-wheel ASTSMC
    % ==========================================================
    [T_L1, z_L1, rho_L1] = astsmc_single_wheel( ...
        lambda_L1_use, lambda_d, ...
        Fx_net_L1, Fx_net_total, F_load, ...
        R_F, J_F, m, Ts, ...
        k1_0_F, k2_0_F, a1_F, a2_F, ...
        gamma_F, eta_F, ...
        rho_min_F, rho_max_F, ...
        mu_F, phi_F, lz_F, ...
        z_L1, rho_L1, ...
        T_min_F, T_max_F);

    %% =========================================================
    % 10. Front-right-wheel ASTSMC
    % ==========================================================
    [T_R1, z_R1, rho_R1] = astsmc_single_wheel( ...
        lambda_R1_use, lambda_d, ...
        Fx_net_R1, Fx_net_total, F_load, ...
        R_F, J_F, m, Ts, ...
        k1_0_F, k2_0_F, a1_F, a2_F, ...
        gamma_F, eta_F, ...
        rho_min_F, rho_max_F, ...
        mu_F, phi_F, lz_F, ...
        z_R1, rho_R1, ...
        T_min_F, T_max_F);

    %% =========================================================
    % 11. Rear-left-wheel ASTSMC
    % ==========================================================
    [T_L2, z_L2, rho_L2] = astsmc_single_wheel( ...
        lambda_L2_use, lambda_d, ...
        Fx_net_L2, Fx_net_total, F_load, ...
        R_R, J_R, m, Ts, ...
        k1_0_R, k2_0_R, a1_R, a2_R, ...
        gamma_R, eta_R, ...
        rho_min_R, rho_max_R, ...
        mu_R, phi_R, lz_R, ...
        z_L2, rho_L2, ...
        T_min_R, T_max_R);

    %% =========================================================
    % 12. Rear-right-wheel ASTSMC
    % ==========================================================
    [T_R2, z_R2, rho_R2] = astsmc_single_wheel( ...
        lambda_R2_use, lambda_d, ...
        Fx_net_R2, Fx_net_total, F_load, ...
        R_R, J_R, m, Ts, ...
        k1_0_R, k2_0_R, a1_R, a2_R, ...
        gamma_R, eta_R, ...
        rho_min_R, rho_max_R, ...
        mu_R, phi_R, lz_R, ...
        z_R2, rho_R2, ...
        T_min_R, T_max_R);
end


function [T_out, z_new, rho_new] = astsmc_single_wheel( ...
    lambda_hat, lambda_d, ...
    Fx_net_ij, Fx_net_total, F_load, ...
    R_i, J_i, m, Ts, ...
    k1_0, k2_0, a1, a2, ...
    gamma, eta, ...
    rho_min, rho_max, ...
    mu, phi, lz, ...
    z_old, rho_old, ...
    T_min, T_max)
%#codegen
% ==============================================================
% Boundary-layer ASTSMC for one driven wheel
% ==============================================================

    %% Sliding surface
    s = lambda_hat - lambda_d;
    abs_s = abs(s);

    %% Current bounded adaptive variable
    rho_use = limit_scalar(rho_old, rho_min, rho_max);

    %% Current adaptive super-twisting gains
    k1 = k1_0 + a1 * rho_use;
    k2 = k2_0 + a2 * rho_use;

    %% Boundary-layer saturation function
    sat_s = boundary_layer_sat(s, phi);

    %% Equivalent control torque
    denominator = m * R_i * (1.0 - lambda_d);

    if denominator <= 1.0e-12
        denominator = 1.0e-12;
    end

    T_eq = Fx_net_ij * R_i + ...
        J_i / denominator * (Fx_net_total - F_load);

    %% Super-twisting control torque
    T_st = -k1 * sqrt(abs_s) * sat_s + z_old;

    %% Complete unconstrained drive-torque command
    T_unsat = T_eq + T_st;

    %% Physical drive-torque constraint
    T_out = limit_scalar(T_unsat, T_min, T_max);

    %% Unconstrained adaptive rate
    if abs_s >= mu
        nu = gamma * (abs_s - mu);
    else
        nu = -eta * (rho_use - rho_min);
    end

    %% Projection adaptive law
    rho_dot = projection_operator( ...
        rho_use, nu, rho_min, rho_max);

    %% Adaptive-variable state update
    rho_new = rho_use + Ts * rho_dot;

    % Numerical protection for finite-step discretization
    rho_new = limit_scalar(rho_new, rho_min, rho_max);

    %% Auxiliary-state dynamics
    z_dot = -k2 * sat_s - lz * z_old;

    %% Auxiliary-state update
    z_new = z_old + Ts * z_dot;

    if ~isfinite(z_new)
        z_new = 0.0;
    end
end


function rho_dot = projection_operator( ...
    rho, nu, rho_min, rho_max)
%#codegen
% ==============================================================
% Projection operator for the bounded adaptive variable
% ==============================================================

    if rho >= rho_max && nu > 0.0
        rho_dot = 0.0;

    elseif rho <= rho_min && nu < 0.0
        rho_dot = 0.0;

    else
        rho_dot = nu;
    end
end


function y = boundary_layer_sat(s, phi)
%#codegen
% ==============================================================
% Boundary-layer saturation function
% ==============================================================

    if phi <= 1.0e-12
        if s > 0.0
            y = 1.0;
        elseif s < 0.0
            y = -1.0;
        else
            y = 0.0;
        end
        return;
    end

    if abs(s) <= phi
        y = s / phi;

    elseif s > 0.0
        y = 1.0;

    elseif s < 0.0
        y = -1.0;

    else
        y = 0.0;
    end
end


function lambda_out = sanitize_slip_ratio( ...
    lambda_in, fallback)
%#codegen
% ==============================================================
% Numerical protection for the estimated traction slip ratio
% ==============================================================

    if isfinite(lambda_in)
        lambda_out = lambda_in;
    else
        lambda_out = fallback;
    end

    lambda_out = limit_scalar(lambda_out, 0.0, 0.999);
end


function y = sanitize_scalar(x, fallback)
%#codegen
% ==============================================================
% Replace a non-finite scalar input with a fixed fallback
% ==============================================================

    if isfinite(x)
        y = x;
    else
        y = fallback;
    end
end


function y = limit_scalar(x, xmin, xmax)
%#codegen
% ==============================================================
% Scalar saturation function
% ==============================================================

    if ~isfinite(x)
        y = xmin;

    elseif x < xmin
        y = xmin;

    elseif x > xmax
        y = xmax;

    else
        y = x;
    end
end