% Clear all variables and close all existing figures.
clear all
close all

addpath('.\functions')

% Set seed for reproducibility
seed = 264;
rng(seed);

nDeployments = 100;
% Set the number of measurements per sensor, starting from 0
K = 251;
% Set number of trials.
nTrials = 500;

% Check if parpool instance is already running
pool = gcp('nocreate');
% If not, then start one
if isempty(pool)
    parpool('local');
end

% Set number of sensors
sensor_vals = 5; % [5,10]
dropout_vals = 0;
sensor_dimension = length(sensor_vals);

% Set observation window length in seconds.
T0 = 4;

% Define event signal parameters.
Tp = 1;
w0 = pi;

% Define time domain parameters
dt = T0/(K-1);
t = dt*(0:K-1)';
three_dim_t = reshape(t,1,1,K);

% Define frequency domain parameters
fs = 1/dt;
L = K-1;
f = 0:fs/(2*L):fs/2;
omega = 2*pi*f;

% Calculate the signal energy (such that the signal is assumed to be within the
% interior to the observation period).
func = @(t) abs(sensor_signal(t, Tp, 1)).^2;
Ps = integral(func,0,T0)/T0;

% Define true parameter values.
alpha_true = 2;
t0_true = 1;

% Define range for distribution of mi and ti (tau_i)
max_mi = 1;
min_mi = 0.5;

max_ti = 1;
min_ti = 0;

% Set max tpi value based on sampling rate to give a frame of reference as to
% how many sample the error may shift the peak
max_tpi = 0.02;
min_tpi = 0;

% Set maximum value of t0 such that the signal from each sensor is guaranteed to
% be in the interior of the observation period.
t0_max = T0-Tp-max_ti-max_tpi;

if t0_true > t0_max
    error("True value of t0 exceeds maximum!")
end

%%% Always generate w, n, and channel gain for maximum S, then use subsets for different S trials
S_max = max(sensor_vals);
% Generate sensor noise with dimensions: K x S x trials x deployments.
all_w = randn(K,S_max,nTrials,nDeployments);
% Generate server noise.
n = (randn(K,S_max,nTrials,nDeployments) + 1i*randn(K,S_max,nTrials,nDeployments));
% Generate channel gains.
all_gi = (randn(1,S_max,1,nDeployments) + 1i*randn(1,S_max,1,nDeployments))/sqrt(2);

% Generate mi and ti.
all_mi = sort(unifrnd(min_mi,max_mi,1,S_max,1,nDeployments),2,'descend');
all_ti = sort(unifrnd(min_ti,max_ti,1,S_max,1,nDeployments),2,'ascend');
all_tpi = zeros(1,S_max,1,nDeployments);

selected_schemes = ["EPC"];

% Define Rayleigh distribution parameters.
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqr = 2*rayleigh_factor^2;

% Define expected value of mi^2.
E_mi_sqr = (1/12) * (max_mi-min_mi)^2 + 0.5*(min_mi+max_mi);

% Set miscellaneous parameters
D_val = 0.5;
norm_fact = 1;

% Choose which experiment to run.
% experiment_list = ["cwe_disjoint_peak_power_opt"; 
%     "cwe_disjoint_peak_power_opt_diff_rho"; 
%     "cwe_disjoint_peak_power_opt_bottom_clipping_nonclipped";
%     "cwe_disjoint_peak_power_opt_bottom_clipping_clipped"; 
%     "cwe_disjoint_peak_power_opt_diff_weights";
%     "cwe_disjoint_peak_power_opt_diff_bounds_on_alpha"; 
%     "cwe_disjoint_peak_power_opt_diff_obj_functions";
%     "cwe_disjoint_peak_power_opt_asymp_approx"];

experiment_list = ["cwe_disjoint_peak_power_opt"; 
    "cwe_disjoint_peak_power_opt_diff_rho"; 
    "cwe_disjoint_peak_power_opt_bottom_clipping_nonclipped";
    "cwe_disjoint_peak_power_opt_bottom_clipping_clipped"; 
    "cwe_disjoint_peak_power_opt_diff_weights";
    "cwe_disjoint_peak_power_opt_diff_bounds_on_alpha"; 
    "cwe_disjoint_peak_power_opt_diff_obj_functions";
    "cwe_disjoint_peak_power_opt_asymp_approx"];

for experiment_idx = 1:numel(experiment_list)
    experiment = experiment_list(experiment_idx);

    pad_str = '----------';
    msg = [pad_str  ' ' 'Running: ' char(experiment) ' ' pad_str];
    fprintf('\n%s\n', msg);

    % Configure strategies
    agent_db_values = 0;
    if contains(experiment, "diff_rho")
        % Define channel snr values.
        channel_snr = 15;
        rho_vals = [0.5, 0.75, 1, 1.25, 1.5];
    
        pivot_vals = rho_vals;
    
        constraints = ["indiv."];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.","hetero."];
        approaches = ["prop. scaling", "opt."];
    
    elseif contains(experiment, "bottom_clipping_nonclipped")
        channel_snr = linspace(0,18,5);
        rho_vals = 0.75;
    
        pivot_vals = rho_vals;
        
        constraints = ["indiv."];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."];
        approaches = ["prop. scaling", "opt."];
    
    elseif contains(experiment, "bottom_clipping_clipped")
        channel_snr = linspace(0,18,5);
        rho_vals = 0.1;
    
        pivot_vals = rho_vals;
        
        constraints = ["indiv."];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."];
        approaches = ["prop. scaling", "opt."];
    
    elseif contains(experiment, "diff_weights")
        channel_snr = linspace(0,18,5);
        rho_vals = 0.5;
        weight_vals = [1, 1e1, 1e2, 1e3, 1e4];
    
        pivot_vals = weight_vals;
    
        constraints = ["indiv."];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."];
        approaches = ["prop. scaling", "opt."];
    
    elseif contains(experiment, "diff_bounds_on_alpha")
        channel_snr = linspace(0,18,5);
        rho_vals = 0.5;
        weight_vals = 1;
        beta_vals = [1, 2, 5, 10];
    
        pivot_vals = beta_vals;
    
        constraints = ["indiv."];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."];
        approaches = ["prop. scaling", "opt."];
    
    elseif contains(experiment, "asymp_approx")
        channel_snr = 0;
        rho_vals = 0.5;
        weight_vals = 1;
        beta_vals = 1;
        agent_db_values = linspace(0,18,5);
    
        pivot_vals = rho_vals;
    
        constraints = ["indiv.", "total", "mixed"];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."];
        approaches = ["asymp. approx.", "prop. scaling", "opt."];
    
    elseif contains(experiment, "diff_obj_functions")
        channel_snr = linspace(0,18,5);
        rho_vals = 0.5;
        weight_vals = 1;
        beta_vals = 1;
    
        obj_function_vals = ["crlb sum", "fim trace", "fim det"];
    
        pivot_vals = obj_function_vals;
    
        constraints = ["indiv."];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."];
        approaches = ["opt."];
    
    else
        channel_snr = linspace(0,18,5);
        rho_vals = 0.5;
    
        pivot_vals = rho_vals;
        
        constraints = ["indiv.", "total", "mixed"];
        transforms = ["linear", "affine"];
        coeff_types = ["homog.", "hetero."]; 
        approaches = ["prop. scaling", "opt."];
    end
    
    iter_arr = [];
    all_strats = [];
    
    for i = 1:numel(constraints)
        for j = 1:numel(approaches)
            for k = 1:numel(coeff_types)
                for l = 1:numel(transforms)
    
                    if constraints(i) ~= "indiv."
                        if coeff_types(k) == "hetero." && approaches(j) == "prop. scaling"
                            continue
                        end
                    end
    
                    if coeff_types(k) == "homog." && approaches(j) == "asymp. approx."
                        continue
                    end
    
                    % Do not include homog. asymp. approx in strats
                    if coeff_types(k) == "homog." && approaches(j) == "opt."
                        continue
    
                    elseif contains(experiment, "diff_rho") && coeff_types(k) == 'hetero.' && approaches(j) == 'asymp. approx'
                        continue
                    else
                        if i == 1
                            iter_arr = [iter_arr; coeff_types(k) + " " + transforms(l) + " " + approaches(j)];
                        end
                        all_strats = [all_strats; constraints(i) + " " + coeff_types(k) + " " + transforms(l) + " " + approaches(j)];
                    end 
                end
            end
        end
    end
    
    if contains(experiment, "bottom_clipping")
        all_strats = [all_strats; "indiv. bottom clipping"];
    end
    
    iter_length = length(transforms) * length(coeff_types) * length(approaches);
    
    num_strats = length(all_strats);
    opt_coeffs = zeros(num_strats,2*max(sensor_vals),1,nDeployments);
    
    % Duplicate channel snr matrices for experiments.
    channel_db_values = repmat(channel_snr, length(selected_schemes), 1, length(agent_db_values));
    
    % Initialize empty performance metric matrices
    rho_crlb = zeros(num_strats,length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),length(rho_vals),sensor_dimension,nDeployments);
    rho_empirical_var = rho_crlb;
    rho_empirical_mse = rho_crlb;
    rho_empirical_bias = rho_crlb;
    
    % Initialize empty energy matrices
    empirical_total_energy = zeros(num_strats,1,1,nDeployments);
    empirical_sensor_energy = zeros(num_strats,max(sensor_vals),1,nDeployments);
    
    clean_y_arr = zeros(num_strats,K,1,nDeployments);
    
    disp("Initializing parallel pooling...")
    if isempty(gcp('nocreate'))
        parpool('Processes');
    end
    
    % Start run timer
    disp("Starting runtime...")
    loopTic = tic;
    
    % Iterate through agent snr values.
    for agent_db_idx = 1:length(agent_db_values)
        agent_db = agent_db_values(agent_db_idx);
    
        % Set sensor noise power. Note the factor of 1/dt, which is equivalent to
        % applying an anti-aliasing filter.
        w_psd_constant = 1;
        gamma_w = (dt/w_psd_constant) * (Ps / db2magTen(agent_db));
        scaled_w = sqrt(gamma_w * w_psd_constant /dt) * all_w; % --> variance of this should be gamma_w/dt
        scaled_w_psd_constant = gamma_w * w_psd_constant; %% = Nw/2
        
        for pivot_idx = 1:length(pivot_vals)
            if contains(experiment, "diff_weights")
                rho = rho_vals(1);
                t0_weight = weight_vals(pivot_idx);
                beta = 1;
                obj_func = "crlb sum";
            elseif contains(experiment, "diff_bounds_on_alpha")
                rho = rho_vals(1);
                t0_weight = 1;
                beta = beta_vals(pivot_idx);
                obj_func = "crlb sum";
            elseif contains(experiment, "diff_obj_functions")
                rho = rho_vals(1);
                t0_weight = 1;
                beta = 1;
                obj_func = obj_function_vals(pivot_idx);
            else
                rho = rho_vals(pivot_idx);
                t0_weight = 1;
                beta = 1;
                obj_func = "crlb sum";
            end

            % Iterate through sensor values.
            for sensor_idx = 1:length(sensor_vals)
                S = sensor_vals(sensor_idx);
        
                % Iterate through dropout values.
                for dropout_idx = 1:length(dropout_vals)
                    num_dropout = dropout_vals(dropout_idx);
        
                    if experiment == "random_dropout"
                        save_dim = dropout_idx;
                    else
                        save_dim = sensor_idx;
                    end
        
                    % Select subset of mi, ti, and gi values.
                    mi = all_mi(:,1:S,:,:);
                    ti = all_ti(:,1:S,:,:);
                    tpi = all_tpi(:,1:S,:,:);
                    gi = all_gi(:,1:S,:,:);
                    delta_i = zeros(size(mi));
        
                    % Calculate quantized phase.
                    M = 4;
                    [qgi,~] = quantize_comp(gi,M);
    
                    E0 = sum(sensor_signal(t-t0_true, Tp, norm_fact).^2 .* dt, 1);
    
                    Gamma_i = gamma_w .* mi.^2 .* E0;
        
                    noisy_unified_xi = alpha_true .* (mi .* sensor_signal(t-t0_true, Tp, norm_fact)) + scaled_w(:,1:S,:,:);
    
                    Mi_true = alpha_true .* mi.^2 .* E0;
                    Gi = Mi_true + Gamma_i;
    
                    P0_max = mean((Gi).^2, "all");
                    Pi_max = rho * P0_max; % rho * (Mi + Gamma_i).^2;
                    P_max = D_val * S * Pi_max;
    
                    alpha_max = beta * alpha_true;
                    Mi = alpha_max .* mi.^2 .* E0;
                    Ci_linear = (Mi + Gamma_i).^2;
                    Ci_affine = (Mi./2 + Gamma_i).^2;
    
                    Ai_linear = sqrt(Pi_max ./ Ci_linear);
                    Ai_affine = sqrt(Pi_max ./ Ci_affine);
    
                    % Define search space for t0
                    t0_search_space = repmat(reshape(t,1,1,[]), 1,1,1,nDeployments);
                    [~, ui_t0_search] = mf_integral_fft(mi .* sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* sensor_signal(t, Tp, norm_fact), 1, 1, K, dt, Tp);
                    % Precompute FFT for matched filter template
                    mfTemplate = mi .* sensor_signal(t, Tp, norm_fact);
                    % FFT size = 2*K - 1
                    N = 2*K - 1;
                    mfTemplateFFT = fft(mfTemplate, N, 1);
    
                    % Iterate through channel snr values.
                    for channel_db_idx = 1:size(channel_db_values,2)
                        channel_db_start = tic;
                        % Iterate through schemes.
                        for scheme_idx = 1:length(selected_schemes)
                            scheme = selected_schemes(scheme_idx);
        
                            % Print statement for at-a-glance performance.
                            disp('')
                            disp("=== " + scheme + " ===")
                            disp("** Agent SNR = " + (agent_db_values(agent_db_idx)) + " dB **")
                            disp("** Channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB **")
                            disp("** S = " + S + ", Dropout = " + dropout_vals(dropout_idx) + " **")
        
                            n_psd_constant = 1;
                            % Define channel gains and server noise power.
                            if scheme == "MPC"
                                ci = ones(size(gi));
                                gamma_n = (dt/n_psd_constant) * Ps * E_mi_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                            elseif scheme == "EPC"
                                ci = abs(gi);
                                gamma_n = (dt/n_psd_constant) * Ps * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                            elseif scheme == "PPC"
                                ci = qgi;
                                gamma_n = (dt/n_psd_constant) * Ps * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                            elseif scheme == "NPC"
                                ci = gi;
                                gamma_n = (dt/n_psd_constant) * Ps * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                            end
        
                            % Set server noise power. Note the factor of 1/dt which is
                            % equivalent to applying an anti-aliasing filter.
                            scaled_n = sqrt( (gamma_n*n_psd_constant/dt) / 2 ) * n;
                            scaled_n_psd_constant = gamma_n * n_psd_constant; % == N0/2
        
                            epsilon = 1e-4;
                            norm_fact = 1;
                            Bee = pi/Tp;
        
                            mag_sqr_S0_internal = @(w) (norm_fact^2) * (2*Bee.^2 .* (1 + cos(w.*pi./Bee)) ) ./ (w.^2 - Bee.^2).^2;
                            mag_sqr_S0 = @(w) zeroIfnan(mag_sqr_S0_internal(w)) + (w == Bee | w == -Bee) .* (mag_sqr_S0_internal(w-epsilon) + mag_sqr_S0_internal(w+epsilon))./2;
        
                            %%%%%%%%%%%%%%%%% Define Anonymous Expressions %%%%%%%%%%%%%%%%%
                            if contains(experiment, "cwe") || experiment == "random_dropout"
                                if (scheme == "MPC" || scheme == "EPC")
                                    V_arr_internal = @(w, ai, mi, ci) scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0_internal(w) + scaled_n_psd_constant/2;
                                    V_arr = @(w, ai, mi, ci) zeroIfnan(V_arr_internal(w, ai, mi, ci)) + (w == Bee | w == -Bee) .* (V_arr_internal(w-epsilon, ai, mi, ci) + V_arr_internal(w+epsilon, ai, mi, ci))./2;
        
                                    H_arr_internal = @(w, ai, mi, ci) (1 + epsilon) ./ (V_arr_internal(w, ai, mi, ci) + epsilon);
                                    H_arr = @(w, ai, mi, ci) zeroIfnan(H_arr_internal(w, ai, mi, ci)) + (w == Bee | w == -Bee) .* (H_arr_internal(w-epsilon, ai, mi, ci) + H_arr_internal(w+epsilon, ai, mi, ci))./2;
        
                                    alpha_objective_array = @(ai, mi, ci) (2*pi) ./( ((sum(ci .* ai .* mi.^2)).^2) .* integral(@(w) (mag_sqr_S0(w)).^2 / V_arr(w, ai, mi, ci), -Inf, Inf, 'ArrayValued', true));
                                    t0_objective_array = @(ai, mi, ci, arb_alpha) (2*pi) ./( (arb_alpha^2 .* (sum(ci .* ai .* mi.^2)).^2) .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_arr(w,ai, mi, ci), -Inf, Inf, 'ArrayValued', true));
                                end
                            end
        
                            %%%%%%%%%%%%%%%%% Optimize pulse shaping coefficients %%%%%%%%%%%%%%%%%
                            total_opt_start = tic;
                            disp("Begin optimization...")
                            for strat_idx = 1:num_strats
                                strat = all_strats(strat_idx);
                                split_str = split(strat);
                                constraint = split_str(1);
    
                                % fprintf("Optimizing coeffs. for " + strat + ":");
                                disp("Optimizing " + strat + " | obj = " + obj_func)
        
                                % Baseline estimation without optimal coefficients
                                if contains(strat, "none")
                                    opt_coeffs(:,1:S,:,:) = ones(size(opt_coeffs(:,1:S,:,:)));
                                    opt_coeffs(:,S+1:end,:,:) = zeros(size(opt_coeffs(:,S+1:end,:,:)));
                                else
                                    % Estimation with coefficients
                                    if scheme == "EPC"
                                        if contains(strat, "indiv. bottom clipping")
                                            opt_ai = 1;
                                            opt_bi = Gi - sqrt(Pi_max);

                                        %%% INDIVIDUAL ENERGY CONSTRAINT
                                        elseif contains(strat, "indiv. homog. linear prop. scaling") || contains(strat, "indiv. hetero. linear opt.")
                                            opt_ai = min(Ai_linear, [], 2);
                                            opt_bi = zeros(size(opt_ai));

                                        elseif contains(strat, "indiv. homog. affine prop. scaling") || contains(strat, "indiv. hetero. affine opt.")
                                            opt_ai = min(Ai_affine, [], 2);
                                            opt_bi = opt_ai .* Mi ./ 2;

                                        elseif contains(strat, "indiv. hetero. linear prop. scaling")
                                            opt_ai = Ai_linear;
                                            opt_bi = zeros(size(opt_ai));

                                        elseif contains(strat, "indiv. hetero. affine prop. scaling")
                                            opt_ai = Ai_affine;
                                            opt_bi = opt_ai .* Mi ./ 2;

                                        elseif contains(strat, "indiv. hetero. linear asymp. approx.")
                                            opt_ai = Ai_linear;
                                            opt_bi = zeros(size(opt_ai));

                                        elseif contains(strat, "indiv. hetero. affine asymp. approx.")
                                            opt_ai = Ai_affine;
                                            opt_bi = opt_ai .* Mi ./ 2;
    
                                        %%% TOTAL ENERGY CONSTRAINT
                                        elseif contains(strat, "total homog. linear prop. scaling") || contains(strat, "total hetero. linear opt.")
                                            opt_ai = sqrt(P_max ./ sum(Ci_linear,2));
                                            opt_bi = zeros(size(opt_ai));
            
                                        elseif contains(strat, "total homog. affine prop. scaling") || contains(strat, "total hetero. affine opt.")
                                            opt_ai = sqrt(P_max ./ sum(Ci_affine, 2));
                                            opt_bi = opt_ai .* Mi ./ 2;
            
                                        elseif contains(strat, "total hetero. linear asymp. approx.")
                                            opt_ai = sqrt(P_max ./ sum( (ci.^2 .* mi.^4) ./ Ci_linear, 2)) .* (ci .* mi.^2) ./ Ci_linear;
                                            opt_bi = zeros(size(opt_ai));
            
                                        elseif contains(strat, "total hetero. affine asymp. approx.")
                                            opt_ai = sqrt(P_max ./ sum( (ci.^2 .* mi.^4) ./ Ci_affine, 2)) .* (ci .* mi.^2) ./ Ci_affine;
                                            opt_bi = opt_ai .* Mi ./ 2;
            
                                        %%% MIXED ENERGY CONSTRAINT
                                        elseif contains(strat, "mixed homog. linear prop. scaling") || contains(strat, "mixed hetero. linear opt.")
                                            indiv_cond = min(Ai_linear, [], 2);
                                            sum_cond = sqrt(P_max ./ sum(Ci_linear,2));
            
                                            opt_ai = min([indiv_cond sum_cond], [], 2);
                                            opt_bi = zeros(size(opt_ai));
    
                                        elseif contains(strat, "mixed homog. affine prop. scaling") || contains(strat, "mixed hetero. affine opt.")
                                            indiv_cond = min(Ai_affine, [], 2);
                                            sum_cond = sqrt(P_max ./ sum(Ci_affine, 2));
            
                                            opt_ai = min([indiv_cond sum_cond], [], 2);
                                            opt_bi = opt_ai .* Mi ./ 2;
    
                                        elseif contains(strat, "mixed") && contains(strat, "hetero.") && contains(strat,"asymp. approx.")
                                            % Initialize ai matrix
                                            opt_ai = zeros(1,S,1,nDeployments);
                                            opt_bi = zeros(1,S,1,nDeployments);
    
                                            % Run active-set approach for each deployment
                                            active_set = ones(S,nDeployments);
                                            saturated_set = zeros(S,nDeployments);
                                            for dep = 1:nDeployments
                                                dep_ci = ci(:,:,:,dep);
                                                dep_mi = mi(:,:,:,dep);
                                                dep_Mi = Mi(:,:,:,dep);
                                                % Select deployment Ci
                                                if contains(strat, "linear")
                                                    dep_Ci = Ci_linear(:,:,:,dep);
                                                elseif contains(strat, "affine")
                                                    dep_Ci = Ci_affine(:,:,:,dep);
                                                else
                                                    error("Unknown strategy specified!")
                                                end
    
                                                prev_dep_active_set = [];
                                                while true
                                                    dep_active_set_idxs = find(active_set(:,dep));
                                                    dep_sat_set_idxs = find(saturated_set(:,dep));
        
                                                    active_ci = dep_ci(dep_active_set_idxs);
                                                    active_mi = dep_mi(dep_active_set_idxs);
                                                    active_Ci = dep_Ci(dep_active_set_idxs);
    
                                                    nu_num = sum((active_ci .* active_mi.^2).^2 ./ active_Ci, 2);
                                                    nu_denom = P_max - Pi_max*numel(dep_sat_set_idxs); % Pi_max is a scalar here
                                                    nu = 0.5 * sqrt(nu_num ./ nu_denom); % nu should be a scalar
                                                    
                                                    active_cond = ge((active_ci .* active_mi.^2) ./ (2*nu.*active_Ci), sqrt(Pi_max ./ active_Ci));
    
                                                    % Update active and saturated sets
                                                    for s_idx = 1:numel(active_cond)
                                                        if active_cond(s_idx)
                                                            i = dep_active_set_idxs(s_idx);
                                                            active_set(i,dep) = 0;
                                                            saturated_set(i,dep) = 1;
                                                        end
                                                    end
    
                                                    % store active set for comparison
                                                    if isequal(active_set(:,dep), prev_dep_active_set)
                                                        % disp(active_set(:,dep).')
                                                        break
                                                    else
                                                        prev_dep_active_set = active_set(:,dep);
                                                    end
                                                end
    
                                                for ii = 1:S
                                                    if active_set(ii,dep)
                                                        opt_ai(1,ii,1,dep) = (dep_ci(ii) * dep_mi(ii)^2) / (2*nu * dep_Ci(ii));
                                                    else
                                                        opt_ai(1,ii,1,dep) = sqrt(Pi_max / dep_Ci(ii));
                                                    end
                                                    
                                                    if contains(strat, "affine")
                                                        opt_bi(1,ii,1,dep) = opt_ai(1,ii,1,dep) .* dep_Mi(ii) / 2;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                    
                                    % Store coeffs if using asymp. approx
                                    if ~contains(strat, "opt.") % contains(strat, "asymp. approx")
                                        opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                                        opt_coeffs(strat_idx,S+1:end,:,:) = opt_bi .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                                    
                                        % Run fminsearch
                                    elseif contains(strat, "opt.")
                                        % Define parameters for fminsearch
                                        options = optimoptions('fmincon','Display','off', ...
                                            'Algorithm','sqp');
                                        opt_coeffs_cell = cell(1, nDeployments);                                    
                                        opt_starting_coeffs = [opt_ai.*ones(1,S,1,nDeployments) opt_bi.*ones(1,S,1,nDeployments)];
    
                                        is_linear = contains(strat, "linear");
                                        is_affine = contains(strat, "affine");
    
                                        % Run fmincon using parallelization
                                        parfor dep_idx = 1:nDeployments
                                            x0 = opt_starting_coeffs(:,1:S,:,dep_idx);
                                            % x0 = ones(1,S);

                                            dep_ci = ci(:,:,:,dep_idx);
                                            dep_mi = mi(:,:,:,dep_idx);
                                            dep_Mi = Mi(:,:,:,dep_idx);
                                            dep_Gamma_i = Gamma_i(:,:,:,dep_idx);
    
                                            if isscalar(Pi_max)
                                                dep_Pi_max = Pi_max;
                                                dep_P_max = P_max;
                                            else
                                                dep_Pi_max = Pi_max(:,:,:,dep_idx);
                                                dep_P_max = P_max(:,:,:,dep_idx);
                                            end
    
                                            lb = zeros(size(x0)); % ai >= 0
                                            ub = []; % Upper bound is unrestricted
    
                                            local_coeffs = zeros(1,2*S);
                                            nonlcon = @(x) fmincon_constraints(x, dep_Mi, dep_Gamma_i, dep_Pi_max, dep_P_max, strat);
                                            
                                            switch obj_func
                                                case "crlb sum"
                                                    obj_form = @(x, mi, ci) alpha_objective_array(x, mi, ci) + ...
                                                        t0_weight .* t0_objective_array(x, mi, ci, 1);
                                                case "fim trace"
                                                    obj_form = @(x, mi, ci) -1 * (1/alpha_objective_array(x, mi, ci) + ...
                                                        1/t0_objective_array(x, mi, ci, 1));
                                                case "fim det"
                                                    obj_form = @(x, mi, ci) -1/(alpha_objective_array(x, mi, ci) * ...
                                                        t0_objective_array(x, mi, ci, 1));
                                                otherwise
                                                    error("Unknown objective function specified!")
                                            end
                                            fmincon_obj = @(x) obj_form(x, dep_mi, dep_ci);
    
                                            [optimal_x, ~] = fmincon(fmincon_obj, x0, [], [], [], [], lb, ub, nonlcon, options);
    
                                            % Assign ai
                                            local_coeffs(1:S) = optimal_x;
    
                                            % Assign bi
                                            if is_linear
                                                local_coeffs(S+1:end) = 0;
                                            elseif is_affine
                                                local_coeffs(S+1:end) = local_coeffs(1:S) .* dep_Mi ./ 2;
                                            end
                                            opt_coeffs_cell{dep_idx} = local_coeffs;
                                        end
                                        % Recombine after parfor
                                        for dep_idx = 1:nDeployments
                                            opt_coeffs(strat_idx, :, :, dep_idx) = opt_coeffs_cell{dep_idx};
                                        end
                                    else
                                        error("Unknown strategy specified!")
                                    end
                                end
                            end
                            total_opt_time = toc(total_opt_start);
                            disp("Total optimization time: " + floor(total_opt_time/60) + " minutes " + mod(total_opt_time,60) + " seconds")
                            fprintf("\n")
        
                            %%%%%%%%%%%%%%%%% ESTIMATION %%%%%%%%%%%%%%%%%
                            total_est_start = tic;
    
                            % Compute ui here so we don't have to recompute FFT for each strat
                            [~, ui] = mf_integral_fft(noisy_unified_xi, mi .* sensor_signal(t-tpi, Tp, norm_fact), 1, 1, K, dt, Tp);
    
                            disp("Begin ML estimation...")
                            for strat_idx = 1:num_strats
                                strat = all_strats(strat_idx);
                                disp("Running ML estimation for " + strat + " | obj = " + obj_func)
        
                                ai = opt_coeffs(strat_idx, 1:S, :, :);
                                bi = opt_coeffs(strat_idx, S+1:end, :, :);    
    
                                if contains(strat, "bottom clipping")
                                    ui_clipping = max(ui-bi, -sqrt(Pi_max));
                                    out_cws = sum(ci .* ui_clipping, 2);
                                    clipping_check = reshape((Mi/2 + Gamma_i) > sqrt(Pi_max), 1, []);
                                else
                                    out_cws = sum(ci .* (ai .* ui - bi), 2);
                                end
    
                                cws = pagetranspose(out_cws);
                                y = cws + pagetranspose(scaled_n(:,1,:,:));
        
                                if contains(experiment, "cwe") || experiment == "random_dropout"
                                    % Compute AC-CWE estimates for full phase compensation.
                                    if scheme == "MPC" || scheme == "EPC"
                                        mag_sqr_H = H_arr(omega, ai, mi, ci);
        
                                        % Get time domain matrix for Qn.
                                        [~, Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments);
    
                                        if contains(strat, "bottom clipping") && any(clipping_check) % sensor signals have been clipped so revert to naive peak search
                                            [max_y_vals,I] = max(abs(real(y)), [], 2);
                                        else
                                            Omega_2_t0_search = reshape(sum(ui_t0_search .* ci .* ai, 2), K, 1, K, 1, nDeployments);
                                            mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                            [max_y_vals,I] = max(mf_with_y, [], 3);
                                        end

                                        t0_estimates_for_plot = reshape((I-1)*dt, 1, 1, nTrials, nDeployments);
        
                                        if contains(experiment, "disjoint")
                                            t0_estimates_for_alpha = t0_true*ones(1,1,nTrials,nDeployments);
                                        else
                                            t0_estimates_for_alpha = t0_estimates_for_plot;
                                        end
    
                                        %%% Estimate alpha
                                        Af = fft(mi .* sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), N, 1);
                                        lag0 = round(Tp/dt);
                                        Y_tensor =  dt*ifft(Af .* mfTemplateFFT, [], 1);
                                        ui_tensor = Y_tensor(lag0+1 : (lag0+K), :, :, :);
                                        Omega = sum(ui_tensor .* ci .* ai, 2);
                                        
                                        resh_Qn = reshape(Qn_matrix,K,K,1,nDeployments);
                                        resh_Omega = reshape(Omega,1,K,nTrials,nDeployments);
    
                                        num = dt*dt*pagemtimes(pagemtimes(real(y) + sum(ci .* bi, 2),resh_Qn),pagetranspose(resh_Omega));
                                        denom = dt*dt*pagemtimes(pagemtimes(resh_Omega,resh_Qn),pagetranspose(resh_Omega));
                                    end
                                    alpha_estimates = num ./ denom;
                                end
        
                                % Compute empirical variance.
                                rho_empirical_var(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,pivot_idx,save_dim,:) = var(alpha_estimates,0,3);
                                rho_empirical_var(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,pivot_idx,save_dim,:) = var(t0_estimates_for_plot,0,3);
        
                                % Compute empirical mse.
                                rho_empirical_mse(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,pivot_idx,save_dim,:) = mean((alpha_estimates - alpha_true).^2,3);
                                rho_empirical_mse(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,pivot_idx,save_dim,:) = mean((t0_estimates_for_plot - t0_true).^2,3);
        
                                % Compute empirical bias.
                                rho_empirical_bias(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,pivot_idx,save_dim,:) = mean(alpha_estimates,3);
                                rho_empirical_bias(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,pivot_idx,save_dim,:) = mean(t0_estimates_for_plot,3);
        
                                if scheme == "EPC"
                                    rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,pivot_idx,save_dim,:) = alpha_objective_array(unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci);
                                    rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,pivot_idx,save_dim,:) = t0_objective_array(unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci, alpha_true);
                                end
                            end
                            total_est_time = toc(total_est_start);
                            disp("Total ML estimation time: " + floor(total_est_time/60) + " minutes " + mod(total_est_time,60) + " seconds")
                        end
                        total_channel_db_time = toc(channel_db_start);
                        disp("Total runtime for channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB: " + floor(total_channel_db_time/60) + " minutes " + mod(total_channel_db_time,60) + " seconds")
                        fprintf("\n");
                    end
                end
            end
        end
    end
    
    % Stop run timer.
    runtime = toc(loopTic);
    % Display total run time.
    disp(experiment + " took " + floor(runtime/60) + " minutes " + mod(runtime,60) + " seconds")
    
    %% Compute averages across deployments.
    if nDeployments == 1
        avg_dep_var = rho_empirical_var;
        avg_dep_mse = rho_empirical_mse;
        avg_dep_bias = rho_empirical_bias;
        avg_dep_crlb = rho_crlb;
    else
        avg_dep_var = mean(rho_empirical_var,length(size(rho_empirical_var)));
        avg_dep_mse = mean(rho_empirical_mse,length(size(rho_empirical_mse)));
        avg_dep_crlb = mean(rho_crlb,length(size(rho_crlb)));
    
        % Placeholder for normalization, but the metrics may be normalized in the plotting scripts as well
        
        % avg_dep_var(:,:,:,1,:,:) = avg_dep_var(:,:,:,1,:,:);
        % avg_dep_mse(:,:,:,1,:,:) = avg_dep_mse(:,:,:,1,:,:);
        % avg_dep_crlb(:,:,:,1,:,:) = avg_dep_crlb(:,:,:,1,:,:);
        % 
        % % Normalize time delay metrics
        % avg_dep_var(:,:,:,2,:,:) = avg_dep_var(:,:,:,2,:,:);
        % avg_dep_mse(:,:,:,2,:,:) = avg_dep_mse(:,:,:,2,:,:);
        % avg_dep_crlb(:,:,:,2,:,:) = avg_dep_crlb(:,:,:,2,:,:);

        avg_dep_bias = mean(rho_empirical_bias,length(size(rho_empirical_bias)));
    end
    
    %% Save experiment results
    save_images = false;
    
    if ~any(experiment == experiment_list)
        disp("Unknown experiment specified. Results not saved!")
    else
        files_to_save = ["avg_dep_var", "avg_dep_mse", "avg_dep_bias", "avg_dep_crlb",...
            "agent_db_values", "channel_snr", "selected_schemes", "dropout_vals", "sensor_vals", "channel_db_values",...
            "experiment", "sensor_dimension", "save_images", "K", "nDeployments",...
            "constraints", "transforms", "coeff_types", "approaches", "iter_arr", "iter_length", "all_strats", "num_strats", "opt_coeffs", "save_images",...
            "T0", "Tp", "alpha_true", "t0_true", "ci", "clean_y_arr", "rho_vals", "Mi", "Gamma_i", "Pi_max", "pivot_vals"];
        save(experiment + "_results.mat", files_to_save{:})
    end
end

%% Functions
function [c, ceq] = fmincon_constraints(x, Mi, Gamma_i, Pi_max, P_max, strat)
    ai = x;
    
    if contains(strat, "linear")
        indiv_peak_power_vals = (ai .* (Mi + Gamma_i)).^2;
    else
        indiv_peak_power_vals = (ai .* (Mi./2 + Gamma_i)).^2;
    end
    
    total_cond = sum(indiv_peak_power_vals) - P_max;
    indiv_upper_cond = (indiv_peak_power_vals - Pi_max)';
    
    if contains(strat, "total")
        c = total_cond;
    elseif contains(strat, "indiv.")
        c = indiv_upper_cond;
    elseif contains(strat, "mixed")
        c = [total_cond; indiv_upper_cond];
    else
        error("Unknown experiment specified!")
    end
    
    ceq = [];
end