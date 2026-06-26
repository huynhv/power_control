% Clear all variables and close all existing figures.
clear all
close all

% check if parpool instance is already running
pool = gcp('nocreate');
% if not, then start one
if isempty(pool)
    parpool('local');
end

% Choose which experiment to run.
experiment_list = ["cwe_disjoint_peak_power_opt"; 
                   "cwe_disjoint_peak_power_opt_diff_rho"; 
                   "cwe_disjoint_peak_power_opt_bottom_clipping_nonclipped";
                   "cwe_disjoint_peak_power_opt_bottom_clipping_clipped"; 
                   "cwe_disjoint_peak_power_opt_diff_weights";
                   "cwe_disjoint_peak_power_opt_diff_bounds_on_alpha"; 
                   "cwe_disjoint_peak_power_opt_diff_obj_functions";
                   "cwe_disjoint_peak_power_opt_asymp_approx"];

experiment = "cwe_disjoint_peak_power_opt";

msg = ['Running ' char(experiment)];
msg_len = strlength(msg);
border = repmat('─', 1, ceil(msg_len/1.5) + 2);
fprintf('\n╭%s╮\n', border);
fprintf('│  %s  │\n', msg);
fprintf('╰%s╯\n\n', border);

% Set seed for reproducibility
seed = 264;
disp(seed);
rng(seed);

nDeployments = 5;
% Set the number of measurements per sensor, starting from 0
K = 251;
% Set number of trials.
nTrials = 500;

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
func = @(t) abs(sensor_signal(t)).^2;
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

%%% Always generate w, n, and channel gain for S = 10, then use subsets for different S trials
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
alpha_weight_vals = 0.5;
D_val = 0.5;
norm_fact = 1;
Gamma = 1;

% Configure strategies
agent_db_values = 0;
if contains(experiment, "diff_rho")
    % Define channel snr values.
    channel_snr = 15; %%% TODO: need to make this dynamic
    rho_vals = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];

    pivot_vals = rho_vals;

    constraints = ["total"]; % "total", "indiv.", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.","hetero."];  % "homog.", "hetero."
    approaches = ["prop. scaling", "asymp. approx.", "opt."]; % "asymp. approx.", "opt."

elseif contains(experiment, "bottom_clipping_nonclipped")
    channel_snr = linspace(0,18,5);
    rho_vals = 0.5;

    pivot_vals = rho_vals;
    
    constraints = ["indiv."]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["prop. scaling", "asymp. approx.", "opt."]; % "asymp. approx.", "opt."

elseif contains(experiment, "bottom_clipping_clipped")
    channel_snr = linspace(0,18,5);
    rho_vals = 0.1;

    pivot_vals = rho_vals;
    
    constraints = ["indiv."]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["prop. scaling", "asymp. approx.", "opt."]; % "asymp. approx.", "opt."

elseif contains(experiment, "diff_weights")
    channel_snr = linspace(0,18,5);
    rho_vals = 0.5;
    weight_vals = [1, 1e1, 1e2, 1e3, 1e4];

    pivot_vals = weight_vals;

    constraints = ["indiv.", "total", "mixed"]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["prop. scaling", "opt."]; % "asymp. approx.", "opt."

elseif contains(experiment, "diff_bounds_on_alpha")
    channel_snr = linspace(0,18,5);
    rho_vals = 0.5;
    weight_vals = 1;
    Xi_vals = [1, 2, 5, 10];

    pivot_vals = Xi_vals;

    constraints = ["indiv.", "total", "mixed"]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["prop. scaling", "opt."]; % "asymp. approx.", "opt."

elseif contains(experiment, "asymp_approx")
    channel_snr = 0;
    rho_vals = 0.5;
    weight_vals = 1;
    Xi_vals = 1;
    agent_db_values = linspace(0,20,5);

    pivot_vals = rho_vals;

    constraints = ["indiv.", "total", "mixed"]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["asymp. approx.", "prop. scaling", "opt."]; % "asymp. approx.", "opt."

elseif contains(experiment, "diff_obj_functions")
    channel_snr = linspace(0,18,5);
    rho_vals = 0.5;
    weight_vals = 1;
    Xi_vals = 1;

    obj_function_vals = ["crlb sum", "fim trace", "fim det"];

    pivot_vals = obj_function_vals;

    constraints = ["indiv.", "total", "mixed"]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["opt."]; % "prop. scaling", "asymp. approx.", "opt."

else
    channel_snr = 18; % linspace(0,18,5);
    rho_vals = 0.5;

    pivot_vals = rho_vals;
    
    constraints = ["indiv.", "total", "mixed"]; % "indiv.", "total", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["prop. scaling", "opt."]; % "prop. scaling", "asymp. approx.", "opt."
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

                % do not include homog. asymp. approx in strats
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
            Xi = 1;
            obj_func = "crlb sum";
        elseif contains(experiment, "diff_bounds_on_alpha")
            rho = rho_vals(1);
            t0_weight = 1;
            Xi = Xi_vals(pivot_idx);
            obj_func = "crlb sum";
        elseif contains(experiment, "diff_obj_functions")
            rho = rho_vals(1);
            t0_weight = 1;
            Xi = 1;
            obj_func = obj_function_vals(pivot_idx);
        else
            rho = rho_vals(pivot_idx);
            t0_weight = 1;
            Xi = 1;
            obj_func = "crlb sum";
        end
        % rho = rho_vals(pivot_idx);
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

                E0 = sum(util_functions.sensor_signal(t-t0_true, Tp, norm_fact).^2 .* dt, 1);
                % Ei = mi.^2 .* E0;

                Gamma_i = gamma_w .* mi.^2 .* E0;
    
                noisy_unified_xi = alpha_true .* (mi .* util_functions.sensor_signal(t-t0_true, Tp, norm_fact)) + scaled_w(:,1:S,:,:);

                disp("Computing peak-power values...")
                
                Mi_true = alpha_true .* mi.^2 .* E0;
                Gi = Mi_true + Gamma_i;

                P0_max = mean((Gi).^2, "all");
                Pi_max = rho * P0_max; % rho * (Mi + Gamma_i).^2;
                P_max = D_val * S*Pi_max;

                alpha_max = Xi * alpha_true;
                Mi = alpha_max .* mi.^2 .* E0;
                Ci_linear = (Mi + Gamma_i).^2;
                Ci_affine = (Mi./2 + Gamma_i).^2;

                Ai_linear = sqrt(Pi_max ./ Ci_linear);
                Ai_affine = sqrt(Pi_max ./ Ci_affine);

                % Define search space for t0
                t0_search_space = repmat(reshape(t,1,1,[]), 1,1,1,nDeployments);
                [~, ui_t0_search] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), 1, 1, K, dt, Tp);
                % Precompute FFT for matched filter template
                mfTemplate = mi .* util_functions.sensor_signal(t, Tp, norm_fact);
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
                        mag_sqr_S0 = @(w) util_functions.zeroIfnan(mag_sqr_S0_internal(w)) + (w == Bee | w == -Bee) .* (mag_sqr_S0_internal(w-epsilon) + mag_sqr_S0_internal(w+epsilon))./2;
                        % mag_sqr_SHdI_anon = @(w) 1 ./ (Gamma + mag_sqr_S0(w));
    
                        %%%%%%%%%%%%%%%%% Define Anonymous Expressions %%%%%%%%%%%%%%%%%
                        if contains(experiment, "cwe") || experiment == "random_dropout"
                            if (scheme == "MPC" || scheme == "EPC")
                                V_arr_internal = @(w, ai, mi, ci) scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0_internal(w) + scaled_n_psd_constant/2;
                                V_arr = @(w, ai, mi, ci) util_functions.zeroIfnan(V_arr_internal(w, ai, mi, ci)) + (w == Bee | w == -Bee) .* (V_arr_internal(w-epsilon, ai, mi, ci) + V_arr_internal(w+epsilon, ai, mi, ci))./2;
    
                                H_arr_internal = @(w, ai, mi, ci) (1 + epsilon) ./ (V_arr_internal(w, ai, mi, ci) + epsilon);
                                H_arr = @(w, ai, mi, ci) util_functions.zeroIfnan(H_arr_internal(w, ai, mi, ci)) + (w == Bee | w == -Bee) .* (H_arr_internal(w-epsilon, ai, mi, ci) + H_arr_internal(w+epsilon, ai, mi, ci))./2;
    
                                alpha_objective_array = @(ai, mi, ci) (2*pi) ./( ((sum(ci .* ai .* mi.^2)).^2) .* integral(@(w) (mag_sqr_S0(w)).^2 / V_arr(w, ai, mi, ci), -Inf, Inf, 'ArrayValued', true));
                                t0_objective_array = @(ai, mi, ci, arb_alpha) (2*pi) ./( (arb_alpha^2 .* (sum(ci .* ai .* mi.^2)).^2) .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_arr(w,ai, mi, ci), -Inf, Inf, 'ArrayValued', true));
                            end
                        end
    
                        %%%%%%%%%%%%%%%%% Optimize pulse shaping coefficients %%%%%%%%%%%%%%%%%
                        total_opt_start = tic;
                        for strat_idx = 1:num_strats
                            strat = all_strats(strat_idx);
                            split_str = split(strat);
                            constraint = split_str(1);

                            % fprintf("Optimizing coeffs. for " + strat + ":");
                            disp("Optimizing " + strat + " | obj = " + obj_func)
    
                            % baseline estimation without optimal coefficients
                            if contains(strat, "none")
                                opt_coeffs(:,1:S,:,:) = ones(size(opt_coeffs(:,1:S,:,:)));
                                opt_coeffs(:,S+1:end,:,:) = zeros(size(opt_coeffs(:,S+1:end,:,:)));
                            else
                                % estimation with coefficients
                                if scheme == "EPC"
                                    if contains(strat, "indiv. bottom clipping")
                                        opt_ai = 1;
                                        % solution using quadratic equation, could explore what if using negative root instead
                                        opt_bi = Gi - sqrt(Pi_max);
                                        % opt_bi = (sum(Gi,2) + sqrt(sum(Gi,2).^2 - S*(sum(Gi.^2,2)-P_max))) ./ S; 

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
                                        % [Gamma_linear, ~] = min(ci .* Ai_linear, [], 2);
                                        opt_ai = Ai_linear;
                                        opt_bi = zeros(size(opt_ai));
        
                                    elseif contains(strat, "indiv. hetero. affine asymp. approx.")
                                        % [Gamma_affine, ~] = min(ci .* Ai_affine, [], 2);
                                        opt_ai = Ai_affine;
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
                                        % initialize ai matrix
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
                                                    disp(active_set(:,dep).')
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
                                    %%% end mixed energy constraint
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
        
                                    % run fmincon using parallelization
                                    parfor dep_idx = 1:nDeployments
                                        % x0 = opt_starting_coeffs(:,:,:,dep_idx);
                                        x0 = ones(1,S);

                                        % x0 only depends on ai, bi values are just placeholders here
                                        % use asymp. approx. as starting point (where applicable)
                                        
                                        % if contains(strat, "hetero. linear")
                                        %     x0 = [opt_ai(:,:,:,dep_idx) zeros(1,S)];
                                        %     % x0 = [ones(1,S) zeros(1,S)];
                                        % elseif contains(strat, "hetero. affine")
                                        %     x0 = [opt_ai(:,:,:,dep_idx) ones(1,S)];
                                        %     % x0 = [ones(1,2*S)];
                                        % else
                                        %     error("Unknown optimal coefficients requested!")
                                        % end

                                        % if contains(strat, "homog. linear")
                                        %     x0 = [1 0];
                                        % elseif contains(strat, "homog. affine")
                                        %     x0 = [1 1];
                                        % elseif contains(strat, "hetero. linear")
                                        %     x0 = [ones(1,S) zeros(1,S)];
                                        % elseif contains(strat, "hetero. affine")
                                        %     x0 = [ones(1,2*S)];  
                                        % else
                                        %     error("Unknown optimal coefficients requested!")
                                        % end
        
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
                                        
                                        lb = zeros(size(x0)); % ai >= 0, bi >= 0
                                        ub = []; % unrestricted
        
                                        local_coeffs = zeros(1,2*S);
                                        nonlcon = @(x) fmincon_constraints(x, dep_Mi, dep_Gamma_i, dep_Pi_max, dep_P_max, strat);
                                        
                                        if obj_func == "crlb sum"
                                            % objective is already in form of 1/lambda_alpha, 1/lambda_t0
                                            fmincon_obj = @(x) alpha_objective_array(x, dep_mi, dep_ci) + t0_weight .* t0_objective_array(x, dep_mi, dep_ci, 1);
                                        elseif obj_func == "fim trace"
                                            fmincon_obj = @(x) -1* (1/alpha_objective_array(x, dep_mi, dep_ci) + 1/t0_objective_array(x, dep_mi, dep_ci, 1));
                                        elseif obj_func == "fim det"
                                            fmincon_obj = @(x) -1/(alpha_objective_array(x, dep_mi, dep_ci) * t0_objective_array(x, dep_mi, dep_ci, 1));
                                        else
                                            error("Unknown objective function specified!")
                                        end
                                        
                                        [optimal_x, ~] = fmincon(fmincon_obj, x0, [], [], [], [], lb, ub, nonlcon, options);
    
                                        % assign ai
                                        local_coeffs(1:S) = optimal_x(1:S);

                                        % assign bi
                                        % local_coeffs(S+1:end) = optimal_x((length(optimal_x)/2)+1:end);
                                        if contains(strat, "linear")
                                            local_coeffs(S+1:end) = 0;
                                        elseif contains(strat, "affine")
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
                        [~, ui] = mf_integral_fft(noisy_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), 1, 1, K, dt, Tp);

                        for strat_idx = 1:num_strats
                            strat = all_strats(strat_idx);
                            disp("Running ML estimation for " + strat);
    
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
    
                                    % estimate t0
                                    % if contains(strat, "clip")
                                    %     [~, temp_for_t0] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    %     Omega_2_t0_search= sum(ci .* temp_for_t0 .* clipping_window, 2);
                                    %     Omega_2_t0_search = reshape(Omega_2_t0_search - clipping_window .* sum(ci .* bi, 2), K, 1, K, 1, nDeployments);
                                    %     mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + clipping_window.' .* sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                    %     % [max_y_vals,I] = max(real(y),[],2);
                                    % else
                                    %     [Omega_2_t0_search, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    %     Omega_2_t0_search = reshape(Omega_2_t0_search - sum(ci .* bi, 2), K, 1, K, 1, nDeployments);
                                    %     mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                    %     [max_y_vals,I] = max(mf_with_y, [], 3);
                                    % end

                                    if contains(strat, "bottom clipping") && any(clipping_check) % sensor signals have been clipped so revert to naive peak search
                                        [max_y_vals,I] = max(abs(real(y)), [], 2);
                                    else
                                        Omega_2_t0_search = reshape(sum(ui_t0_search .* ci .* ai, 2), K, 1, K, 1, nDeployments);
                                        mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                        [max_y_vals,I] = max(mf_with_y, [], 3);
                                    end

                                    % [Omega_2_t0_search, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    % Omega_2_t0_search = reshape(Omega_2_t0_search, K, 1, K, 1, nDeployments);
                                    % mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                    % [max_y_vals,I] = max(mf_with_y, [], 3);

                                    t0_estimates_for_plot = reshape((I-1)*dt, 1, 1, nTrials, nDeployments);
    
                                    if contains(experiment, "disjoint")
                                        t0_estimates_for_alpha = t0_true*ones(1,1,nTrials,nDeployments);
                                    else
                                        t0_estimates_for_alpha = t0_estimates_for_plot;
                                    end

                                    %%% estimate alpha

                                    % Omega = matched_filter_integral(reshape(mi .* sensor_signal(t-ti-t0_estimates),K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-reshape(t,1,1,K)), K,S,K,1,nDeployments), ci, K, S, nTrials, nDeployments, dt);
                                    
                                    % if contains(strat, "clip")
                                    %     [~, temp_for_alpha] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    %     Omega = sum(ci .* temp_for_alpha .* clipping_window, 2);
                                    % 
                                    %     resh_Qn = reshape(Qn_matrix,K,K,1,nDeployments);
                                    %     resh_Omega = reshape(Omega,1,K,nTrials,nDeployments);
                                    % 
                                    %     num = dt*dt*pagemtimes(pagemtimes(real(y) + clipping_window.' .* sum(ci .* bi, 2),resh_Qn),pagetranspose(resh_Omega));
                                    %     denom = dt*dt*pagemtimes(pagemtimes(resh_Omega,resh_Qn),pagetranspose(resh_Omega));
                                    % else
                                    %     [Omega, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    %     resh_Qn = reshape(Qn_matrix,K,K,1,nDeployments);
                                    %     resh_Omega = reshape(Omega,1,K,nTrials,nDeployments);
                                    % 
                                    %     num = dt*dt*pagemtimes(pagemtimes(real(y) + sum(ci .* bi, 2),resh_Qn),pagetranspose(resh_Omega));
                                    %     denom = dt*dt*pagemtimes(pagemtimes(resh_Omega,resh_Qn),pagetranspose(resh_Omega));
                                    % end

                                    % Compute matched filter integral using FFT
                                    % [Omega, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    
                                    Af = fft(mi .* util_functions.sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), N, 1);
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
                                % if contains(strat, "clip")
                                %     % continue
                                %     [~, temp_for_crlb] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_true, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                %     Omega_for_crlb = sum(ci .* temp_for_crlb .* clipping_window, 2);
                                %     resh_Omega_for_crlb = reshape(Omega_for_crlb,1,K,1,nDeployments);
                                % 
                                %     temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-Tp)) + sin(w0*(t-ti-t0_true)).*-1.*(dirac(t-ti-t0_true) - dirac(t-ti-t0_true-Tp));
                                %     dst0 = reshape(squeeze(sum(ci .* reshape(clipping_window, 1,1,K) .* sum(alpha_true .* mi.^2 .* temp .* sensor_signal(t-ti-reshape(t,1,1,K)) * dt,1),2)),1,K,1,nDeployments);
                                % 
                                % 
                                %     rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = 1./(dt*dt*pagemtimes(pagemtimes(resh_Omega_for_crlb,resh_Qn),pagetranspose(resh_Omega_for_crlb)));
                                %     rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = 1./(dt*dt*pagemtimes(pagemtimes(dst0,resh_Qn),pagetranspose(dst0)));
                                % else
                                %     rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = alpha_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci);
                                %     rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = t0_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci, alpha_true);
                                % end
                                
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,pivot_idx,save_dim,:) = alpha_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci);
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,pivot_idx,save_dim,:) = t0_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci, alpha_true);
                            end
                        end
                        total_est_time = toc(total_est_start);
                        disp("Total ML estimation time: " + floor(total_est_time/60) + " minutes " + mod(total_est_time,60) + " seconds")
                        % fprintf('%s ML estimation time: %d minutes %d seconds\n', strat, floor(total_est_time/60), mod(total_est_time,60));
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

    % normalize magnitude metrics
    avg_dep_var(:,:,:,1,:,:) = avg_dep_var(:,:,:,1,:,:);
    avg_dep_mse(:,:,:,1,:,:) = avg_dep_mse(:,:,:,1,:,:);
    avg_dep_crlb(:,:,:,1,:,:) = avg_dep_crlb(:,:,:,1,:,:);

    % normalize time delay metrics
    avg_dep_var(:,:,:,2,:,:) = avg_dep_var(:,:,:,2,:,:);
    avg_dep_mse(:,:,:,2,:,:) = avg_dep_mse(:,:,:,2,:,:);
    avg_dep_crlb(:,:,:,2,:,:) = avg_dep_crlb(:,:,:,2,:,:);

    avg_dep_bias = mean(rho_empirical_bias,length(size(rho_empirical_bias)));
end

%% Save experiment results.
save_images = false;

if ~any(experiment == experiment_list)
    disp("Unknown experiment specified. Results not saved!")
else
    files_to_save = ["avg_dep_var", "avg_dep_mse", "avg_dep_bias", "avg_dep_crlb",...
        "agent_db_values", "channel_snr", "selected_schemes", "dropout_vals", "sensor_vals", "channel_db_values",...
        "experiment", "sensor_dimension", "save_images", "K", "nDeployments",...
        "constraints", "transforms", "coeff_types", "approaches", "iter_arr", "iter_length", "all_strats", "num_strats", "opt_coeffs", "alpha_weight_vals", "save_images",...
        "T0", "Tp", "alpha_true", "t0_true", "ci", "clean_y_arr", "rho_vals", "Mi", "Gamma_i", "Pi_max", "pivot_vals"];
    save(experiment + "_results.mat", files_to_save{:})
end

%% Plotting
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr([unique(iter_arr); "bottom clipping"]), colors(1:numel([unique(iter_arr); "bottom clipping"])));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Peak Power AirComp Results\";

exclude = all_strats(contains(all_strats, "hetero.") & contains(all_strats, "asymp. approx."));

% for agent_db_idx = 1:length(agent_db_values)
%     selected_strat_idxs = find((1:numel(all_strats)) .* ~ismember(all_strats,exclude).');
%     selected_strats = all_strats(selected_strat_idxs);
%     for param_idx = 1:2
%         % Collect all plot values for y-axis scaling.
%         all_vals = [avg_dep_mse(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension)];
%             %         avg_dep_crlb(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension);
%             %         avg_dep_var(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension);
%             %         avg_dep_mse(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension)
%             %         ];
% 
%         for scheme_idx = 1:length(selected_schemes)
%             scheme = selected_schemes(scheme_idx);
% 
%             % set(fig1,'Position',[0,0,1920,1080], 'Name', params_text(param_idx) + " hat | " + constraints(1) + " | alpha = " + alpha_true + ", t0 = " + t0_true + ", Tp = " + Tp + ", T0 = " + T0, 'NumberTitle', 'off')
%             % tiledlayout(1,length(constraints));
% 
%             count_vals = sensor_vals;
% 
%             for constraint_idx = 1:length(constraints)
%                 constraint = constraints(constraint_idx);
%                 data_idx_seq = selected_strat_idxs(find(contains(selected_strats, constraint)).');
%                 % Plot pseudoplots for legend symbols.
% 
%                 fig1 = figure;
%                 set(fig1,'Position',[100,100,450,450])
%                 ax = gca;
%                 hold(ax, 'on');
% 
%                 lgd = [];
% 
%                 for count_idx = data_idx_seq
%                     strat = all_strats(count_idx);
%                     split_str = split(strat);
%                     iter_key = join(split_str(2:end), " ");
% 
%                     lgd = [lgd iter_key];
% 
%                     plot(ax, nan, nan, 'color', colorMap(iter_key), 'LineWidth', 2);
%                 end
% 
%                 plot(ax, nan,nan,'x','color','black');
%                 % plot(ax, nan,nan,'^','color','black');
%                 % plot(ax, nan,nan,'o','color','black');
% 
%                 x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
%                 plot_line_width = 1.5;
% 
%                 for count_idx = 1:length(count_vals)
%                     for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
%                         strat = all_strats(strat_idx);
%                         split_str = split(strat);
%                         iter_key = join(split_str(2:end), " ");
% 
%                         plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
%                         % plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
%                         % plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
%                     end
%                 end
% 
%                 xlabel('Channel SNR (dB)')
%                 ylabel(" ")
% 
%                 title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
% 
%                 ax.FontSize = 15;          % font size
%                 ax.YScale = 'log';          % log scale if needed
%                 grid(ax, 'on');             % turn on grid
% 
%                 legend(ax, [lgd, "MSE"], 'Location', 'best')
% 
%                 grid on
% 
%                 y_lower = 10^(-0.25)*min(all_vals,[],"all");
%                 y_upper = 1.05*max(all_vals,[],"all");
%                 ylim([y_lower,y_upper]);
% 
%                 ax.XTick = channel_snr;
% 
%                 % Save the figure.
%                 if save_images
%                     save_folder = folder_path + "overlaid";
%                     if ~exist(save_folder, 'dir')
%                         mkdir(save_folder);
%                     end
%                     exportgraphics(fig1, fullfile(save_folder, sprintf("overlaid" + "_" + constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
%                 end
%             end
%         end
%     end
% end

%% Functions
function [c, ceq] = fmincon_constraints(x, Mi, Gamma_i, Pi_max, P_max, strat)
    ai = x;
    
    if contains(strat, "linear")
        indiv_peak_power_vals = (ai .* (Mi + Gamma_i)).^2;
    else
        % indiv_peak_power_vals = ai .* (Mi + Gamma_i) - bi;
        indiv_peak_power_vals = (ai .* (Mi./2 + Gamma_i)).^2;
    end
    
    total_cond = sum(indiv_peak_power_vals) - P_max;
    indiv_upper_cond = (indiv_peak_power_vals - Pi_max)';
    % indiv_lower_cond = (bi + ai .* Gamma_i - sqrt(Pi_max))';
    
    if contains(strat, "total")
        c = total_cond;
    elseif contains(strat, "indiv.")
        c = indiv_upper_cond; % , indiv_lower_cond];
    elseif contains(strat, "mixed")
        c = [total_cond; indiv_upper_cond];
    else
        error("Unknown experiment specified!")
    end
    
    ceq = [];
end