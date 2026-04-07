% Clear all variables and close all exisiting figures.
clear all
close all

% Choose which experiment to run.
experiment_list = ["cwe_disjoint_peak_power_opt", "cwe_disjoint_peak_power_opt_diff_rho"];
experiment = "cwe_disjoint_peak_power_opt";

msg = ['Running ' char(experiment)];
msg_len = strlength(msg);
border = repmat('─', 1, ceil(msg_len/1.5) + 2);
fprintf('\n╭%s╮\n', border);
fprintf('│  %s  │\n', msg);
fprintf('╰%s╯\n\n', border);

% Set seed for reproducibility
seed = 264;
rng(seed);

nDeployments = 10;
% Set the number of measurements per sensor, starting from 0
K = 301;
% Set number of trials.
nTrials = 600;

% Set number of sensors
sensor_vals = 5; % [5,10]
agent_db_values = 0;
dropout_vals = 0;
sensor_dimension = length(sensor_vals);

% Set observation window length in seconds.
T0 = 5;

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
% interioir fo the observation period).
func = @(t) abs(sensor_signal(t)).^2;
P_s = integral(func,0,T0)/T0;

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
% be in the interioir of the observation period.
t0_max = T0-Tp-max_ti-max_tpi;

if t0_true > t0_max
    error("True value of t0 exceeds maximum!")
end

%%% Always generate w, n, and channel gain for S = 10, then use subsets for different S trials
max_sensor_limit = 10;
% Generate sensor noise with dimensions: K x S x trials x deployments.
all_w = randn(K,max_sensor_limit,nTrials,nDeployments);
% Generate server noise.
n = (randn(K,max_sensor_limit,nTrials,nDeployments) + 1i*randn(K,max_sensor_limit,nTrials,nDeployments));
% Generate channel gains.
all_gi = (randn(1,max_sensor_limit,1,nDeployments) + 1i*randn(1,max_sensor_limit,1,nDeployments))/sqrt(2);

% Generate mi and ti.
all_mi = sort(unifrnd(min_mi,max_mi,1,max_sensor_limit,1,nDeployments),2,'descend');
all_ti = sort(unifrnd(min_ti,max_ti,1,max_sensor_limit,1,nDeployments),2,'ascend');

all_tpi = zeros(1,max_sensor_limit,1,nDeployments);

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
all_strats = [];

if contains(experiment, "diff_rho")
    % Define channel snr values.
    channel_snr = 15; %%% TODO: need to make this dynamic
    rho_vals = [0.5, 0.75, 1, 1.25, 1.5, 2];

    constraints = ["total"]; % "total", "indiv.", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.","hetero."];  % "homog.", "hetero."
    approaches = ["surr. approx", "fmincon"]; % "surr. approx.", "fmincon"
else
    channel_snr = linspace(0,15,5);
    rho_vals = 0.50;
    
    constraints = ["mixed"]; % "total", "indiv.", "mixed"
    transforms = ["linear","affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["surr. approx.", "fmincon"]; % "surr. approx.", "fmincon"
end

% constraints = ["none"]; % "total", "indiv.", "mixed"
% transforms = ["none"]; % "linear", "affine"
% coeff_types = ["none"];  % "homog.", "hetero."
% approaches = ["none"]; % "surr. approx.", "fmincon"
% 
% [sub_c, tr, ct, ap] = ndgrid(constraints, transforms, coeff_types, approaches);
% 
% iter_arr = ct + " " + tr + " " + ap;
% all_strats = sub_c + " " + iter_arr;
% all_strats = all_strats(:);   % flatten into list

iter_arr = [];
all_strats = [];

for i = 1:numel(constraints)
    for j = 1:numel(coeff_types)
        for k = 1:numel(transforms)
            for l = 1:numel(approaches)
                % do not include homog. surr. approx in strats
                % if coeff_types(j) == 'homog.' && approaches(l) == 'fmincon'
                %     continue

                % elseif contains(experiment, "diff_rho") && coeff_types(j) == 'hetero.' && approaches(l) == 'surr. approx'
                %     continue

                if true
                    if i == 1
                        iter_arr = [iter_arr; coeff_types(j) + " " + transforms(k) + " " + approaches(l)];
                    end
                    all_strats = [all_strats; constraints(i) + " " + coeff_types(j) + " " + transforms(k) + " " + approaches(l)];
                end 
            end
        end
    end
end

% if ~contains(experiment, "diff_rho")
%     all_strats = [all_strats; "clip"];
% end

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

% Start run timer
disp("Starting runtime...")
loopTic = tic;

for rho_idx = 1:length(rho_vals)
    rho = rho_vals(rho_idx);
    % Iterate through agent snr values.
    for agent_db_idx = 1:length(agent_db_values)
        agent_db = agent_db_values(agent_db_idx);
    
        % Set sensor noise power. Note the factor of 1/dt, which is equivalent to
        % applying an anti-aliasing filter.
        w_psd_constant = 1;
        gamma_w = (dt/w_psd_constant) * (P_s / db2magTen(agent_db));
        scaled_w = sqrt(gamma_w * w_psd_constant /dt) * all_w; % --> variance of this should be gamma_w/dt
        scaled_w_psd_constant = gamma_w * w_psd_constant; %% = Nw/2
    
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
    
                clean_unified_xi = alpha_true .* (mi .* util_functions.sensor_signal(t-t0_true, Tp, norm_fact));
                noisy_unified_xi = clean_unified_xi + scaled_w(:,1:S,:,:);

                E0 = sum(util_functions.sensor_signal(t-t0_true, Tp, norm_fact).^2 .* dt, 1);
                Ei = mi.^2 .* E0;

                Mi = alpha_true .* mi.^2 .* E0;
                Gamma_i = gamma_w .* mi.^2 .* E0;

                [~, clean_ui_1] = util_functions.mf_integral_fft(clean_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), 1, 1, K, dt, Tp);
                clean_ui = pagetranspose(clean_ui_1);

                [~, wi_tilde] = util_functions.mf_integral_fft(scaled_w(:,1:S,:,:), mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), 1, 1, K, dt, Tp);
    
                [~, ui_1] = util_functions.mf_integral_fft(noisy_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), 1, 1, K, dt, Tp);
                ui = pagetranspose(ui_1);

                E_uit = pagetranspose(sum(mean(ui,3),2)) * dt;
                E_uit_sqrd = pagetranspose(sum(mean(ui.^2,3),2)) * dt;
    
                %%% Compute energy expressions
                disp("Computing energy constraint expressions...")
    
                %%% vector energy expressions
                vect_Rs0 = pagemtimes(util_functions.sensor_signal(t-t0_true, Tp, norm_fact)', util_functions.sensor_signal(t-three_dim_t, Tp, norm_fact)) * dt;
                vect_int_Rs0 = sum(vect_Rs0) * dt;
                int_Rs0 = vect_int_Rs0;
    
                vect_int_Rs0_sqrd = sum(vect_Rs0.^2) * dt;
                int_Rs0_sqrd = vect_int_Rs0_sqrd;
    
                vect_gamma_i = sum(mi.^2 .* util_functions.sensor_signal(t-ti-three_dim_t, Tp, norm_fact).^2, 1) * dt;
                vect_int_gamma_i = sum(vect_gamma_i,3)*dt;
                int_gamma_i = vect_int_gamma_i;
    
                % omniscient assumption (hypothetical upper bound on energy)
                omni_C1i = alpha_true^2 .* mi.^4 .* int_Rs0_sqrd + scaled_w_psd_constant .* int_gamma_i; % E[ui(t)^2]
                omni_C2i = alpha_true .* mi.^2 .* int_Rs0; % E[ui(t)]
                omni_Ei_max = omni_C1i;
                omni_E_total = sum(omni_Ei_max,2);

                C1i = omni_C1i;
                C2i = omni_C2i;
                Ci_tilde = C1i - (C2i.^2)/T0;

                Ei_max = rho * C1i;
                E_total = D_val * sum(Ei_max,2);
                
                Pi_max = rho * (Mi + Gamma_i).^2;
                P_total = D_val * sum(Pi_max,2);

                Ci_linear = (Mi + Gamma_i).^2;
                Ci_affine = (Mi./2 + Gamma_i).^2;

                Ai_linear = sqrt(Pi_max ./ Ci_linear);
                Ai_affine = sqrt(Pi_max ./ Ci_affine);

                fundamental_E = rho*int_Rs0_sqrd;

                if any(contains(all_strats, "clip"))
                    s0 = @(tau) (heaviside(tau)-heaviside(tau-1)) .* sin(pi/Tp .* tau);
                    Rss = @(t) integral(@(u) s0(u-t0_true) .* s0(u-t), 0, T0, 'ArrayValued', true);
               
                    my_int_Rs0_sqr = integral(@(t) Rss(t).^2, 0, T0, 'ArrayValued', true);
                    
                    calc_energy = @(bias) integral(@(t) max(0, Rss(t)-bias).^2, 0, T0, 'ArrayValued', true);
                    saturation_obj = @(bias) calc_energy(bias) - fundamental_E;
    
                    L_optimal = fzero(saturation_obj, [0, calc_energy(0)]);
                    Li = pagetranspose(L_optimal .* alpha_true .* mi.^2);
    
                    clip_ui = max(ui-Li, 0);
    
                    zero_obj = @(t1) Rss(t1) - L_optimal;
                    my_t1 = fzero(zero_obj, [0,Tp]);
                    my_t2 = fzero(zero_obj, [1,2*Tp]);
    
                    [~,t1] = max(diff(sign(Rss(t)-L_optimal)));
                    t1 = dt*(t1+1); % we want to capture the region where the signal is positive
                    [~,t2] = min(diff(sign(Rss(t)-L_optimal)));
                    t2 = dt*t2;
    
                    clipping_window = (t >= my_t1 & t <= my_t2);
    
                    E_clip_uit_sqrd = pagetranspose(sum(mean(clip_ui.^2,3),2)) * dt;
                    E_clip_uit = pagetranspose(sum(mean(clip_ui,3),2)) * dt;
                    clip_energy_nae = mean(abs(Ei_max - E_clip_uit_sqrd) ./ Ei_max, "all");
                    fprintf("clip indiv. energy nae = %f\n", clip_energy_nae);
                end
                

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
                            gamma_n = (dt/n_psd_constant) * P_s * E_mi_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                        elseif scheme == "EPC"
                            ci = abs(gi);
                            gamma_n = (dt/n_psd_constant) * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                        elseif scheme == "PPC"
                            ci = qgi;
                            gamma_n = (dt/n_psd_constant) * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                        elseif scheme == "NPC"
                            ci = gi;
                            gamma_n = (dt/n_psd_constant) * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                        end
    
                        % Set server noise power. Note the factor of 1/dt which is
                        % equivalent to applying an anti-aliasing filter.
                        scaled_n = sqrt( (gamma_n*n_psd_constant/dt) / 2 ) * n;
                        scaled_n_psd_constant = gamma_n * n_psd_constant; % == N0/2
    
                        epsilon = 0.0001;
                        norm_fact = 1;
                        Bee = pi/Tp;
    
                        mag_sqr_S0_internal = @(w) (norm_fact^2) * (2*Bee.^2 .* (1 + cos(w.*pi./Bee)) ) ./ (w.^2 - Bee.^2).^2;
                        mag_sqr_S0 = @(w) util_functions.zeroIfnan(mag_sqr_S0_internal(w)) + (w == Bee | w == -Bee) .* (mag_sqr_S0_internal(w-epsilon) + mag_sqr_S0_internal(w+epsilon))./2;
                        mag_sqr_SHdI_anon = @(w) 1 ./ (Gamma + mag_sqr_S0(w));
    
                        %%%%%%%%%%%%%%%%% Define Anonymous Expressions %%%%%%%%%%%%%%%%%
                        if contains(experiment, "cwe") || experiment == "random_dropout"
                            if (scheme == "MPC" || scheme == "EPC")
                                V_arr_internal = @(w, ai, mi, ci) scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0_internal(w) + scaled_n_psd_constant/2;
                                V_arr = @(w, ai, mi, ci) util_functions.zeroIfnan(V_arr_internal(w, ai, mi, ci)) + (w == Bee | w == -Bee) .* (V_arr_internal(w-epsilon, ai, mi, ci) + V_arr_internal(w+epsilon, ai, mi, ci))./2;
    
                                H_arr_internal = @(w, ai, mi, ci) 1./ V_arr_internal(w, ai, mi, ci);
                                H_arr = @(w, ai, mi, ci) util_functions.zeroIfnan(H_arr_internal(w, ai, mi, ci)) + (w == Bee | w == -Bee) .* (H_arr_internal(w-epsilon, ai, mi, ci) + H_arr_internal(w+epsilon, ai, mi, ci))./2;
    
                                alpha_objective_array = @(ai, mi, ci) (2*pi) ./( ((sum(ci .* ai .* mi.^2)).^2) .* integral(@(w) (mag_sqr_S0(w)).^2 / V_arr(w, ai, mi, ci), -Inf, Inf, 'ArrayValued', true));
                                t0_objective_array = @(ai, mi, ci, arb_alpha) (2*pi) ./( (arb_alpha^2 .* (sum(ci .* ai .* mi.^2)).^2) .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_arr(w,ai, mi, ci), -Inf, Inf, 'ArrayValued', true));
                            end
                        end
    
                        %%%%%%%%%%%%%%%%% Optimize pulse shaping coefficients %%%%%%%%%%%%%%%%%
                        for strat_idx = 1:num_strats
                            strat = all_strats(strat_idx);
                            % fprintf("Optimizing coeffs. for " + strat + ":");
                            disp("Optimizing coeffs. for " + strat)
                            total_opt_start = tic;
    
                            % baseline estimation without optimal coefficients
                            if contains(strat, "none")
                                opt_coeffs(:,1:S,:,:) = ones(size(opt_coeffs(:,1:S,:,:)));
                                opt_coeffs(:,S+1:end,:,:) = zeros(size(opt_coeffs(:,S+1:end,:,:)));
                            else
                                % estimation with coefficients
                                if scheme == "EPC"
                                    if contains(strat, "clip")
                                        opt_ai = 1;
                                        opt_bi = 0;

                                    %%% TOTAL ENERGY CONSTRAINT
                                    elseif contains(strat, "total homog. linear")
                                        opt_ai = sqrt(P_total ./ sum(Ci_linear,2));
                                        opt_bi = 0;
        
                                    elseif contains(strat, "total homog. affine")
                                        opt_ai = sqrt(P_total ./ sum(Ci_affine, 2));
                                        opt_bi = opt_ai .* Mi ./ 2;
        
                                    elseif contains(strat, "total hetero. linear")
                                        opt_ai = (1./ci) .* sqrt(P_total ./ sum(Ci_linear ./ ci.^2, 2));
                                        opt_bi = 0;
        
                                    elseif contains(strat, "total hetero. affine")
                                        opt_ai = (1./ci) .* sqrt(P_total ./ sum(Ci_affine ./ ci.^2, 2));
                                        opt_bi = opt_ai .* Mi ./ 2;
        
                                    %%% INDIVIDUAL ENERGY CONSTRAINT
                                    elseif contains(strat, "indiv. homog. linear")
                                        opt_ai = min(Ai_linear, [], 2);
                                        opt_bi = 0;
        
                                    elseif contains(strat, "indiv. homog. affine")
                                        opt_ai = min(Ai_affine, [], 2);
                                        opt_bi = opt_ai .* Mi ./ 2;
        
                                    elseif contains(strat, "indiv. hetero. linear")
                                        [Gamma_linear, ~] = min(ci .* Ai_linear, [], 2);
                                        opt_ai = Gamma_linear ./ ci;
                                        opt_bi = 0;
        
                                    elseif contains(strat, "indiv. hetero. affine")
                                        [Gamma_affine, ~] = min(ci .* Ai_affine, [], 2);
                                        opt_ai = Gamma_affine ./ ci;
                                        opt_bi = opt_ai .* Mi ./ 2;
        
                                    %%% MIXED ENERGY CONSTRAINT
                                    elseif contains(strat, "mixed homog. linear")
                                        indiv_cond = min(Ai_linear, [], 2);
                                        sum_cond = sqrt(P_total ./ sum(Ci_linear,2));
        
                                        opt_ai = min([indiv_cond sum_cond], [], 2);
                                        opt_bi = 0;
                                    elseif contains(strat, "mixed homog. affine")
                                        indiv_cond = min(Ai_affine, [], 2);
                                        sum_cond = sqrt(P_total ./ sum(Ci_affine, 2));
        
                                        opt_ai = min([indiv_cond sum_cond], [], 2);
                                        opt_bi = opt_ai .* Mi ./ 2;
                                    elseif contains(strat, "mixed hetero. linear")
                                        [Gamma_linear, ~] = min(ci .* Ai_linear, [], 2);
                                        sum_cond_linear = sqrt(P_total ./ sum(Ci_linear ./ ci.^2, 2));
                                        mixed_cond_linear = [Gamma_linear sum_cond_linear];
        
                                        opt_ai = min(mixed_cond_linear, [], 2) ./ ci;
                                        opt_bi = 0;
                                    elseif contains(strat, "mixed hetero. affine")
                                        [Gamma_affine, ~] = min(ci .* Ai_affine, [], 2);
                                        sum_cond_affine = sqrt(P_total ./ sum(Ci_affine ./ ci.^2, 2));
                                        mixed_cond_affine = [Gamma_affine sum_cond_affine];
        
                                        opt_ai = min(mixed_cond_affine, [], 2) ./ ci;
                                        opt_bi = opt_ai .* Mi ./ 2;
                                    end
                                    %%% end mixed energy constraint
                                end
                                % Store coeffs if using surr. approx
                                if ~contains(strat, "fmincon") % contains(strat, "surr. approx")
                                    opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                                    opt_coeffs(strat_idx,S+1:end,:,:) = opt_bi .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                                % Run fminsearch
                                elseif contains(strat, "fmincon")
                                    % Define parameters for fminsearch
                                    options = optimoptions('fmincon','Display','off','Algorithm','sqp');
                                    opt_coeffs_cell = cell(1, nDeployments);
        
                                    % run fmincon using parallelization
                                    parfor dep_idx = 1:nDeployments
                                        % x0 only depends on ai, bi values are just placeholders here
                                        % use surr. approx. as starting point (where applicable)
                                        if contains(strat, "homog. linear")
                                            % x0 = [opt_ai(:,:,:,dep_idx) 0];
                                            x0 = [1 0];
                                        elseif contains(strat, "homog. affine")
                                            % x0 = [opt_ai(:,:,:,dep_idx) 1];
                                            x0 = [1 1];
                                        elseif contains(strat, "hetero. linear")
                                            % x0 = [opt_ai(:,:,:,dep_idx) zeros(1,S)];
                                            x0 = [ones(1,S) zeros(1,S)];
                                        elseif contains(strat, "hetero. affine")
                                            % x0 = [opt_ai(:,:,:,dep_idx) ones(1,S)];
                                            x0 = [ones(1,2*S)]
                                        end
        
                                        dep_ci = ci(:,:,:,dep_idx);
                                        dep_mi = mi(:,:,:,dep_idx);
        
                                        lb = zeros(size(x0)); % ai >= 0, bi >= 0
                                        ub = []; % unrestricted
        
                                        local_coeffs = zeros(1,2*S);
                                        % constraints function expects input for x0 to be [ai bi]
                                        % nonlcon = @(x) fmincon_constraints(x, C1i(:,:,:,dep_idx), C2i(:,:,:,dep_idx), T0, Ei_max(:,:,:,dep_idx), E_total(:,:,:,dep_idx), strat, D_val);
                                        nonlcon = @(x) fmincon_constraints(x, Mi(:,:,:,dep_idx), Gamma_i(:,:,:,dep_idx), Pi_max(:,:,:,dep_idx), P_total(:,:,:,dep_idx), strat);
                                        
        
                                        % compute optimal ai and bi using weighted sum objective function
                                        for jj = 1:length(alpha_weight_vals)
                                            alpha_weight = alpha_weight_vals(jj);
                                            t0_weight = 1 - alpha_weight;
        
                                            if scheme == "EPC"
                                                weighted_sum_objective = @(x) alpha_weight .* alpha_objective_array(x(1:length(x)/2), dep_mi, dep_ci) + t0_weight .* t0_objective_array(x(1:length(x)/2), dep_mi, dep_ci, 1);
                                            end
                                            [optimal_x, ~] = fmincon(weighted_sum_objective, x0, [], [], [], [], lb, ub, nonlcon, options);
        
                                            % assign ai
                                            local_coeffs(1:S) = optimal_x(1:length(optimal_x)/2);
                                            % assign bi
                                            % local_coeffs(S+1:end) = optimal_x((length(optimal_x)/2)+1:end);
                                            if contains(strat, "linear")
                                                local_coeffs(S+1:end) = 0;
                                            elseif contains(strat, "affine")
                                                local_coeffs(S+1:end) = local_coeffs(1:S) .* Mi(:,:,:,dep_idx) ./ 2;
                                            end
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
                            
                            %%% compute energy
                            % ai = opt_coeffs(strat_idx, 1:S, :, :);
                            % bi = opt_coeffs(strat_idx, S+1:end, :, :);
                            % 
                            % if contains(strat, "clip")
                            %     sqrd_E_quantity = E_clip_uit_sqrd;
                            %     E_quantity = E_clip_uit;
                            % else
                            %     sqrd_E_quantity = E_uit_sqrd;
                            %     E_quantity = E_uit;
                            % end
                            % 
                            % deployment_sensor_energy = ai.^2 .* sqrd_E_quantity - 2 .* ai .* bi .* E_quantity + bi.^2 * T0;
                            % empirical_sensor_energy(strat_idx,:,:,:) = deployment_sensor_energy;
                            % indiv_energy_nae = mean(abs(Ei_max - deployment_sensor_energy) ./ Ei_max, "all");
                            % 
                            % deployment_total_energy = sum(ai.^2 .* sqrd_E_quantity - 2 .* ai .* bi .* E_quantity + bi.^2 * T0, 2);
                            % empirical_total_energy(strat_idx,:,:,:) = deployment_total_energy;
                            % total_energy_nae = mean(abs(squeeze(E_total)-squeeze(deployment_total_energy))./squeeze(E_total));
                            % 
                            % disp(strat + " Indiv. Energy Norm Abs Err: " + num2str(indiv_energy_nae))
                            % disp(strat + " Total Energy Norm Abs Err: " + num2str(total_energy_nae))
                            %%% end compute energy
    
                            total_opt_time = toc(total_opt_start);
                            disp("Total " + strat + " time: " + floor(total_opt_time/60) + " minutes " + mod(total_opt_time,60) + " seconds")
                            fprintf("\n")
                        end
    
                        % Compute energy scalars for surr. approx.
    
                        % guarantee same energy usage, always scale to hetero.
                        % coefficients because they consider CRLB obj. funct.
                        % idx1 = find(all_strats == "indiv. homog. linear surr. approx.");
                        % idx2 = find(all_strats == "indiv. hetero. linear surr. approx.");
                        % linear_homog_scaler = empirical_total_energy(idx2,:,:,:) ./ empirical_total_energy(idx1,:,:,:);
                        % 
                        % idx3 = find(all_strats == "indiv. homog. affine surr. approx.");
                        % idx4 = find(all_strats == "indiv. hetero. affine surr. approx.");
                        % affine_homog_scaler = empirical_total_energy(idx4,:,:,:) ./ empirical_total_energy(idx3,:,:,:);
    
                        %%%%%%%%%%%%%%%%% ESTIMATION %%%%%%%%%%%%%%%%%
                        total_est_start = tic;
                        for strat_idx = 1:num_strats
                            strat = all_strats(strat_idx);
                            disp("Starting ML estimation for " + strat);
                            
                            % if scheme == "EPC"
                            %     if strat == "indiv. homog. linear surr. approx."
                            %         opt_coeffs(strat_idx, 1:S, :, :) = opt_coeffs(strat_idx, 1:S, :, :) .* linear_homog_scaler;
                            %     elseif strat == "indiv. homog. affine surr. approx."
                            %         opt_coeffs(strat_idx, 1:S, :, :) = opt_coeffs(strat_idx, 1:S, :, :) .* affine_homog_scaler;
                            %         opt_coeffs(strat_idx,S+1:end,:,:) = opt_coeffs(strat_idx, 1:S, :, :) .* C2i ./ T0;
                            %     end
                            % end
    
                            ai = opt_coeffs(strat_idx, 1:S, :, :);
                            bi = opt_coeffs(strat_idx, S+1:end, :, :);
    
                            [~, ui] = mf_integral_fft(noisy_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), ci, ai, K, dt, Tp);

                            if contains(strat, "clip")
                                ui_clip = max(ui-pagetranspose(Li), 0);
                                out_cws = sum(ci .* ui_clip, 2);
                            else
                                out_cws = sum(ci .* (ai .* ui - bi), 2);
                            end

                            cws = pagetranspose(out_cws);
                            y = cws + pagetranspose(scaled_n(:,1,:,:));
    
                            if contains(experiment, "cwe") || experiment == "random_dropout"
                                % Define search space for t0
                                t0_search_space = repmat(reshape(t,1,1,[]), 1,1,1,nDeployments);
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

                                    [Omega_2_t0_search, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    Omega_2_t0_search = reshape(Omega_2_t0_search - sum(ci .* bi, 2), K, 1, K, 1, nDeployments);
                                    mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                    [max_y_vals,I] = max(mf_with_y, [], 3);

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

                                    [Omega, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    resh_Qn = reshape(Qn_matrix,K,K,1,nDeployments);
                                    resh_Omega = reshape(Omega,1,K,nTrials,nDeployments);

                                    num = dt*dt*pagemtimes(pagemtimes(real(y) + sum(ci .* bi, 2),resh_Qn),pagetranspose(resh_Omega));
                                    denom = dt*dt*pagemtimes(pagemtimes(resh_Omega,resh_Qn),pagetranspose(resh_Omega));
                                end
                                alpha_estimates = num ./ denom;
                            end
    
                            % Compute empirical variance.
                            rho_empirical_var(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = var(alpha_estimates,0,3);
                            rho_empirical_var(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = var(t0_estimates_for_plot,0,3);
    
                            % Compute empirical mse.
                            rho_empirical_mse(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = mean((alpha_estimates - alpha_true).^2,3);
                            rho_empirical_mse(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = mean((t0_estimates_for_plot - t0_true).^2,3);
    
                            % Compute empirical bias.
                            rho_empirical_bias(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = mean(alpha_estimates,3);
                            rho_empirical_bias(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = mean(t0_estimates_for_plot,3);
    
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
                                
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = alpha_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci);
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = t0_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci, alpha_true);
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
    avg_dep_var(:,:,:,2,:,:) = avg_dep_var(:,:,:,2,:,:) ./ T0^2;
    avg_dep_mse(:,:,:,2,:,:) = avg_dep_mse(:,:,:,2,:,:) ./ T0^2;
    avg_dep_crlb(:,:,:,2,:,:) = avg_dep_crlb(:,:,:,2,:,:) ./ T0^2;

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
        "T0", "Tp", "alpha_true", "t0_true", "ci", "clean_y_arr", "rho_vals"];
    save(experiment + "_results.mat", files_to_save{:})
end

%% Plotting
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr(unique(iter_arr)), colors(1:numel(unique(iter_arr))));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Energy Efficient AirComp Results\";

exclude = ["None"];

if contains(experiment, "diff_rho")
    %%% rho plots
    for agent_db_idx = 1:length(agent_db_values)
        channel_db_idx = length(channel_snr);
        for param_idx = 1:2
            % Collect all plot values for y-axis scaling.
            all_vals = [avg_dep_mse(:,agent_db_idx,channel_db_idx,param_idx,:,1:length(rho_vals))
                        % avg_dep_crlb(:,agent_db_idx,channel_db_idx,param_idx,:,1:length(rho_vals));
                        % avg_dep_var(:,agent_db_idx,channel_db_idx,param_idx,:,1:length(rho_vals));
                        ];

            for scheme_idx = 1:length(selected_schemes)
                scheme = selected_schemes(scheme_idx);

                count_vals = rho_vals;

                for constraint_idx = 1:length(constraints)
                    constraint = constraints(constraint_idx);
                    data_idx_seq = length(constraints)*(1:iter_length) - (length(constraints)-constraint_idx);
                    % Plot pseudoplots for legend symbols.

                    fig1 = figure;
                    set(fig1,'Position',[100,100,450,450])
                    ax = gca;
                    hold(ax, 'on');

                    for count_idx = data_idx_seq
                        strat = all_strats(count_idx);
                        split_str = split(strat);
                        iter_key = join(split_str(2:end), " ");

                        if contains(strat, exclude)
                            continue
                        end

                        plot(ax, nan,nan,'color',colorMap(iter_key),'LineWidth',2);
                    end

                    plot(ax, nan,nan,'x','color','black');

                    x_axis_series = rho_vals;
                    plot_line_width = 1.5;

                    for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                        strat = all_strats(strat_idx);
                        split_str = split(strat);
                        iter_key = join(split_str(2:end), " ");

                        % plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,:,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,:,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        % plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,:,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
                    end

                    xlabel('$$\rho$$','Interpreter','latex')
                    ylabel(" ")

                    title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')

                    ax.FontSize = 15;          % font size
                    ax.YScale = 'log';          % log scale if needed
                    grid(ax, 'on');             % turn on grid

                    legend(ax, [all_strats(data_idx_seq)',"MSE"], 'Location', 'best')

                    grid on

                    y_lower = 10^(-0.25)*min(all_vals,[],"all");
                    y_upper = 1.05*max(all_vals,[],"all");
                    ylim([y_lower,y_upper]);

                    % Save the figure.
                    if save_images
                        save_folder = folder_path + constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp);
                        if ~exist(save_folder, 'dir')
                            mkdir(save_folder);
                        end
                        exportgraphics(fig1, fullfile(save_folder, sprintf(constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                    end
                end
            end
        end
    end
end

% %% overlaid plots
for agent_db_idx = 1:length(agent_db_values)
    % selected_strat_idxs = [find(contains(all_strats, "fmincon"))].';
    selected_strat_idxs = 1:numel(all_strats);
    for param_idx = 1:2
        % Collect all plot values for y-axis scaling.
        all_vals = [avg_dep_crlb(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                    avg_dep_var(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                    avg_dep_mse(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension)
                    ];

        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);

            % set(fig1,'Position',[0,0,1920,1080], 'Name', params_text(param_idx) + " hat | " + constraints(1) + " | alpha = " + alpha_true + ", t0 = " + t0_true + ", Tp = " + Tp + ", T0 = " + T0, 'NumberTitle', 'off')
            % tiledlayout(1,length(constraints));

            count_vals = sensor_vals;

            for constraint_idx = 1:length(constraints)
                constraint = constraints(constraint_idx);
                data_idx_seq = intersect(selected_strat_idxs, numel(iter_arr)*(constraint_idx-1) + (1:numel(iter_arr)), 'stable');
                % Plot pseudoplots for legend symbols.

                fig1 = figure;
                set(fig1,'Position',[100,100,450,450])
                ax = gca;
                hold(ax, 'on');

                lgd = [];

                for count_idx = data_idx_seq
                    strat = all_strats(count_idx);
                    split_str = split(strat);
                    iter_key = join(split_str(2:end), " ");

                    lgd = [lgd iter_key];

                    plot(ax, nan,nan,'color',colorMap(iter_key),'LineWidth',2);
                end

                plot(ax, nan,nan,'^','color','black');
                plot(ax, nan,nan,'x','color','black');
                plot(ax, nan,nan,'o','color','black');

                x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
                plot_line_width = 1.5;

                for count_idx = 1:length(count_vals)
                    for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                        strat = all_strats(strat_idx);
                        split_str = split(strat);
                        iter_key = join(split_str(2:end), " ");

                        plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
                    end
                end
                % plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx+1,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color','black','LineWidth', plot_line_width)
                % plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx+1,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color','black','LineWidth', plot_line_width)
                % plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx+1,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color','black','LineWidth', plot_line_width)

                xlabel('Channel SNR (dB)')
                ylabel(" ")

                title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')

                ax.FontSize = 15;          % font size
                ax.YScale = 'log';          % log scale if needed
                grid(ax, 'on');             % turn on grid

                legend(ax, [lgd, "VAR","MSE","CRLB"], 'Location', 'ne')

                grid on

                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 1.05*max(all_vals,[],"all");
                ylim([y_lower,y_upper]);

                % Save the figure.
                % if save_images
                %     save_folder = folder_path + "overlaid";
                %     if ~exist(save_folder, 'dir')
                %         mkdir(save_folder);
                %     end
                %     exportgraphics(fig1, fullfile(save_folder, sprintf("overlaid" + "_" + constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                % end
            end

            % if param_idx == 1
            %     for constraint_idx = 1:length(constraints)
            %         %%% Plot histogram of coefficients
            %         fig2 = figure;
            %         set(fig2,'Position',[0,0,1920,1080])
            %         tiledlayout(2,4);
            %         data_idx_seq = length(constraints)*(1:iter_length) - (length(constraints)-constraint_idx);
            % 
            %         for idx = data_idx_seq
            %             strat = all_strats(idx);
            %             ax2 = nexttile;
            %             bar(ax2, opt_coeffs(idx,:,:,1));
            %             title(strat)
            %         end
            %         sgtitle(scheme + " " + constraints(constraint_idx))
            %     end
            % end
        end
    end
end

% %% individual homog and hetero plots
% for agent_db_idx = 1:length(agent_db_values)
%     for param_idx = 1:2
%         % Collect all plot values for y-axis scaling.
%         all_vals = [avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
%                     avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
%                     avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension)
%                     ];
% 
%         for scheme_idx = 1:length(selected_schemes)
%             scheme = selected_schemes(scheme_idx);
% 
%             count_vals = sensor_vals;
% 
%             for constraint_idx = 1:length(constraints)
%                 constraint = constraints(constraint_idx);
% 
%                 data_idx_seq = numel(iter_arr)*(constraint_idx-1) + (1:numel(iter_arr));
%                 hetero_seq = data_idx_seq(1:(numel(data_idx_seq)/2));
%                 homog_seq = data_idx_seq((numel(data_idx_seq)/2+1):end);
% 
%                 for seq_type_idx = 1:2
%                     if seq_type_idx == 1
%                         my_seq = hetero_seq;
%                         coeff_type = "hetero";
%                     else
%                         my_seq = homog_seq;
%                         coeff_type = "homog";
%                     end
% 
%                     fig1 = figure;
%                     ax = gca;
%                     set(fig1,'Position',[100,100,450,450])
%                     hold(ax, 'on');
% 
%                     % Plot pseudoplots for legend symbols.
%                     for count_idx = my_seq
%                         strat = all_strats(count_idx);
%                         split_str = split(strat);
%                         iter_key = join(split_str(2:end), " ");
% 
%                         if contains(strat, exclude)
%                             continue
%                         end
% 
%                         plot(ax, nan,nan,'color',colorMap(iter_key),'LineWidth',2);
%                     end
% 
%                     plot(ax, nan,nan,'^','color','black');
%                     plot(ax, nan,nan,'x','color','black');
%                     plot(ax, nan,nan,'o','color','black');
% 
%                     x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
%                     plot_line_width = 1.5;
% 
%                     for count_idx = 1:length(count_vals)
%                         for strat_idx = my_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
%                             strat = all_strats(strat_idx);
%                             split_str = split(strat);
%                             iter_key = join(split_str(2:end), " ");
% 
%                             if contains(strat, exclude)
%                                 continue
%                             end
% 
%                             plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
%                             plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
%                             plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
%                         end
%                     end
% 
%                     xlabel('Channel SNR (dB)')
%                     ylabel(" ")
% 
%                     title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
% 
%                     ax.FontSize = 15;          % font size
%                     ax.YScale = 'log';          % log scale if needed
%                     grid(ax, 'on');             % turn on grid
% 
%                     % if constraint == "total"
%                     legend_strats = unique(tr + " " + ap);
%                     legend(ax, [all_strats(my_seq)',"VAR","MSE","CRLB"], 'Location', 'ne')
%                     % end
% 
%                     grid on
% 
%                     y_lower = 10^(-0.25)*min(all_vals,[],"all");
%                     y_upper = 1.05*max(all_vals,[],"all");
%                     ylim([y_lower,y_upper]);
% 
%                     % Save the figure.
%                     if save_images
%                         % save_folder = folder_path;
%                         save_folder = folder_path + "separated";
%                         if ~exist(save_folder, 'dir')
%                             mkdir(save_folder);
%                         end
%                         exportgraphics(fig1, fullfile(save_folder, sprintf(constraint + "_" + coeff_type + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
%                     end
%                 end
%             end
%         end
%     end
% end

%% Functions
function [c, ceq] = fmincon_constraints(x, Mi, Gamma_i, Pi_max, P_total, strat)
ai = x(1:length(x)/2);
% bi = x(length(x)/2 + 1:end);

if contains(strat, "linear")
    indiv_peak_power_vals = (ai .* (Mi + Gamma_i)).^2;
else
    % indiv_peak_power_vals = ai .* (Mi + Gamma_i) - bi;
    indiv_peak_power_vals = (ai .* (Mi./2 + Gamma_i)).^2;
end

total_cond = sum(indiv_peak_power_vals) - P_total;
indiv_upper_cond = (indiv_peak_power_vals - Pi_max)';
% indiv_lower_cond = (bi + ai .* Gamma_i - sqrt(Pi_max))';

if contains(strat, "total")
    c = [total_cond];
elseif contains(strat, "indiv.")
    c = [indiv_upper_cond]; % , indiv_lower_cond];
elseif contains(strat, "mixed")
    c = [total_cond; indiv_upper_cond];
else
    error("Unknown experiment specified!")
end

ceq = [];
end