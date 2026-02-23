% Clear all variables and close all exisiting figures.
clear all
close all

% Choose which experiment to run.
experiment_list = ["cwe_disjoint_energy_opt", "cwe_disjoint_energy_opt_diff_rho"];
experiment = "cwe_disjoint_energy_opt";

msg = ['Running ' char(experiment)];
border = repmat('─', 1, length(msg)+4);

fprintf('\n╭%s╮\n', border);
fprintf('│  %s  │\n', msg);
fprintf('╰%s╯\n\n', border);

% Set seed for reproducibility
seed = 100;
rng(seed);

nDeployments = 5;
% Set the number of measurements per sensor, starting from 0
K = 501;
% Set number of trials.
nTrials = 1000;

% Set number of sensors
sensor_vals = 8; % [5,10]
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
all_mi = unifrnd(min_mi,max_mi,1,max_sensor_limit,1,nDeployments);
all_ti = unifrnd(min_ti,max_ti,1,max_sensor_limit,1,nDeployments);
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
    channel_snr = 20; %%% TODO: need to make this dynamic
    rho_vals = [0.5, 0.75, 1, 1.25, 1.5, 2];

    constraints = ["sum"]; % "sum", "indiv.", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.","hetero."];  % "homog.", "hetero."
    approaches = ["surr. approx."]; % "surr. approx.", "fmincon"
else
    channel_snr = [0,5,10,15,20];
    rho_vals = 0.50;
    
    constraints = ["indiv."]; % "sum", "indiv.", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.","hetero."];  % "homog.", "hetero."
    approaches = ["surr. approx."]; % "surr. approx.", "fmincon"
end

% constraints = ["none"]; % "sum", "indiv.", "mixed"
% transforms = ["none"]; % "linear", "affine"
% coeff_types = ["none"];  % "homog.", "hetero."
% approaches = ["none"]; % "surr. approx.", "fmincon"

[sub_c, tr, ct, ap] = ndgrid(constraints, transforms, coeff_types, approaches);

iter_arr = tr + " " + ct + " " + ap;
all_strats = sub_c + " " + iter_arr;
all_strats = all_strats(:);   % flatten into list

all_strats = [all_strats; "clip"];

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
    
                % Calculate quantized phase.
                M = 4;
                [qgi,~] = quantize_comp(gi,M);
    
                clean_unified_xi = alpha_true .* (mi .* util_functions.sensor_signal(t-t0_true, Tp, norm_fact));
                noisy_unified_xi = clean_unified_xi + scaled_w(:,1:S,:,:);

                Ei = sum((mi .* util_functions.sensor_signal(t-t0_true, Tp, norm_fact)).^2 .* dt, 1);

                [~, clean_ui_1] = util_functions.mf_integral_fft(clean_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), 1, 1, K, dt, Tp);
                clean_ui = pagetranspose(clean_ui_1);
    
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
    
                vect_gamma_i = sum(mi .* util_functions.sensor_signal(t-ti-three_dim_t, Tp, norm_fact).^2,1) * dt;
                vect_int_gamma_i = sum(vect_gamma_i,3)*dt;
                int_gamma_i = vect_int_gamma_i;
    
                % omniscient assumption (hypothetical upper bound on energy)
                omni_C1i = alpha_true^2 .* mi.^4 .* int_Rs0_sqrd + scaled_w_psd_constant .* int_gamma_i; % E[ui(t)^2]
                omni_C2i = alpha_true .* mi.^2 .* int_Rs0; % E[ui(t)]
                omni_Ei_max = omni_C1i;
                omni_E_total = sum(omni_Ei_max,2);

                C1i = omni_C1i;
                C2i = omni_C2i;
                Ei_max = rho * C1i;
                E_total = D_val * sum(Ei_max,2);
    
                Ci_tilde = C1i - (C2i.^2)/T0;
                Ai_affine = sqrt(Ei_max ./ Ci_tilde);
                Ai_linear = sqrt(Ei_max ./ C1i);

                fundamental_E = rho*int_Rs0_sqrd;

                s0 = @(tau) (heaviside(tau)-heaviside(tau-1)) .* sin(pi/Tp .* tau);
                Rs0 = @(t) integral(@(u) s0(u-t0_true) .* s0(u-t), 0, T0);
           
                my_int_Rs0_sqr = integral(@(t) Rs0(t).^2, 0, T0, 'ArrayValued', true);

                calc_energy = @(t1) 2*integral(@(t) Rs0(t).^2, 0, t1, 'ArrayValued', true) + Rs0(t1).^2 * (2*Tp - 2*t1);

                saturation_obj = @(t1) calc_energy(t1) - fundamental_E;

                t1_optimal = fzero(saturation_obj, [0, Tp]);
                
                clip_L = pagetranspose(Rs0(t1_optimal) .* alpha_true .* mi.^2);
                clip_ui = min(ui, clip_L);
                E_clip_uit_sqrd = pagetranspose(sum(mean(clip_ui.^2,3),2)) * dt;
                E_clip_uit = pagetranspose(sum(mean(clip_ui,3),2)) * dt;
                clip_energy_nae = mean(abs(Ei_max - E_clip_uit_sqrd) ./ Ei_max, "all");

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
    
                            elseif (scheme == "PPC" || scheme == "NPC")
                                giR = real(ci);
                                giI = imag(ci);
    
                                % Compute polynomial to find coefficients for
                                % decorrelation filter.
                                b = @(ai,mi,giR,giI) sum(giR .* giI .* ai.^2 .* mi.^2,2) ./ (sum(ai.^2 .* mi.^2 .* (giR.^2 - giI.^2),2));
                                discr = @(ai,mi,giR,giI) 1 + 4.*b(ai,mi,giR,giI).^2;
                                a1 = @(ai,mi,giR,giI)(-1 + sqrt(discr(ai,mi,giR,giI))) ./ (2*b(ai,mi,giR,giI));
                                a2 = @(ai,mi,giR,giI) (-1 - sqrt(discr(ai,mi,giR,giI))) ./ (2*b(ai,mi,giR,giI));
                                a_root = @(ai,mi,giR,giI) a1(ai,mi,giR,giI);
    
                                % Compute pR and qR.
                                pR = @(ai,mi,giR,giI) scaled_w_psd_constant * sum(ai.^2 .* mi.^2 .* (a_root(ai,mi,giR,giI).*giR - giI).^2, 2);
                                qR = @(ai,mi,giR,giI) (scaled_n_psd_constant/2 * (a_root(ai,mi,giR,giI).^2 + 1)) - Gamma*pR(ai,mi,giR,giI);
    
                                % Compute pI and qI.
                                pI = @(ai,mi,giR,giI) scaled_w_psd_constant * sum(ai.^2 .* mi.^2 .* (giR + a_root(ai,mi,giR,giI).*giI).^2, 2);
                                qI = @(ai,mi,giR,giI) (scaled_n_psd_constant/2 * (a_root(ai,mi,giR,giI).^2 + 1)) - Gamma*pI(ai,mi,giR,giI);
    
                                % Compute SKndR and SKndI.
                                SKndR_anon = @(w,ai,mi,giR,giI) pR(ai,mi,giR,giI) + mag_sqr_SHdI_anon(w) .* qR(ai,mi,giR,giI);
                                SKndI_anon = @(w,ai,mi,giR,giI) pI(ai,mi,giR,giI) + mag_sqr_SHdI_anon(w) .* qI(ai,mi,giR,giI);
    
                                BR = @(ai,mi,giR,giI) (a_root(ai,mi,giR,giI) .* sum(giR .* ai .* mi.^2, 2) - sum(giI .* ai .* mi.^2, 2)).^2;
                                BI = @(ai,mi,giR,giI) (a_root(ai,mi,giR,giI) .* sum(giI .* ai .* mi.^2, 2) + sum(giR .* ai .* mi.^2, 2)).^2;
    
                                alpha_crlb = @(ai,mi,giR,giI) (2*pi) ./ (BR(ai,mi,giR,giI) .* integral(@(w) (mag_sqr_S0(w)).^2 .* mag_sqr_SHdI_anon(w) ./ SKndR_anon(w,ai,mi,giR,giI), -Inf, Inf, 'ArrayValued', true)...
                                    + BI(ai,mi,giR,giI) .* integral(@(w) (mag_sqr_S0(w)).^2 .* mag_sqr_SHdI_anon(w) ./ SKndI_anon(w,ai,mi,giR,giI), -Inf, Inf, 'ArrayValued', true));
                                t0_crlb = @(ai,mi,giR,giI,arb_alpha) (2*pi) ./ (BR(ai,mi,giR,giI) .* integral(@(w) arb_alpha.^2 .* (w.*mag_sqr_S0(w)).^2 .* mag_sqr_SHdI_anon(w) ./ SKndR_anon(w,ai,mi,giR,giI), -Inf, Inf, 'ArrayValued', true)...
                                    + BI(ai,mi,giR,giI) .* integral(@(w) arb_alpha.^2 .* (w*mag_sqr_S0(w)).^2 .* mag_sqr_SHdI_anon(w) ./ SKndI_anon(w,ai,mi,giR,giI), -Inf, Inf, 'ArrayValued', true));
                            end
                        end
    
                        %%%%%%%%%%%%%%%%% Optimize pulse shaping coefficients %%%%%%%%%%%%%%%%%
                        for strat_idx = 1:num_strats
                            strat = all_strats(strat_idx);
                            disp("Starting " + strat + "...");
                            total_opt_start = tic;
    
                            % baseline estimation without optimal coefficients
                            if contains(strat, "none") || contains(strat, "clip")
                                opt_coeffs(:,1:S,:,:) = ones(size(opt_coeffs(:,1:S,:,:)));
                                opt_coeffs(:,S+1:end,:,:) = zeros(size(opt_coeffs(:,S+1:end,:,:)));
                            else
                                % estimation with coefficients
                                if scheme == "EPC"
                                    if contains(strat, "sum linear homog.")
                                        opt_ai = sqrt(E_total ./ sum(C1i,2));
                                        opt_bi = 0;
        
                                    elseif contains(strat, "sum affine homog.")
                                        opt_ai = sqrt(E_total ./ sum(Ci_tilde, 2));
                                        opt_bi = opt_ai .* C2i ./ T0;
        
                                    elseif contains(strat, "sum linear hetero.")
                                        opt_ai = (1./ci) .* sqrt(E_total ./ sum(C1i ./ ci.^2, 2));
                                        opt_bi = 0;
        
                                    elseif contains(strat, "sum affine hetero.")
                                        opt_ai = (1./ci) .* sqrt(E_total ./ sum(Ci_tilde ./ ci.^2, 2));
                                        opt_bi = opt_ai .* C2i ./ T0;
                                        %%% end total energy constraint
        
                                        %%% individual energy constraint
                                    elseif contains(strat, "indiv. linear homog.")
                                        opt_ai = min(Ai_linear, [], 2);
                                        opt_bi = 0;
        
                                    elseif contains(strat, "indiv. affine homog.")
                                        opt_ai = min(Ai_affine, [], 2);
                                        opt_bi = opt_ai .* C2i ./ T0;
        
                                    elseif contains(strat, "indiv. linear hetero.")
                                        [Gamma_linear, ~] = min(ci .* Ai_linear, [], 2);
                                        opt_ai = Gamma_linear ./ ci;
                                        opt_bi = 0;
        
                                    elseif contains(strat, "indiv. affine hetero.")
                                        [Gamma_affine, ~] = min(ci .* Ai_affine, [], 2);
                                        opt_ai = Gamma_affine ./ ci;
                                        opt_bi = opt_ai .* C2i ./ T0;
                                        %%% individual energy constraint
        
                                        %%% mixed energy constraint
                                    elseif contains(strat, "mixed linear homog.")
                                        indiv_cond = min(Ai_linear, [], 2);
                                        sum_cond = sqrt(D_val * E_total ./ sum(C1i,2));
        
                                        opt_ai = min([indiv_cond sum_cond], [], 2);
                                        opt_bi = 0;
                                    elseif contains(strat, "mixed affine homog.")
                                        indiv_cond = min(Ai_affine, [], 2);
                                        sum_cond = sqrt(E_total ./ sum(Ci_tilde, 2));
        
                                        opt_ai = min([indiv_cond sum_cond], [], 2);
                                        opt_bi = opt_ai .* C2i ./ T0;
                                    elseif contains(strat, "mixed linear hetero.")
                                        [Gamma_linear, ~] = min(ci .* Ai_linear, [], 2);
                                        sum_cond_linear = sqrt(E_total ./ sum(C1i ./ ci.^2, 2));
                                        mixed_cond_linear = [Gamma_linear sum_cond_linear];
        
                                        opt_ai = min(mixed_cond_linear, [], 2) ./ ci;
                                        opt_bi = 0;
                                    elseif contains(strat, "mixed affine hetero.")
                                        [Gamma_affine, ~] = min(ci .* Ai_affine, [], 2);
                                        sum_cond_affine = sqrt(E_total ./ sum(Ci_tilde ./ ci.^2, 2));
                                        mixed_cond_affine = [Gamma_affine sum_cond_affine];
        
                                        opt_ai = min(mixed_cond_affine, [], 2) ./ ci;
                                        opt_bi = opt_ai .* C2i ./ T0;
                                    end
                                elseif scheme == "PPC"
                                    if contains(strat, "hetero.")
                                        opt_ai = ones(size(ci));
                                        opt_bi = zeros(size(ci));
                                    elseif contains(strat, "homog.")
                                        opt_ai = ones(1,1,1,nDeployments);
                                        opt_bi = 0;
                                    end
                                end
                                % Store coeffs if using surr. approx
                                if contains(strat, "surr. approx")
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
                                        if contains(strat, "affine homog.")
                                            x0 = [opt_ai(:,:,:,dep_idx) 1];
                                        elseif contains(strat, "affine hetero.")
                                            x0 = [opt_ai(:,:,:,dep_idx) ones(1,S)];
                                        elseif contains(strat, "linear homog.")
                                            x0 = [opt_ai(:,:,:,dep_idx) 0];
                                        elseif contains(strat, "linear hetero.")
                                            x0 = [opt_ai(:,:,:,dep_idx) zeros(1,S)];
                                        end
        
                                        dep_ci = ci(:,:,:,dep_idx);
                                        dep_mi = mi(:,:,:,dep_idx);
        
                                        lb = zeros(size(x0)); % ai >= 0, bi >= 0
                                        ub = []; % unrestricted
        
                                        local_coeffs = zeros(1,2*S);
                                        % constraints function expects input for x0 to be [ai bi]
                                        nonlcon = @(x) fmincon_constraints(x, C1i(:,:,:,dep_idx), C2i(:,:,:,dep_idx), T0, Ei_max(:,:,:,dep_idx), E_total(:,:,:,dep_idx), strat, D_val);
        
                                        % compute optimal ai and bi using weighted sum objective function
                                        for jj = 1:length(alpha_weight_vals)
                                            alpha_weight = alpha_weight_vals(jj);
                                            t0_weight = 1 - alpha_weight;
        
                                            if scheme == "EPC"
                                                weighted_sum_objective = @(x) alpha_weight .* alpha_objective_array(x(1:length(x)/2), dep_mi, dep_ci) + t0_weight .* t0_objective_array(x(1:length(x)/2), dep_mi, dep_ci, 1);
                                            elseif scheme == "PPC"
                                                weighted_sum_objective = @(x) alpha_weight .* alpha_crlb(x(1:length(x)/2), dep_mi, real(dep_ci), imag(dep_ci)) + t0_weight .* t0_crlb(x(1:length(x)/2), dep_mi, real(dep_ci), imag(dep_ci), 1);
                                            end
                                            [optimal_x, ~] = fmincon(weighted_sum_objective, x0, [], [], [], [], lb, ub, nonlcon, options);
        
                                            % assign ai
                                            local_coeffs(1:S) = optimal_x(1:length(optimal_x)/2);
                                            % assign bi
                                            if contains(strat, "linear")
                                                local_coeffs(S+1:end) = 0;
                                            elseif contains(strat, "affine")
                                                local_coeffs(S+1:end) = local_coeffs(1:S) .* C2i(:,:,:,dep_idx) ./ T0;
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
                            if channel_db_idx == 1
                                ai = opt_coeffs(strat_idx, 1:S, :, :);
                                bi = opt_coeffs(strat_idx, S+1:end, :, :);
    
                                if contains(strat, "clip")
                                    sqrd_E_quantity = E_clip_uit_sqrd;
                                    E_quantity = E_clip_uit;
                                else
                                    sqrd_E_quantity = E_uit_sqrd;
                                    E_quantity = E_uit;
                                end

                                deployment_sensor_energy = ai.^2 .* sqrd_E_quantity - 2 .* ai .* bi .* E_quantity + bi.^2 * T0;
                                empirical_sensor_energy(strat_idx,:,:,:) = deployment_sensor_energy;
                                indiv_energy_nae = mean(abs(Ei_max - deployment_sensor_energy) ./ Ei_max, "all");
                                disp(strat + " Indiv. Energy Norm Abs Err: " + num2str(indiv_energy_nae))
    
                                deployment_total_energy = sum(ai.^2 .* sqrd_E_quantity - 2 .* ai .* bi .* E_quantity + bi.^2 * T0, 2);
                                empirical_total_energy(strat_idx,:,:,:) = deployment_total_energy;
                                total_energy_nae = mean(abs(squeeze(E_total)-squeeze(deployment_total_energy))./squeeze(E_total));
                                disp(strat + " Total Energy Norm Abs Err: " + num2str(total_energy_nae))
                            end
                            %%% end compute energy
    
                            total_opt_time = toc(total_opt_start);
                            disp("Total " + strat + " time: " + floor(total_opt_time/60) + " minutes " + mod(total_opt_time,60) + " seconds")
                            fprintf("\n")
                        end
    
                        % Compute energy scalars for surr. approx.
    
                        % guarantee same energy usage, always scale to hetero.
                        % coefficients because they consider CRLB obj. funct.
                        idx1 = find(all_strats == "indiv. linear homog. surr. approx.");
                        idx2 = find(all_strats == "indiv. linear hetero. surr. approx.");
                        linear_homog_scaler = empirical_total_energy(idx2,:,:,:) ./ empirical_total_energy(idx1,:,:,:);
    
                        idx3 = find(all_strats == "indiv. affine homog. surr. approx.");
                        idx4 = find(all_strats == "indiv. affine hetero. surr. approx.");
                        affine_homog_scaler = empirical_total_energy(idx4,:,:,:) ./ empirical_total_energy(idx3,:,:,:);
    
                        %%%%%%%%%%%%%%%%% ESTIMATION %%%%%%%%%%%%%%%%%
                        total_est_start = tic;
                        for strat_idx = 1:num_strats
                            strat = all_strats(strat_idx);
                            disp("Starting ML estimation for " + strat);
                            if scheme == "EPC"
                                if strat == "indiv. linear homog. surr. approx."
                                    opt_coeffs(strat_idx, 1:S, :, :) = opt_coeffs(strat_idx, 1:S, :, :) .* linear_homog_scaler;
                                elseif strat == "indiv. affine homog. surr. approx."
                                    opt_coeffs(strat_idx, 1:S, :, :) = opt_coeffs(strat_idx, 1:S, :, :) .* affine_homog_scaler;
                                    opt_coeffs(strat_idx,S+1:end,:,:) = opt_coeffs(strat_idx, 1:S, :, :) .* C2i ./ T0;
                                end
                            end
    
                            ai = opt_coeffs(strat_idx, 1:S, :, :);
                            bi = opt_coeffs(strat_idx, S+1:end, :, :);
    
                            % [clean_cws_2, clean_ui_2] = util_functions.mf_integral_fft(clean_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), ci, ai, K, dt, Tp);
                            % clean_y_arr(strat_idx,:,:,:) = clean_cws_2 - sum(ci .* bi, 2);
    
                            % Construct y
                            [~, ui_2] = mf_integral_fft(noisy_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), ci, ai, K, dt, Tp);
                            
                            if contains(strat, "clip")
                                ui_2 = min(ui_2, pagetranspose(clip_L));
                            end

                            out_cws = sum(ui_2 .* ci .* ai, 2);
                            cws = pagetranspose(out_cws);
    
                            y = cws - sum(ci .* bi, 2) + pagetranspose(scaled_n(:,1,:,:));
    
                            if contains(experiment, "cwe") || experiment == "random_dropout"
                                % Define search space for t0
                                t0_search_space = repmat(reshape(t,1,1,[]), 1,1,1,nDeployments);
                                % Compute AC-CWE estimates for full phase compensation.
                                if scheme == "MPC" || scheme == "EPC"
                                    mag_sqr_H = H_arr(omega, ai, mi, ci);
    
                                    % Get time domain matrix for Qn.
                                    [~, Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments);
    
                                    % estimate t0
                                    [Omega_2_t0_search, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    Omega_2_t0_search = reshape(Omega_2_t0_search - sum(ci .* bi, 2), K, 1, K, 1, nDeployments);
    
                                    if contains(strat, "clip")
                                        [max_y_vals,I] = max(real(y),[],2);
                                    else
                                        mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(real(y) + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
                                        [max_y_vals,I] = max(mf_with_y, [], 3);
                                    end

                                    t0_estimates_for_plot = reshape((I-1)*dt, 1, 1, nTrials, nDeployments);
    
                                    if contains(experiment, "disjoint")
                                        t0_estimates_for_alpha = t0_true*ones(1,1,nTrials,nDeployments);
                                    else
                                        t0_estimates_for_alpha = t0_estimates_for_plot;
                                    end
    
                                    if contains(strat, "clip")
                                        num = max_y_vals;
                                        denom = sum(real(ci) .* Ei,2);
                                    else
                                        % Omega = matched_filter_integral(reshape(mi .* sensor_signal(t-ti-t0_estimates),K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-reshape(t,1,1,K)), K,S,K,1,nDeployments), ci, K, S, nTrials, nDeployments, dt);
                                        [Omega, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_estimates_for_alpha, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
        
                                        resh_Qn = reshape(Qn_matrix,K,K,1,nDeployments);
                                        resh_Omega = reshape(Omega,1,K,nTrials,nDeployments);
        
                                        num = dt*dt*pagemtimes(pagemtimes(real(y) + sum(ci .* bi, 2),resh_Qn),pagetranspose(resh_Omega));
                                        denom = dt*dt*pagemtimes(pagemtimes(resh_Omega,resh_Qn),pagetranspose(resh_Omega));
                                    end
                                    
                                elseif (scheme == "PPC" || scheme == "NPC")
                                    mag_sqr_SHdI = ones(1,K,1,nDeployments) .* mag_sqr_SHdI_anon(omega);
                                    % Define magnitude of SHdI.
                                    mag_SHdI = sqrt(mag_sqr_SHdI);
    
                                    % Get time domain matrix for hdI.
                                    [~, hdI_matrix] = get_time_domain(mag_SHdI,dt,nDeployments);
    
                                    % Get time domain matrix for hdR.
                                    hdR_matrix = reshape(a_root(ai,mi,giR,giI),1,1,nDeployments) .* hdI_matrix;
    
                                    % Compute SKndR and SKndI.
                                    SKndR = SKndR_anon(omega,ai,mi,giR,giI);
                                    SKndI = SKndI_anon(omega,ai,mi,giR,giI);
    
                                    % Check to make sure SKndR and SKndI are positive.
                                    if sum(SKndR <= 0, "all")
                                        error("SKndR less than or equal to 0!")
                                    elseif sum(SKndI <= 0, "all")
                                        error("SKndI less than or equal to 0!")
                                    end
    
                                    % Generate whitening filters based on inverse
                                    % spectrum.
                                    SQnR = 1 ./ SKndR;
                                    SQnI = 1 ./ SKndI;
    
                                    % Get time domain matrix  for QnR and QnI.
                                    [~,QnR_matrix] = util_functions.get_time_domain(SQnR,dt,nDeployments);
                                    [~,QnI_matrix] = util_functions.get_time_domain(SQnI,dt,nDeployments);
    
                                    Qn_matrix = QnR_matrix + 1i*QnI_matrix;
    
                                    % Compute closed-form expressions for alpha
                                    t0_search_space = repmat(reshape(t,1,1,[]), 1,1,1,nDeployments);
                                    % estimate t0
                                    [Omega_2_t0_search, ~] = util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_search_space, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, ai, K, dt, Tp);
                                    Omega_2_t0_search = reshape(Omega_2_t0_search - sum(ci .* bi, 2), K, 1, K, 1, nDeployments);
    
                                    mf_with_y = dt*dt*pagemtimes(pagemtimes(reshape(y + sum(ci .* bi, 2), 1, K, 1, nTrials, nDeployments), reshape(Qn_matrix,K,K,1,1,nDeployments)), Omega_2_t0_search);
    
                                    [max_y_vals,I] = max(mf_with_y, [], 3);
                                    t0_estimates_for_plot = reshape((I-1)*dt, 1, 1, nTrials, nDeployments);
    
                                    if contains(experiment, "disjoint")
                                        t0_estimates_for_alpha = t0_true*ones(1,1,nTrials,nDeployments);
                                    else
                                        t0_estimates_for_alpha = t0_estimates_for_plot;
                                    end
    
                                    % Compute closed-form expressions for alpha
                                    [Omega, ~] = mf_integral_fft(mi .* sensor_signal(t-t0_estimates_for_alpha), mi .* sensor_signal(t), ci, 1, K, dt, Tp);
    
                                    Omega = pagetranspose(Omega);
                                    OmegaR = real(Omega);
                                    OmegaI = imag(Omega);
    
                                    %%% TODO: double check alpha is being processed correctly.
                                    yR = real(y + sum(ci .* bi, 2));
                                    yI = imag(y + sum(ci .* bi, 2));
    
                                    resh_hdR = reshape(hdR_matrix,K,K,1,nDeployments);
                                    resh_hdI = reshape(hdI_matrix,K,K,1,nDeployments);
    
                                    resh_QnR = reshape(QnR_matrix,K,K,1,nDeployments);
                                    resh_QnI = reshape(QnI_matrix,K,K,1,nDeployments);
    
                                    % The d here is for decorrelation, not derivative.
                                    OmegadR = dt*(pagemtimes(OmegaR,resh_hdR) - pagemtimes(OmegaI,resh_hdI));
                                    OmegadI = dt*(pagemtimes(OmegaI,resh_hdR) + pagemtimes(OmegaR,resh_hdI));
    
                                    ydR = dt*(pagemtimes(yR,resh_hdR) - pagemtimes(yI,resh_hdI));
                                    ydI = dt*(pagemtimes(yI,resh_hdR) + pagemtimes(yR,resh_hdI));
    
                                    num1 = pagemtimes(pagemtimes(ydR,resh_QnR),pagetranspose(OmegadR));
                                    num2 = pagemtimes(pagemtimes(ydI,resh_QnI),pagetranspose(OmegadI));
                                    num = dt*dt*(num1+num2);
    
                                    denom1 = pagemtimes(pagemtimes(OmegadR,resh_QnR),pagetranspose(OmegadR));
                                    denom2 = pagemtimes(pagemtimes(OmegadI,resh_QnI),pagetranspose(OmegadI));
                                    denom = dt*dt*(denom1+denom2);
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
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = alpha_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci);
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = t0_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, ci, alpha_true);
                            elseif scheme == "PPC"
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,rho_idx,save_dim,:) = alpha_crlb(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, real(ci), imag(ci));
                                rho_crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,rho_idx,save_dim,:) = t0_crlb(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3), mi, real(ci), imag(ci), alpha_true);
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

% if ~any(experiment == experiment_list)
%     disp("Unknown experiment specified. Results not saved!")
% else
%     files_to_save = ["avg_dep_var", "avg_dep_mse", "avg_dep_bias", "avg_dep_crlb",...
%         "agent_db_values", "selected_schemes", "dropout_vals", "sensor_vals", "channel_db_values",...
%         "experiment", "sensor_dimension", "save_images", "K", "nDeployments",...
%         "constraints", "transforms", "coeff_types", "approaches", "iter_length", "all_strats", "num_strats", "opt_coeffs", "alpha_weight_vals", "save_images",...
%         "T0", "Tp", "alpha_true", "t0_true", "ci", "clean_y_arr", "rho_vals"];
%     save(experiment + "_results.mat", files_to_save{:})
% end

%% Plotting
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr(unique(iter_arr)), colors(1:iter_length));

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

%%% overlaid plots
for agent_db_idx = 1:length(agent_db_values)
    for param_idx = 1:2
        % Collect all plot values for y-axis scaling.
        all_vals = [avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
            avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
            avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension)
            ];

        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);

            % set(fig1,'Position',[0,0,1920,1080], 'Name', params_text(param_idx) + " hat | " + constraints(1) + " | alpha = " + alpha_true + ", t0 = " + t0_true + ", Tp = " + Tp + ", T0 = " + T0, 'NumberTitle', 'off')
            % tiledlayout(1,length(constraints));

            count_vals = sensor_vals;

            for constraint_idx = 1:length(constraints)
                constraint = constraints(constraint_idx);
                data_idx_seq = length(constraints)*(1:iter_length) - (length(constraints)-constraint_idx);
                % Plot pseudoplots for legend symbols.

                fig1 = figure;
                set(fig1,'Position',[100,100,450,450])
                ax = gca;
                hold(ax, 'on');

                plot(ax, nan,nan,'^','color','black');
                plot(ax, nan,nan,'x','color','black');
                plot(ax, nan,nan,'o','color','black');

                x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
                plot_line_width = 1.5;

                for count_idx = 1:length(count_vals)
                    for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                        strat = all_strats(strat_idx);
                        disp(strat)
                        split_str = split(strat);
                        iter_key = join(split_str(2:end), " ");

                        plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
                    end
                end
                % plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx+1,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color','black','LineWidth', plot_line_width)
                plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx+1,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color','black','LineWidth', plot_line_width)
                % plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx+1,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color','black','LineWidth', plot_line_width)

                xlabel('Channel SNR (dB)')
                ylabel(" ")

                title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')

                ax.FontSize = 15;          % font size
                ax.YScale = 'log';          % log scale if needed
                grid(ax, 'on');             % turn on grid

                legend(ax, ["VAR","MSE","CRLB"], 'Location', 'ne')

                grid on

                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 1.05*max(all_vals,[],"all");
                ylim([y_lower,y_upper]);

                % Save the figure.
                if save_images
                    save_folder = folder_path + "overlaid";
                    if ~exist(save_folder, 'dir')
                        mkdir(save_folder);
                    end
                    exportgraphics(fig1, fullfile(save_folder, sprintf("overlaid" + "_" + constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                end
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
%             avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
%             avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension)
%             ];
% 
%         for scheme_idx = 1:length(selected_schemes)
%             scheme = selected_schemes(scheme_idx);
% 
% 
%             % set(fig1,'Position',[0,0,1920,1080], 'Name', params_text(param_idx) + " hat | " + constraints(1) + " | alpha = " + alpha_true + ", t0 = " + t0_true + ", Tp = " + Tp + ", T0 = " + T0, 'NumberTitle', 'off')
%             % tiledlayout(1,length(constraints));
% 
%             count_vals = sensor_vals;
% 
%             for constraint_idx = 1:length(constraints)
%                 constraint = constraints(constraint_idx);
% 
%                 data_idx_seq = 3*(1:iter_length) - (3-constraint_idx);
%                 hetero_seq = data_idx_seq([1,2,5,6]);
%                 homog_seq = data_idx_seq([3,4,7,8]);
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
%                     if constraint == "sum"
%                         legend_strats = unique(tr + " " + ap);
%                         legend(ax, [legend_strats([4,2,3,1])',"VAR","MSE","CRLB"], 'Location', 'ne')
%                     end
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
function [c, ceq] = fmincon_constraints(x, C1i, C2i, T0, Ei_max, E_total, strat, D)
ai = x(1:length(x)/2);
bi = x(length(x)/2 + 1:end);

if contains(strat, "linear")
    indiv_energy_vals = ai.^2 .* C1i;
else
    indiv_energy_vals = ai.^2 .* (C1i - (C2i.^2 ./ T0));
end

sum_cond = sum(indiv_energy_vals) - E_total;
indiv_cond = (indiv_energy_vals - Ei_max)';

if contains(strat, "sum")
    c = [sum_cond];
elseif contains(strat, "indiv.")
    c = indiv_cond;
elseif contains(strat, "mixed")
    c = [sum_cond; indiv_cond];
else
    error("Unknown experiment specified!")
end

ceq = [];
end