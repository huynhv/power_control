% 11-18-2025: Go towards using homogenous and heterogenous approaches
% Clear all variables and close all exisiting figures.

clear all
% close all

disp("Initializing simulation environment...")
% Set a seed for reproducibility.
seed = 264;
rng(seed);

% Choose which experiment to run.
experiment_list = ["nae_disjoint", "nae_joint", "cwe_disjoint", "cwe_joint", "cwe_joint_grid", "random_dropout", ...
    "nae_joint_imperfect_tpi", "cwe_joint_imperfect_tpi", "cwe_disjoint_energy_opt"];
experiment = "cwe_disjoint_energy_opt";
save_images = false;

% Set number of deployments.
nDeployments = 50; % do not change
% Set number of trials.
nTrials = 500; % do not change
% Set number of samplle points. The sample index starts at 0 so
% K = 1 + number of sampling points
K = 201;
epsilon = 0.0001;

% Set observation window length in seconds.
A = 1;
T0 = 5*A;
Tp = 1*A;

% Define true parameter values.
n_params = 2;
alpha_true = 2; % do not scale this by A
t0_true = 1*A;

% Set some bookkeeping variables depending on which experiment we are running.
if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    agent_db_values = 0;
    sensor_dimension = length(dropout_vals);
    % Here we only consider maximum number of sensors and drop from there.
    sensor_vals = 10;
else
    if experiment == "cwe_joint_grid"
        sensor_vals = 5;
        agent_db_values = [0,5,10,15,20];
    elseif experiment == "cwe_disjoint_energy_opt"
        sensor_vals = 10; %%%%% gotta investigate the difference in the plots
        agent_db_values = 0;
    else
        sensor_vals = [5,10];
        agent_db_values = 0;
    end
    dropout_vals = 0;
    sensor_dimension = length(sensor_vals);
end

% Parse experiment name for organization.
exp_split = strsplit(experiment,'_');

% Define time vector
dt = T0/(K-1);
fs = 1/dt;
t = dt*(0:K-1)';
three_dim_t = reshape(t,1,1,length(t));

% Calculate the signal energy (such that the signal is assumed to be within the
% interioir fo the observation period).
func = @(t) abs(util_functions.sensor_signal(t, Tp, 1)).^2;
Es = integral(func,0,T0);
norm_fact = 1/sqrt(Es);
Es_tilde = Es * 1; % norm_fact^2;
Ps_tilde = Es_tilde;

% Define range for distribution of mi and ti (tau_i)
upper_mi = 1;
lower_mi = 0.5;

upper_ti = 1*A;
lower_ti = 0;

% Set max tpi value based on sampling rate to give a frame of reference as to
% how many sample the error may shift the peak
if contains(experiment, "imperfect_tpi")
    upper_tpi = 0.02*A;
else
    upper_tpi = 0;
end
lower_tpi = 0;

% Define range for distributions of alpha and t0, non-informative priors
upper_alpha = alpha_true;
lower_alpha = 0;

% Set maximum value of t0 such that the signal from each sensor is guaranteed to
% be in the interioir of the observation period.
t0_max = T0-(2*Tp)-upper_ti-upper_tpi;

if t0_true > t0_max
    error("True value of t0 exceeds maximum!")
end

% Generate sensor noise with dimensions: K x S x trials x deployments.
all_w = randn(K,max(sensor_vals),nTrials,nDeployments);
% Generate server noise.
n = (randn(K,1,nTrials,nDeployments) + 1i*randn(K,1,nTrials,nDeployments));
% Generate channel gains.
all_gi = (randn(1,max(sensor_vals),1,nDeployments) + 1i*randn(1,max(sensor_vals),1,nDeployments))/sqrt(2);

% Define Rayleigh distribution parameters.
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqrd = 2*rayleigh_factor^2;

% Choose which schemes we want to plot.
selected_schemes = "EPC";

% Generate mi and ti.
all_mi = unifrnd(lower_mi,upper_mi,1,max(sensor_vals),1,nDeployments);
% all_ti = unifrnd(lower_ti,upper_ti,1,max(sensor_vals),1,nDeployments);
all_ti = zeros(1,max(sensor_vals),1,nDeployments);

if contains(experiment, "imperfect_tpi")
    % Generate clock offset vector
    all_tpi = unifrnd(lower_tpi,upper_tpi,1,max(sensor_vals),1,nDeployments);
else
    all_tpi = 0;
end

%%%% add new random variables down here?
if experiment == "random_dropout"
    which_dropout = zeros(nDeployments,max(dropout_vals));
    for idx = 1:nDeployments
        which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
    end
end

% Define expected value of mi^2.
E_mi_sqrd = (1/12) * (upper_mi-lower_mi)^2 + 0.5*(lower_mi+upper_mi);
E_alpha_sqrd = (upper_alpha^3 - lower_alpha^3)/(3*(upper_alpha-lower_alpha));
E_alpha = (lower_alpha+upper_alpha)/2;

% Define channel snr values.
channel_snr = linspace(5,20,5);
% Duplicate channel snr matrices for experiments.
channel_db_values = repmat(channel_snr, length(selected_schemes), 1, length(agent_db_values));

scaled_n_psd_array = zeros(1,length(channel_snr));
scaled_w_psd_array = zeros(1,length(agent_db_values));

alpha_weight_vals = 0.5;

if experiment == "cwe_disjoint_energy_opt"
    all_strats = [];
    constraints = ["indiv."]; % "sum", "indiv.", "mixed"
    transforms = ["linear", "affine"]; % "linear", "affine"
    coeff_types = ["homog.", "hetero."];  % "homog.", "hetero."
    approaches = ["surr. approx.", "fmincon"]; % "surr. approx.", "fmincon"

    [sub_c, tr, ct, ap] = ndgrid(constraints, transforms, coeff_types, approaches);

    all_strats = sub_c + " " + tr + " " + ct + " " + ap;
    all_strats = all_strats(:);   % flatten into list
end

num_strats = length(all_strats);

opt_coeffs = zeros(num_strats,2*max(sensor_vals),1,nDeployments);

% Initialize empty matrices for crlb, variance, mse, and bias. The parameters
% are organized such that alpha is index 1, t0 is index 2.
% The dimensions of crlb are: 1 x agent snr x channel snr x parameter x scheme x num sensors x deployments.
crlb = zeros(num_strats,length(agent_db_values),size(channel_db_values,2),n_params,length(selected_schemes),sensor_dimension,nDeployments);
empirical_var = crlb;
empirical_mse = crlb;
empirical_bias = crlb;
old_crlb = zeros(1,length(agent_db_values),size(channel_db_values,2),n_params,length(selected_schemes),sensor_dimension,nDeployments);

empirical_total_energy = zeros(num_strats,1,1,nDeployments);
empirical_sensor_energy = zeros(num_strats,max(sensor_vals),1,nDeployments);

clean_y_arr = zeros(num_strats,K,1,nDeployments);

disp("Starting runtime...")
% Start run timer.
loopTic = tic;

%%
% Iterate through agent snr values.
for agent_db_idx = 1:length(agent_db_values)
    agent_db = agent_db_values(agent_db_idx);

    % Set sensor noise psd.Note the factor of 1/dt, which is equivalent to
    % applying an anti-aliasing filter.
    w_psd_constant = 1;
    gamma_w = (dt/w_psd_constant) * (Ps_tilde * E_mi_sqrd / util_functions.db2magTen(agent_db));
    scaled_w = sqrt(gamma_w * w_psd_constant / dt) * all_w; % --> variance of this should be gamma_w/dt
    scaled_w_psd_constant = gamma_w * w_psd_constant;
    scaled_w_psd_array(agent_db_idx) = scaled_w_psd_constant;

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
            gi = all_gi(:,1:S,:,:);

            if contains(experiment,  "imperfect_tpi")
                tpi = all_tpi(:,1:S,:,:);
            else
                tpi = 0;
            end

            % Calculate quantized phase.
            M = 4;
            [qgi,~] = util_functions.quantize_comp(gi,M);

            clean_unified_xi = alpha_true .* (mi .* util_functions.sensor_signal(t-t0_true, Tp, norm_fact));
            noisy_unified_xi = clean_unified_xi + scaled_w(:,1:S,:,:);

            [~, ui_1] = util_functions.mf_integral_fft(noisy_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), 1, K, dt, Tp);
            ui = pagetranspose(ui_1);
            
            E_uit = pagetranspose(sum(mean(ui,3),2)) * dt;
            E_uit_sqrd = pagetranspose(sum(mean(ui.^2,3),2)) * dt;

            if contains(experiment, "energy_opt")
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

                % constants for energy constraint calculation
                % approx_C1i = E_alpha_sqrd .* mi.^4 .* int_Rs0_sqrd + scaled_w_psd_constant .* int_gamma_i;
                % approx_C2i = E_alpha .* mi.^2 .* int_Rs0;
                % approx_Ei_max = approx_C1i;
                % approx_E_max = sum(approx_Ei_max,2);

                % omniscient assumption (hypothetical upper bound on energy)
                omni_C1i = alpha_true^2 .* mi.^4 .* int_Rs0_sqrd + scaled_w_psd_constant .* int_gamma_i; % E[ui(t)^2]
                omni_C2i = alpha_true .* mi.^2 .* int_Rs0; % E[ui(t)]
                omni_Ei_max = omni_C1i;
                omni_E_total = sum(omni_Ei_max,2);

                % C1i = approx_C1i;
                % C2i = approx_C2i;
                % Ei_max = approx_Ei_max;
                % E_max = approx_E_max;

                C1i = omni_C1i;
                C2i = omni_C2i;
                Ei_max = omni_Ei_max;
                E_total = omni_E_total;

                Ci_tilde = C1i - (C2i.^2)/T0;
                Ai_affine = sqrt(Ei_max ./ Ci_tilde);
                Ai_linear = sqrt(Ei_max ./ C1i);
            end

            % Iterate through channel snr values.
            for channel_db_idx = 1:size(channel_db_values,2)
                % Iterate through schemes.
                for scheme_idx = 1:length(selected_schemes)
                    scheme = selected_schemes(scheme_idx);

                    % Print statement for at-a-glance performance.
                    disp('')
                    disp("=== " + scheme + " ===")
                    disp("** Agent SNR = " + (agent_db_values(agent_db_idx)) + " dB | Channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB | S = " + S + ", Dropout = " + dropout_vals(dropout_idx) + " **")

                    % Define channel gains and server noise psd.
                    if scheme == "MPC"
                        ci = ones(size(gi));
                        gamma_n = dt * Ps_tilde * E_mi_sqrd / util_functions.db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "EPC"
                        ci = abs(gi);
                        gamma_n = dt * Ps_tilde * E_mi_sqrd * E_mag_g_sqrd / util_functions.db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "PPC"
                        ci = qgi;
                        gamma_n = dt * Ps_tilde * E_mi_sqrd * E_mag_g_sqrd / util_functions.db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "NPC"
                        ci = gi;
                        gamma_n = dt * Ps_tilde * E_mi_sqrd * E_mag_g_sqrd / util_functions.db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    end

                    % individual energy constraint
                    
                    % end individual energy constraint

                    % Apply dropout (by setting dropout sensor channels to
                    % zero).
                    if num_dropout > 0
                        dropout_elements = which_dropout(:,1:num_dropout);
                        for item = 1:nDeployments
                            ci(1,dropout_elements(item,:),1,item) = 0;
                            mi(1,dropout_elements(item,:),1,item) = 0;
                        end
                    end

                    % Set server noise psd. Note the factor of 1/dt which is
                    % equivalent to applying an anti-aliasing filter.
                    n_psd_constant = 1;
                    scaled_n = sqrt( (gamma_n*n_psd_constant/dt) / 2 ) * n;
                    % Define psd function of n in time domain (td).
                    % n_psd_function_td = gamma_n/dt;
                    scaled_n_psd_constant = gamma_n * n_psd_constant;
                    scaled_n_psd_array(channel_db_idx) = scaled_n_psd_constant;

                    %% Optimization

                    % Compute CRLB symbolically
                    if scheme == "EPC"
                        Bee = pi/Tp;
                        mag_sqr_S0_internal = @(w) (norm_fact^2) * (2*Bee.^2 .* (1 + cos(w.*pi./Bee)) ) ./ (w.^2 - Bee.^2).^2;
                        mag_sqr_S0 = @(w) util_functions.zeroIfnan(mag_sqr_S0_internal(w)) + (w == Bee | w == -Bee) .* (mag_sqr_S0_internal(w-epsilon) + mag_sqr_S0_internal(w+epsilon))./2;

                        % define colored noise spectrum V(w)
                        V_arr_internal = @(w, ai) scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0_internal(w) + scaled_n_psd_constant/2;
                        V_arr = @(w, ai) util_functions.zeroIfnan(V_arr_internal(w, ai)) + (w == Bee | w == -Bee) .* (V_arr_internal(w-epsilon, ai) + V_arr_internal(w+epsilon, ai))./2;
                        V_indiv = @(w, ai, dep_idx) scaled_w_psd_constant .* sum(ci(:,:,:,dep_idx).^2 .* ai.^2 .* mi(:,:,:,dep_idx).^2, 2) .* mag_sqr_S0(w) + scaled_n_psd_constant/2;

                        H_arr_internal = @(w, ai) 1./(scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0_internal(w) + scaled_n_psd_constant/2);
                        H_arr = @(w, ai) util_functions.zeroIfnan(H_arr_internal(w, ai)) + (w == Bee | w == -Bee) .* (H_arr_internal(w-epsilon, ai) + H_arr_internal(w+epsilon, ai))./2;

                        % use these to compute crlb
                        alpha_objective_array = @(ai) 1./( ((sum(ci .* ai .* mi.^2)).^2 / (2*pi)) .* integral(@(w) (mag_sqr_S0(w)).^2 / V_arr(w,ai), -Inf, Inf, 'ArrayValued', true));
                        t0_objective_array = @(ai) 1./( (alpha_true^2 .* (sum(ci .* ai .* mi.^2)).^2 / (2*pi)) .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_arr(w,ai), -Inf, Inf, 'ArrayValued', true));

                        % use these as objective function for optimization
                        eig_alpha_indiv = @(x, dep_idx) ( ((sum(ci(:,:,:,dep_idx) .* x(1:max(sensor_vals)) .* mi(:,:,:,dep_idx).^2)).^2 / (2*pi)) .* integral(@(w) (mag_sqr_S0(w)).^2 / V_indiv(w, x(1:max(sensor_vals)), dep_idx), -Inf, Inf, 'ArrayValued', true));
                        eig_t0_indiv = @(x, dep_idx, arb_alpha) ((arb_alpha^2 .* (sum(ci(:,:,:,dep_idx) .* x(1:max(sensor_vals)) .* mi(:,:,:,dep_idx).^2)).^2 / (2*pi)) .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_indiv(w, x(1:max(sensor_vals)), dep_idx), -Inf, Inf, 'ArrayValued', true));

                        alpha_objective_indiv = @(x, dep_idx) 1./eig_alpha_indiv(x, dep_idx);
                        t0_objective_indiv = @(x, dep_idx, arb_alpha) 1./eig_t0_indiv(x, dep_idx, arb_alpha);

                        % % use these to compute fim
                        % alpha_int = @(x) integral(@(w) (mag_sqr_S0(w)).^2 / V_arr(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true);
                        % t0_int = @(x) alpha_true^2 .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V_arr(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true);
                        % cross_int = @(x) alpha_true .* integral(@(w) w .* (mag_sqr_S0(w)).^2 / V_arr(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true);
                        % fisher_mat = @(x) ((sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) .* [alpha_int(x) cross_int(x); cross_int(x) t0_int(x)];
                        %
                        % fim = squeeze(fisher_mat(ones(1,10)));
                        % inv_fim = pageinv(fim);

                        % Compute CRLB values
                        % crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_fisher_mat(1,1,:);
                        % crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_fisher_mat(2,2,:);
                    end

                    for strat_idx = 1:num_strats
                        strat = all_strats(strat_idx);
                        total_opt_start = tic;
                        
                        %% total energy constraint
                        if strat == "sum linear homog. surr. approx."
                            opt_ai = sqrt(E_total ./ sum(C1i,2));
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                            opt_coeffs(strat_idx,S+1:end,:,:) = 0;

                        elseif strat == "sum affine homog. surr. approx."
                            opt_ai = sqrt(E_total ./ sum(Ci_tilde, 2));
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                            opt_coeffs(strat_idx,S+1:end,:,:) = opt_ai .* C2i ./ T0;

                        elseif strat == "sum linear hetero. surr. approx."
                            opt_ai = (1./ci) .* sqrt(E_total ./ sum(C1i ./ ci.^2, 2));
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai;
                            opt_coeffs(strat_idx,S+1:end,:,:) = 0;

                        elseif strat == "sum affine hetero. surr. approx."
                            opt_ai = (1./ci) .* sqrt(E_total ./ sum(Ci_tilde ./ ci.^2, 2));
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai;
                            opt_coeffs(strat_idx,S+1:end,:,:) = opt_ai .* C2i ./ T0;
                        %%% end total energy constraint
                        
                        %% individual energy constraint
                        elseif strat == "indiv. linear homog. surr. approx."
                            opt_ai = min(Ai_linear);
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                            opt_coeffs(strat_idx,S+1:end,:,:) = 0;

                        elseif strat == "indiv. affine homog. surr. approx."
                            opt_ai = min(Ai_affine);
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                            opt_coeffs(strat_idx,S+1:end,:,:) = opt_ai .* C2i ./ T0;
                        
                        elseif strat == "indiv. linear hetero. surr. approx."
                            [Gamma_linear, ~] = min(ci .* Ai_linear);
                            opt_ai = Gamma_linear ./ ci;
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                            opt_coeffs(strat_idx,S+1:end,:,:) = 0;
                            
                        elseif strat == "indiv. affine hetero. surr. approx."
                            [Gamma_affine, ~] = min(ci .* Ai_affine);
                            opt_ai = Gamma_affine ./ ci;
                            opt_coeffs(strat_idx,1:S,:,:) = opt_ai .* ones(size(opt_coeffs(strat_idx,1:S,:,:)));
                            opt_coeffs(strat_idx,S+1:end,:,:) = opt_ai .* C2i ./ T0;
                        %%% individual energy constraint

                        %% fmincon
                        elseif contains(strat, "fmincon")
                            % Define parameters for fminsearch
                            num_vars = S; % Number of variables
                            x0 = [ones(1,num_vars)]; % Initial guesses for ai
                            lb = zeros(1,num_vars); % ai >= 0
                            ub = []; % unrestricted

                            options = optimoptions('fmincon','Display','off','Algorithm','sqp');
                            
                            opt_coeffs_cell = cell(1, nDeployments);
                            
                            parfor dep_idx = 1:nDeployments
                                local_coeffs = zeros(1,size(opt_coeffs,2));
                                nonlcon = @(x) scsh_constraints(x, C1i(:,:,:,dep_idx), C2i(:,:,:,dep_idx), T0, E_total(:,:,:,dep_idx), strat, D);
                                
                                % compute optimal ai and bi for sas using weighted sum objective function
                                
                                for jj = 1:length(alpha_weight_vals)
                                    alpha_weight = alpha_weight_vals(jj);
                                    t0_weight = 1 - alpha_weight;

                                    weighted_sum_objective = @(x) alpha_weight .* alpha_objective_indiv(x, dep_idx) + t0_weight .* t0_objective_indiv(x, dep_idx, 1);
                                    % weighted_sum_objective = @(x) -(alpha_weight .* eig_alpha_indiv(x, dep_idx) + t0_weight .* eig_t0_indiv(x, dep_idx, 1));

                                    [weighted_sum_sas_x_opt, ~] = fmincon(weighted_sum_objective, x0, [], [], [], [], lb, ub, nonlcon, options);
                                    % store ai
                                    % opt_coeffs(strat_idx,1:S,:,dep_idx) = weighted_sum_sas_x_opt;
                                    local_coeffs(1:S) = weighted_sum_sas_x_opt
                                    local_coeffs(S+1:end) = local_coeffs(1:S) .* C2i(:,:,:,dep_idx) ./ T0;
                                    
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

                        ai = opt_coeffs(strat_idx, 1:S, :, :);
                        bi = opt_coeffs(strat_idx, S+1:end, :, :);
                        
                        %% compute energy
                        deployment_sensor_energy = ai.^2 .* E_uit_sqrd - 2 .* ai .* bi .* E_uit + bi.^2 * T0;
                        empirical_sensor_energy(strat_idx,:,:,:) = deployment_sensor_energy;
                        indiv_energy_nae = mean(abs(Ei_max - deployment_sensor_energy) ./ Ei_max, "all");
                        disp(strat + " Indiv. Energy NAE: " + num2str(indiv_energy_nae))

                        deployment_total_energy = sum(ai.^2 .* E_uit_sqrd - 2 .* ai .* bi .* E_uit + bi.^2 * T0, 2);
                        empirical_total_energy(strat_idx,:,:,:) = deployment_total_energy;
                        total_energy_nae = mean(abs(squeeze(E_total)-squeeze(deployment_total_energy))./squeeze(E_total));
                        disp(strat + " Total Energy NAE: " + num2str(total_energy_nae))
                        fprintf("\n")
                        %%% end compute energy

                        total_opt_time = toc(total_opt_start);
                        disp("Total " + strat + " time: " + floor(total_opt_time/60) + " minutes " + mod(total_opt_time,60) + " seconds")
                    end
                    disp("");

                    % guarantee same energy usage, always scale to hetero.
                    % coefficients because they consider CRLB obj. funct.
                    idx1 = find(all_strats == "indiv. linear homog. surr. approx.");
                    idx2 = find(all_strats == "indiv. linear hetero. surr. approx.");
                    il_homog_scaler = empirical_total_energy(idx2,:,:,:) ./ empirical_total_energy(idx1,:,:,:);

                    idx3 = find(all_strats == "indiv. affine homog. surr. approx.");
                    idx4 = find(all_strats == "indiv. affine hetero. surr. approx.");
                    ia_homog_scaler = empirical_total_energy(idx4,:,:,:) ./ empirical_total_energy(idx3,:,:,:);

                    %% Estimation
                    total_est_start = tic;
                    for strat_idx = 1:num_strats
                        strat = all_strats(strat_idx);

                        if strat == "indiv. linear homog. surr. approx."
                            opt_coeffs(strat_idx, 1:S, :, :) = opt_coeffs(strat_idx, 1:S, :, :) .* il_homog_scaler;
                        elseif strat == "indiv. affine homog. surr. approx."
                            opt_coeffs(strat_idx, 1:S, :, :) = opt_coeffs(strat_idx, 1:S, :, :) .* ia_homog_scaler;
                            opt_coeffs(strat_idx,S+1:end,:,:) = opt_coeffs(strat_idx, 1:S, :, :) .* C2i ./ T0;
                        end
                        
                        ai = opt_coeffs(strat_idx, 1:S, :, :);
                        bi = opt_coeffs(strat_idx, S+1:end, :, :);

                        [clean_cws_2, ~] = util_functions.mf_integral_fft(ai .* clean_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), ci, K, dt, Tp);
                        clean_y_arr(strat_idx,:,:,:) = clean_cws_2 - sum(ci .* bi, 2);

                        % Compute convolution using fft
                        [cws_2, ~] = util_functions.mf_integral_fft(ai .* noisy_unified_xi, mi .* util_functions.sensor_signal(t-tpi, Tp, norm_fact), ci, K, dt, Tp);
                        cws = pagetranspose(cws_2);
                        
                        % Generate y(t).
                        y = cws - sum(ci .* bi, 2) + pagetranspose(scaled_n);

                        % Search for t0 from index 1 to index of t0_max/dt + 1.
                        % Since we know a priori that the estimate of t0 cannot
                        % exceed t0_max. If we don't impose this restriction then we
                        % could select a t0 value such that the joint estimate for
                        % alpha = 0/0, which will result in NaN.

                        y_searchable = y(:,1:(floor(t0_max/dt) + 1),:,:);
                        if exp_split(1) == "nae" || scheme == "MPC" || scheme == "EPC"
                            [~,I] = max(abs(real(y_searchable)), [], 2);
                        else
                            [~,I] = max(abs(y_searchable), [], 2);
                        end
                        t0_estimates_for_plot = (I-1)*dt;

                        % Set t0 estimates to true or estimated values depending on
                        % experiment.
                        if experiment == "cwe_disjoint" || experiment == "nae_disjoint" || experiment == "cwe_disjoint_energy_opt"
                            t0_estimates = t0_true*ones(1,1,nTrials,nDeployments);
                        else
                            t0_estimates = t0_estimates_for_plot;
                        end

                        % Compute alpha estimates for AC-NAE
                        if exp_split(1) == "nae"
                            if exp_split(2) == "joint"
                                % Select estimated peak index in vectorized manner.
                                II = squeeze(I);
                                % Generate the subscript indices for the 3rd and 4th
                                % dimensions.
                                [dim3, dim4] = ndgrid(1:size(y, 3), 1:size(y, 4));

                                % Create linear indices for the desired elements.
                                linear_indices = sub2ind(size(y), ones(size(II)), II, dim3, dim4);

                                % Extract the elements from y.
                                peak_y_vals = y(linear_indices);
                                peak_y_vals = reshape(peak_y_vals,size(I));
                            elseif exp_split(2) == "disjoint"
                                % Compute y(t0). We have the index of what we think
                                % is the peak value of y (where all the matched
                                % filter peaks should line up).
                                
                                %%% need to update this because it uses a
                                %%% deprecated version of matched filter
                                % Generate si(t-t0).
                                si_true = mi .* util_functions.sensor_signal(t-ti-t0_true, Tp, norm_fact); % K x S
                                % Generate xi(t).
                                xi = alpha_true * si_true + scaled_w(:,1:S,:,:); % K x S x nTrials x nDeployments
                                [cws_known_t0, ~] = reshape(util_functions.matched_filter_integral(reshape(xi,K,S,1,nTrials,nDeployments), reshape(mi .* util_functions.sensor_signal(t-ti-tpi-repmat(t0_true,1,1,K), Tp, norm_fact), K,S,K,1,nDeployments), ci, K, S, nTrials, nDeployments, dt, Tp, mi, ti),1,K,nTrials,nDeployments);
                                %%%

                                % All column idx in t0_cws are the same currently,
                                % hence 2nd idx = 1.
                                peak_y_vals =  cws_known_t0(:,1,:,:) + pagetranspose(scaled_n(1,1,:,:));
                            end
                            % Compute Ei.
                            Ei = sum(abs(si_true).^2 .* dt,1); % 1 x S
                            % compute alpha estimates
                            alpha_estimates =  real(peak_y_vals) ./ sum(real(ci) .* Ei,2);

                        % Compute alpha for AC-CWE.
                        elseif exp_split(1) ~= "nae"
                            % Define frequency domain parameters
                            fs = 1/dt;
                            L = K-1;
                            f = 0:fs/(2*L):fs/2;
                            omega = 2*pi*f;
                            % Check if there are any indeterminate values (will
                            % depend on the form of si(t) but in this case we
                            % calculated this ahead of time).
                            indet_idx = find(abs(round(f,4)) == 0.5);

                            % This section of code involves implementing the
                            % expressions for the decorrelating and whitening
                            % filters.

                            % Compute alpha estimates for AC-CWE with partial or no
                            % phase compensation.
                            if (scheme == "PPC" || scheme == "NPC")
                                real_channel = real(ci);
                                imag_channel = imag(ci);

                                % Compute polynomial to find coefficients for
                                % decorrelation filter.
                                b = sum(real_channel.*imag_channel.*mi.^2,2) ./ (sum(mi.^2 .* (real_channel.^2 - imag_channel.^2),2));
                                disc = 1 + 4.*b.^2;
                                a1 = (-1 + sqrt(disc)) ./ (2*b);
                                a2 = (-1 - sqrt(disc)) ./ (2*b);
                                % Initially choose a = a1, but a future ablation
                                % study could certainly be conducted to determine if
                                % one a1 or a2 performs better. Arguably both will
                                % provide us with the required filter coefficients
                                a = a1;

                                % Define frequency domain expression for s0(t).
                                SEs = ones(1,K,1,nDeployments) .* (2 * pi^2 * (1+cos(omega)) ./ (omega.^2 - pi^2).^2);
                                Gamma = 1;

                                % Define magnitude squared of SHdI.
                                mag_sqr_SHdI = 1 ./ (Gamma + SEs);

                                % Replace indeterminate values using L'Hopital's
                                % rule, also will depend on the form of si(t).
                                for idx=indet_idx
                                    % disp('replacing indeterminate values')
                                    % Fill indeterminate values using L'Hopital's
                                    % rule.
                                    w0 = pi;
                                    mag_sqr_SHdI(1,idx,:) = 8*w0^2 / (Gamma*8*w0^2 - 2*pi^2*cos(w0));
                                end

                                % Define magnitude of SHdI.
                                mag_SHdI = sqrt(mag_sqr_SHdI);

                                % Get time domain matrix for hdI.
                                [hdI, hdI_matrix] = util_functions.get_time_domain(mag_SHdI,dt,nDeployments);

                                % Get time domain matrix for hdR.
                                hdR_matrix = reshape(a,1,1,nDeployments) .* hdI_matrix;

                                % Create time domain matrix for hd.
                                hd_matrix = hdR_matrix + 1i*hdI_matrix;

                                % Compute pR and qR.
                                pR = scaled_w_psd_constant * sum(mi.^2 .* (a.*real_channel - imag_channel).^2, 2);
                                qR = (scaled_n_psd_constant/2 * (a.^2 + 1)) - Gamma*pR;

                                % Compute pI and qI.
                                pI = scaled_w_psd_constant * sum(mi.^2 .* (real_channel + a.*imag_channel).^2, 2);
                                qI = (scaled_n_psd_constant/2 * (a.^2 + 1)) - Gamma*pI;

                                % Compute SKndR and SKndI.
                                SKndR = pR + mag_sqr_SHdI .* qR;
                                SKndI = pI + mag_sqr_SHdI .* qI;

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

                                % Compute closed-form expressions for alpha

                                % Compute convolution using fft
                                [aaa_2, ~] = pagetranpose(util_functions.mf_integral_fft(mi .* util_functions.sensor_signal(t-t0_estimates, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, K, dt, Tp));
                                aaaR = real(aaa_2);
                                aaaI = imag(aaa_2);
                                
                                % The d here is for decorrelation, not derivative.
                                aaadR = dt*(pagemtimes(aaaR,reshape(hdR_matrix,K,K,1,nDeployments)) - pagemtimes(aaaI,reshape(hdI_matrix,K,K,1,nDeployments)));
                                aaadI = dt*(pagemtimes(aaaI,reshape(hdR_matrix,K,K,1,nDeployments)) + pagemtimes(aaaR,reshape(hdI_matrix,K,K,1,nDeployments)));

                                yR = real(y);
                                yI = imag(y);

                                ydR = dt*(pagemtimes(yR,reshape(hdR_matrix,K,K,1,nDeployments)) - pagemtimes(yI,reshape(hdI_matrix,K,K,1,nDeployments)));
                                ydI = dt*(pagemtimes(yI,reshape(hdR_matrix,K,K,1,nDeployments)) + pagemtimes(yR,reshape(hdI_matrix,K,K,1,nDeployments)));

                                resh_QnR_matrix = reshape(QnR_matrix,length(t),length(t),1,nDeployments);
                                resh_QnI_matrix = reshape(QnI_matrix,length(t),length(t),1,nDeployments);

                                num1 = pagemtimes(pagemtimes(ydR,resh_QnR_matrix),pagetranspose(aaadR));
                                num2 = pagemtimes(pagemtimes(ydI,resh_QnI_matrix),pagetranspose(aaadI));
                                num = dt*dt*(num1+num2);

                                denom1 = pagemtimes(pagemtimes(aaadR,resh_QnR_matrix),pagetranspose(aaadR));
                                denom2 = pagemtimes(pagemtimes(aaadI,resh_QnI_matrix),pagetranspose(aaadI));
                                denom = dt*dt*(denom1+denom2);
                            else
                                mag_sqr_H = H_arr(omega, ai);

                                % Get time domain matrix for Qn.
                                [~,Qn_matrix] = util_functions.get_time_domain(mag_sqr_H,dt,nDeployments);

                                % Compute convolution using fft
                                [Omega_2, ~] = util_functions.mf_integral_fft(ai .* mi .* util_functions.sensor_signal(t-t0_estimates, Tp, norm_fact), mi .* util_functions.sensor_signal(t, Tp, norm_fact), ci, K, dt, Tp);
                                num = dt*dt*pagemtimes(pagemtimes(real(y) + sum(ci .* bi, 2), reshape(Qn_matrix,length(t),length(t),1,nDeployments)), Omega_2);
                                denom = dt*dt*pagemtimes(pagemtimes(pagetranspose(Omega_2),reshape(Qn_matrix,length(t),length(t),1,nDeployments)), Omega_2);
                            end
                            alpha_estimates = num ./ denom;
                        end

                        % Compute empirical variance.
                        empirical_var(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = var(alpha_estimates,0,3);
                        empirical_var(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = var(t0_estimates_for_plot,0,3);

                        % Compute empirical mse.
                        empirical_mse(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = mean((alpha_estimates - alpha_true).^2,3);
                        empirical_mse(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean((t0_estimates_for_plot - t0_true).^2,3);

                        % Compute empirical bias.
                        empirical_bias(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = mean(alpha_estimates,3);
                        empirical_bias(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = mean(t0_estimates_for_plot,3);

                        %%%%%%%%%%%%%%%%% CRLB %%%%%%%%%%%%%%%%%

                        if scheme == "EPC"
                            crlb(strat_idx,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = alpha_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3));
                            crlb(strat_idx,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = t0_objective_array(util_functions.unsqueeze(opt_coeffs(strat_idx, 1:S, :),3));
                        end
                    end
                    total_est_time = toc(total_est_start);
                    disp("Total ML estimation time: " + floor(total_est_time/60) + " minutes " + mod(total_est_time,60) + " seconds")
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
    avg_dep_var = empirical_var;
    avg_dep_mse = empirical_mse;
    avg_dep_crlb = crlb;
    avg_dep_bias = empirical_bias;

    % avg_dep_old_crlb = old_crlb;
    % avg_dep_my_var = my_var;
else
    avg_dep_var = mean(empirical_var,length(size(empirical_var)));
    avg_dep_mse = mean(empirical_mse,length(size(empirical_mse)));
    avg_dep_crlb = mean(crlb,length(size(crlb)));

% normalize magnitude metrics
    avg_dep_var(:,:,:,1) = avg_dep_var(:,:,:,1);
    avg_dep_mse(:,:,:,1) = avg_dep_mse(:,:,:,1);
    avg_dep_crlb(:,:,:,1) = avg_dep_crlb(:,:,:,1);  

    % normalize time delay metrics
    avg_dep_var(:,:,:,2) = avg_dep_var(:,:,:,2) ./ T0^2;
    avg_dep_mse(:,:,:,2) = avg_dep_mse(:,:,:,2) ./ T0^2;
    avg_dep_crlb(:,:,:,2) = avg_dep_crlb(:,:,:,2) ./ T0^2;

    avg_dep_bias = mean(empirical_bias,length(size(empirical_bias)));
    % avg_dep_old_crlb = mean(old_crlb,length(size(old_crlb)));
end

%% Save experiment results.
% if ~any(experiment == experiment_list)
%     disp("Unknown experiment specified. Results not saved!")
% else
%     files_to_save = ["avg_dep_var", "avg_dep_mse", "avg_dep_bias", "avg_dep_crlb",...
%         "agent_db_values", "selected_schemes", "dropout_vals", "sensor_vals", "channel_db_values",...
%         "experiment","exp_split","sensor_dimension", "save_images", "K", "nDeployments"];
%     if experiment == "cwe_disjoint_energy_opt"
%         files_to_save = [files_to_save, "constraints", "transforms", "coeff_types", "approaches", "all_strats", "num_strats", "opt_coeffs", "alpha_weight_vals", "save_images", ...
%             "A", "T0", "Tp", "alpha_true", "t0_true"];
%     end
%     save(constraints(1) + "_" + experiment + "_results.mat", files_to_save{:})
% end

%% Define plot parameters.
params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];

% names = {'baseline', 'sum baseline', 'sum surr. approx.', 'sum fmincon', 'indiv. baseline', 'indiv. surr. approx.', 'indiv. fmincon', 'mixed baseline', 'mixed surr. approx.', 'mixed fmincon'};
names = {'linear homog.', 'linear hetero.', 'affine homog.', 'affine hetero.'};
colors = {
    ... --- sum colors (blue gradient) ---
    '#A2142F', ...  % Tableau Red
    '#70ED5A', ...  % Light Green
    '#FFAE00', ...  % Amber / Yellow-Orange
    '#114FF7', ...  % Vivid Blue
    '#11E0F7', ...  % Bright Cyan
    ... --- indiv. colors (green gradient) ---
    '#004236', ...
    '#009500', ...  % Forest Green
    '#19415E', ...  % Dark Blue
    ... --- mixed colors (orange-yellow gradient) ---
    '#FF5801', ...  % Vivid Orange
    '#E3AB6D', ...  % Sandy Orange / Tan
    };
colorMap = containers.Map(names, colors(1:length(names)));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Final AirComp Results\";

exclude = ["None"];
iter_length = length(transforms) + length(coeff_types);

%% Linear plots
if experiment ~= "cwe_joint_grid"
    for agent_db_idx = 1:length(agent_db_values)
        % Create save directory if it does not exist.

        for param_idx = 1:2
            % Collect all plot values for y-axis scaling.
            all_vals = [avg_dep_var(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                avg_dep_mse(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                avg_dep_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                % avg_dep_old_crlb(:,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                ];

            for scheme_idx = 1:length(selected_schemes)
                scheme = selected_schemes(scheme_idx);

                % fig1 = figure;
                % set(fig1,'Position',[100,100,450,450])
                % ax = gca;
                % hold on
                
                fig1 = figure;
                set(fig1,'Position',[100,100,1500,450], 'Name', params_text(param_idx) + " hat | " + constraints(1) + " | A = " + A + ", alpha = " + alpha_true + ", t0 = " + t0_true + ", Tp = " + Tp + ", T0 = " + T0, 'NumberTitle', 'off')
                tiledlayout(1,length(constraints));

                if experiment == "random_dropout"
                    count_vals = dropout_vals;
                else
                    count_vals = sensor_vals;
                end

                if scheme == "NPC"
                    count_vals = count_vals(1);
                end
    
                for constraint_idx = 1:length(constraints)
                    constraint = constraints(constraint_idx);
                    ax = nexttile;
                    hold(ax, 'on');
                    % Plot pseudoplots for legend symbols.
                    if experiment == "cwe_disjoint_energy_opt"
                        for count_idx = (constraint_idx-1)*iter_length + (1:iter_length)
                            strat = all_strats(count_idx);
                            strat_split = split(strat);
                            cmap_idx = strat_split(2) + " " + strat_split(3);
                            if contains(strat, exclude)
                                continue
                            end
                            plot(ax, nan,nan,'color',colorMap(cmap_idx),'LineWidth',2);
                        end
                    else
                        for count_idx = 1:length(count_vals)
                            plot(nan,nan,'color',color_vec(count_idx),'LineWidth',2);
                        end
                    end
    
                    plot(ax, nan,nan,'^','color','black');
                    plot(ax, nan,nan,'x','color','black');
                    plot(ax, nan,nan,'o','color','black');
    
                    x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
                    plot_line_width = 1.5;

                    for count_idx = 1:length(count_vals)
                        for strat_idx = (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                            strat = all_strats(strat_idx);
                            strat_split = split(strat);
                            cmap_idx = strat_split(2) + " " + strat_split(3);
                            if contains(strat, exclude)
                                continue
                            end
                            
                            plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',colorMap(cmap_idx),'LineWidth', plot_line_width)
                            plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',colorMap(cmap_idx),'LineWidth', plot_line_width)
                            plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',colorMap(cmap_idx),'LineWidth', plot_line_width)
                        end
                        % % plot non-symbolic crlb to validate symbolic results
                        % plot(x_axis_series,(squeeze(avg_dep_old_crlb(:,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',color_vec(3),'LineWidth', 1)
                    end
    
                    xlabel('Channel SNR (dB)')
                    ylabel(" ")

                    title('$$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')

                    ax.FontSize = 15;          % font size
                    ax.YScale = 'log';          % log scale if needed
                    grid(ax, 'on');             % turn on grid
    
                    if true % scheme == "MPC"
                        if experiment == "random_dropout"
                            legend(["Drop = " + dropout_vals,"VAR","MSE","CRLB"], 'Location', 'best')
                        elseif experiment == "cwe_disjoint_energy_opt"
                            legend(ax, [all_strats((constraint_idx-1)*iter_length + (1:iter_length))',"VAR","MSE","CRLB"], 'Location', 'best')
                        else
                            legend(["S = " + sensor_vals,"VAR","MSE","CRLB"], 'Location', 'best')
                        end
                    end
    
                    % Add box around plot axes.
                    % box(ax, 'on');
                    % ax.GridColor = [0 0 0];
                    % ax.GridAlpha = 0.5;
                    % ax.LineWidth = 1;

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
                        exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                    end
                end
            end
        end
    end
    %% Surface plots
else
    % Create save directory if it does not exist.
    save_folder = folder_path + experiment + "_" + string(agent_db_values(end)) + "_db";
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    % Toggle if you want to convert meshgrid values to log (you should).
    convert_to_log = true;

    for param_idx = 1:2
        % Collect all plot values for y-axis scaling.
        all_vals = [avg_dep_var(:,:,:,param_idx,:,1:sensor_dimension);...
            avg_dep_mse(:,:,:,param_idx,:,1:sensor_dimension);...
            avg_dep_crlb(:,:,:,param_idx,:,1:sensor_dimension)];
        if convert_to_log
            all_vals = log10(all_vals);
        end

        for scheme_idx = 1:length(selected_schemes)
            fig1 = figure;
            set(fig1,'Position',[100,100,675,475])
            ax = gca;

            count_vals = sensor_vals;

            for count_idx = 1:length(count_vals)
                z_vals = squeeze(avg_dep_var(:,:,:,param_idx,scheme_idx,count_idx,1));
                if convert_to_log
                    z_vals = log10(z_vals);
                end
                surf(agent_db_values, channel_db_values(1,:,1), z_vals)
            end

            cb = colorbar;
            colormap(turbo)
            % Change view angle to better see both x and y dimensions
            view(150,20)
            xl = xlabel('Channel SNR (dB)', 'FontWeight', 'bold');
            yl = ylabel('Sensor SNR (dB)', 'FontWeight', 'bold');


            title('$$\hat{' + params_latex(param_idx) + '}$$', 'Interpreter','latex')
            set(ax, 'FontSize', 15);

            zlabel('$$ \mathbf{\log_{10} ((\mathrm{Var} ( \hat{'+params_latex(param_idx)+'} ))}$$','Interpreter','latex', 'FontSize', 20)

            % Add box around plot axes.
            box(ax, 'on');
            grid on

            if convert_to_log
                z_lower = 1.05*min(all_vals,[],"all");
            else
                set(ax, 'ZScale', 'log')
                z_lower = 10^(-0.25)*min(all_vals,[],"all");
            end

            z_upper = 1.05*max(all_vals,[],"all");

            zlim([z_lower,z_upper]);
            clim([z_lower,z_upper])

            if ~convert_to_log
                z_ticks = power_10_range(z_lower,z_upper);
                ax.ZTick = z_ticks;
            end

            ax.GridColor = [0 0 0];
            ax.GridAlpha = 1;
            ax.LineWidth = 1;

            xpos = get(ax, 'Xlabel');
            xpos = xpos.Position;
            xpos(1) = xpos(1) + 2;
            xpos(2) = xpos(2) - 2;

            xl.Position = xpos;
            xl.Rotation = 10;

            ypos = get(ax, 'Ylabel');
            ypos = ypos.Position;
            ypos(1) = ypos(1) - 2;
            ypos(2) = ypos(2) + 2.5;

            % if scheme_idx == 1
            %     ypos(1) = ypos(1) - 5.5;
            %     ypos(2) = ypos(2) - 2.5;
            % else
            %     ypos(1) = ypos(1) - 4.5;
            %     ypos(2) = ypos(2) - 2.5;
            % end

            yl.Position = ypos;
            yl.Rotation = -30;

            axis normal

            % Save the figure.
            if save_images
                exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(end)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
            end
        end
    end
end

%% Functions
function [c, ceq] = scsh_constraints(x, C1i, C2i, T0, E_max, strat, D)
% num_vars = length(C1i
% ai = x(1:num_vars);
% bi = x(num_vars+1:end);
% ind_power_const = C1i;
% ind_power_vals = ai.^2 .* C1i - 2 .* ai .* bi .* C2i + bi.^2 .* T0;
%
% First constraint: sum(ai.^2 .* C1i - 2 .* ai .* bi .* C2i + bi.^2 .* T0) <= E_max
% sum_cond = sum(ind_power_vals) - E_max;
% indiv_cond = (ind_power_vals-ind_power_const)';

% Simplified constraint
ai = x;
indiv_energy_constraint = C1i;
indiv_energy_vals = ai.^2 .* (C1i - (C2i.^2 ./ T0));

sum_cond = @(D) sum(indiv_energy_vals) - D.*E_max;
indiv_cond = (indiv_energy_vals-indiv_energy_constraint)';

if strat == "sum fmincon"
    c = [sum_cond(1)];
% elseif strat == "sc sum fmincon"
%     c = [sum(ai.^2 .* C1i) - E_max];
elseif strat == "indiv. fmincon"
    c = indiv_cond;
elseif contains(strat, "mixed")
    % % Use a regular expression to find the number after "D="
    % numberStr = regexp(str, 'D=([0-9.]+)', 'tokens');
    % % Convert the extracted string to a numeric value
    % D_value = str2double(numberStr{1});
    D_value = D;
    c = [sum_cond(D_value); indiv_cond];
else
    error("Unknown experiment specified!")
end

ceq = [];
end
