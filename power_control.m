%%
% 07-12-2025: update power control based on feedback from QE and first
% manuscript
%%
% Clear all variables and close all exisiting figures
clear all
close all

% set a seed for reproducibility
seed = 264;
rng(seed);

% Choose which experiment to run.
% naive uses NAE, all other experiment options use CWE.
experiment_list = ["naive_disjoint", "naive_joint", "disjoint", "joint", "joint_grid", "random_dropout"];
experiment = "joint";

% Set number of deployments.
nDeployments = 1;
% Set number of trials.
nTrials = 500;
% Set the number of measurements per sensor. The sample index starts at 0 so
% sampling from index 0 to 100 requires K = 101 measurements.
K = 101;

% Set some bookkeeping variables depending on which experiment we are running.
if experiment == "random_dropout"
    dropout_vals = [0,2,4];
    agent_db_values = [0];
    sensor_dimension = length(dropout_vals);
    % only consider maximum number of sensors and drop from there
    sensor_vals = [10];
    
    which_dropout = zeros(nDeployments,max(dropout_vals));
    for idx = 1:nDeployments
        which_dropout(idx,:) = randperm(max(sensor_vals),max(dropout_vals));
    end
else
    if experiment == "joint_grid"
        sensor_vals = [5];
        agent_db_values = [0,5,10,15,20];
    else
        sensor_vals = [5,10];
        agent_db_values = [0];
    end
    dropout_vals = 0;
    sensor_dimension = length(sensor_vals);
end

% Parse experiment name for organization.
exp_split = strsplit(experiment,'_');

% Set observation window length in seconds.
T_obs = 5;
T0 = T_obs;

% Define event signal parameters.
T_s = 1;
w0 = pi;

% Define time vector
dt = T_obs/(K-1);
fs = 1/dt;
t = dt*(0:K-1)';

% Calculate the signal energy (such that the signal is assumed to be within the
% interioir fo the observation period).
func = @(t) abs(sensor_signal(t)).^2;
P_s = integral(func,0,T_obs)/T_obs;

% Define true parameter values.
alpha_true = 2;
t0_true = 1;

% Define range for distribution of alpha and t0
upper_alpha = 3;
lower_alpha = 1;

upper_t0 = 2;
lower_t0 = 0;

% range for distribution of m
max_mi = 1;
min_mi = 0.5;

max_ti = 1;
min_ti = 0;

% maximum t0 value to guarantee the signal is in the interior of the
% observation period
t0_max = T_obs-T_s-max_ti;

% noise dimensions: K x S x Trials x Deployments

% Generate sensor noise with dimensions: K x S x trials x deployments.
all_w = randn(K,max(sensor_vals),nTrials,nDeployments);
% Generate server noise.
n = (randn(K,1,nTrials,nDeployments) + 1i*randn(K,1,nTrials,nDeployments));
% Generate channel gains.
all_gi = (randn(1,max(sensor_vals),1,nDeployments) + 1i*randn(1,max(sensor_vals),1,nDeployments))/sqrt(2);

% Define Rayleigh distribution parameters.
rayleigh_factor = 1/sqrt(2);
E_mag_g_sqr = 2*rayleigh_factor^2;

% Choose which schemes we want to plot. Note that although we still generate the
% results for NPC, we do not necessarily plot them. there is no gurantee that
% more sensors will perform better for NPC due to the uncompensated channel
% phase.
selected_schemes = ["EPC"];

% Generate mi and ti.
all_mi = unifrnd(min_mi,max_mi,1,max(sensor_vals),1,nDeployments);
all_ti = unifrnd(min_ti,max_ti,1,max(sensor_vals),1,nDeployments);

% Define expected value of mi^2.
E_mi_sqr = (1/12) * (max_mi-min_mi)^2 + 0.5*(min_mi+max_mi);

% Define channel snr values.
channel_snr = [0,5,10,15,20];
% Duplicate channel snr matrices for experiments.
channel_db_values = repmat(channel_snr, length(selected_schemes), 1, length(agent_db_values));

% Initialize empty matrices for crlb, variance, mse, and bias. The parameters
% are organized such that alpha is index 1, t0 is index 2. The dimensions of
% crlb are: 1 x agent snr x channel snr x parameter x scheme x num sensors x
% deployments.
crlb = zeros(1,length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),sensor_dimension,nDeployments);
% CRLB computed using anonymous function in MATLAB
anon_crlb = crlb;
% scale and shift (ai and bi)
sas_crlb = crlb;
% sas but use f(alpha) + f(t0) as objective function
combined_sas_crlb = crlb;
sas_alpha_crlb = crlb;
sas_t0_crlb = crlb;

C_weighted_values = [1];
sas_weighted_sum_crlb = zeros(length(C_weighted_values),length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),sensor_dimension,nDeployments);

% outline strategies
temp_legend = strings(1,length(C_weighted_values));
for jj = 1:length(C_weighted_values)
    temp_legend(jj) = "CW=" + floatToFraction(C_weighted_values(jj));
end
all_strats = ["normal", "alpha", "t0", temp_legend];
num_strats = length(all_strats);

% Initialize performance metric matrices for each strategy
empirical_var = zeros(1,length(agent_db_values),size(channel_db_values,2),2,length(selected_schemes),sensor_dimension,nDeployments,num_strats);
empirical_mse = empirical_var;
bias = empirical_mse;

scale_opt = zeros(2,2*max(sensor_vals),nDeployments);
sas_opt = zeros(2,2*max(sensor_vals),nDeployments);
opt_coeffs = zeros(num_strats,2*max(sensor_vals),nDeployments);

% pre-allocate optimal coefficient vectors
sas_alpha_opt = zeros(2,2*max(sensor_vals),nDeployments);
sas_t0_opt = sas_alpha_opt;
sas_weighted_sum_opt = sas_alpha_opt;

% Compute expected values for alpha and t0
E_alpha_sqr = (upper_alpha^3 - lower_alpha^3)/(3*(upper_alpha-lower_alpha));
E_alpha = (lower_alpha+upper_alpha)/2;
E_t0 = (lower_t0+upper_t0)/2;

% start run timer
loopTic = tic;

% iterate through sensor snr values
for agent_db_idx = 1:length(agent_db_values)
    agent_db = agent_db_values(agent_db_idx);    
    
    % Set sensor noise power.Note the factor of 1/dt, which is equivalent to
    % applying an anti-aliasing filter.
    w_psd_constant = 1;
    gamma_w = (dt/w_psd_constant) * (P_s / db2magTen(agent_db));
    scaled_w = sqrt(gamma_w * w_psd_constant /dt) * all_w; % --> variance of this should be gamma_w/dt
    scaled_w_psd_constant = gamma_w * w_psd_constant;

    % iterate through sensor values
    for sensor_idx = length(sensor_vals)
        S = sensor_vals(sensor_idx);

        for dropout_idx = 1:length(dropout_vals)
            num_dropout = dropout_vals(dropout_idx);
            
            % Select subset of mi, ti, and gi values.
            mi = all_mi(:,1:S,:,:);
            ti = all_ti(:,1:S,:,:);
            gi = all_gi(:,1:S,:,:);

            % calculate quantized phase compensation
            M = 4;
            [qg,qg_cq] = quantize_comp(gi,M);

            % Generate si(t-t0).
            si_true = mi .* sensor_signal(t - ti - t0_true); % K x S
            % Generate xi(t).
            xi = alpha_true * si_true + scaled_w(:,1:S,:,:); % K x S x nTrials x nDeployments
            % Compute Ei.
            Ei = sum(abs(si_true).^2 .* dt,1); % 1 x S

            % Define anonymous functions to speed up CRLB calculation for each
            % iteration of numerical optimization.

            % define underlying signal
            s0 = @(tau) (heaviside(tau)-heaviside(tau-1)) .* sin(pi .* tau);
            si = @(tau) mi .* s0(tau-ti);
            % calculte m(t)
            mt = @(t) integral(@(u) s0(u-E_t0) .* s0(u-t), 0, T0);
            % this is going to be calculated for each agent i, but it should be the same
            int_mt = integral(@(t) mt(t), 0, T0, 'ArrayValued', true);
            int_mt_sqr = integral(@(t) mt(t).^2, 0, T0, 'ArrayValued', true);
            
            gamma_i = @(t) integral(@(t1) si(t1-ti-t).^2, 0, T0, 'ArrayValued', true);
            int_gamma_i = integral(@(t) gamma_i(t), 0, T0, 'ArrayValued', true);

            % constants for energy constraint calculation
            C1i = mi.^4 .* E_alpha_sqr .* int_mt_sqr + scaled_w_psd_constant .* int_gamma_i;
            C2i = mi.^2 .* E_alpha .* int_mt;

            P_max = sum(C1i)/T_obs;

            individual_power = C1i/T_obs;

            % iterate through channel snr values
            for channel_db_idx = 1:length(channel_db_values)
                for scheme_idx = 1:length(selected_schemes)
                    scheme = selected_schemes(scheme_idx);
                    
                    % Print statement for at-a-glance performance.
                    disp('')
                    disp("=== " + scheme + " ===")
                    disp("** Agent SNR = " + (agent_db_values(agent_db_idx)) + " dB**")
                    disp("** Channel SNR = " + (channel_db_values(scheme_idx,channel_db_idx,agent_db_idx)) + " dB**")
                    disp("** S = " + S + ", Dropout = " + dropout_vals(dropout_idx) + " **")

                    % Define channel gains and server noise power.
                    if scheme == "MPC"
                        channel_gains = ones(size(gi));
                        gamma_n = dt * P_s * E_mi_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "EPC"
                        channel_gains = abs(gi);
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "PPC"
                        channel_gains = qgi;
                        gamma_n = dt * P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    elseif scheme == "NPC"
                        channel_gains = gi;
                        gamma_n = P_s * E_mi_sqr * E_mag_g_sqr / db2magTen(channel_db_values(scheme_idx,channel_db_idx,agent_db_idx));
                    end

                    % Apply dropout (by setting dropout sensor channels to
                    % zero).
                    if num_dropout > 0
                        dropout_elements = which_dropout(:,1:num_dropout);
                        for item = 1:nDeployments
                            channel_gains(1,dropout_elements(item,:),1,item) = 0;
                            mi(1,dropout_elements(item,:),1,item) = 0;
                        end
                    end
                    
                    % Set server noise power. Note the factor of 1/dt which is
                    % equivalent to applying an anti-aliasing filter.
                    n_psd_constant = 1;
                    scaled_n = sqrt( (gamma_n*n_psd_constant/dt) / 2 ) * n;
                    % Define psd function of n in time domain (td).
                    n_psd_function_td = gamma_n/dt;
                    scaled_n_psd_constant = gamma_n * n_psd_constant;

                    %%% solve optimization problem %%%
                    if experiment == "random_dropout"
                        save_dim = dropout_idx;
                    else
                        save_dim = sensor_idx;
                    end

                    num_vars = length(mi); % Number of variables
                    x0 = [ones(1,2*num_vars)]; % Initial guesses for ai, bi, and t
                    lb = zeros(1,2*num_vars); % ai, bi >= 0
                    ub = []; % unrestricted

                    % sas is scale and shift (ai and bi)
                    sas_nonlcon = @(x) sas_constraints(x, C1i, C2i, T0, P_max);
    
                    options = optimoptions('fmincon','Display','off','Algorithm','sqp');

                    optTic = tic;
                    for dep_idx = 1:nDeployments
                        disp("Optimizing deployment " + dep_idx + "...")
                        % ci will change depending on the channel deployment
                        ci = channel_gains(:,:,:,dep_idx);
                        
                        % for excessively large values of SNR, this seems to take a very long time
                        % to compute
                        if scheme == "EPC"
                            % define magnitude squared of FT of s0(t)
                            mag_sqr_S0 = @(w) 2*pi^2 .* zeroIfnan((1 + cos(w)) ./ (w.^2 - pi^2).^2) + (w == pi) .* (pi^2/2);
                            % define colored noise spectrum V(w)
                            V = @(w, ai) scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0(w) + scaled_n_psd_constant/2;
                            VV = @(w, ai) scaled_w_psd_constant .* sum(ci.^2 .* ai.^2 .* mi.^2, 2) .* mag_sqr_S0(w);
                            % define objective function (CRLB)
                            alpha_objective = @(x) 1./(((sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) * integral(@(w) (mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true));
                            t0_objective = @(x) 1./((alpha_true^2 .* (sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) * integral(@(w) (w .* mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true));

                            eig_alpha = @(x) (((sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) * integral(@(w) (mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true));
                            eig_t0 = @(x) ((alpha_true^2 .* (sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) * integral(@(w) (w .* mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true));

                            % ((sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) *
                            alpha_int = @(x) integral(@(w) (mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true);
                            t0_int = @(x) alpha_true^2 .* integral(@(w) (w .* mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true);
                            cross_int = @(x) alpha_true .* integral(@(w) w .* (mag_sqr_S0(w)).^2 / V(w,x(1:max(sensor_vals))), -Inf, Inf, 'ArrayValued', true);
                            fisher_mat = @(x) ((sum(ci .* x(1:max(sensor_vals)) .* mi.^2)).^2 / (2*pi)) .* [alpha_int(x) cross_int(x); cross_int(x) t0_int(x)];

                            C_weighted = 1;
                            weighted_sum_objective = @(x) alpha_objective(x) + C_weighted.*t0_objective(x);
                        end

                        % calculate unoptimized CRLB using anon functions
                        anon_crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,dep_idx) = alpha_objective(ones(1,10));
                        anon_crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,dep_idx) = t0_objective(ones(1,10));

                        opt_coeffs(1,:,dep_idx) = [ones(1,max(sensor_vals)), zeros(1,max(sensor_vals))];

                        %%% scaling and shifting
                        % compute optimal ai and bi for scaling and shifting
                        [alpha_sas_x_opt, ~] = fmincon(alpha_objective, x0, [], [], [], [], lb, ub, sas_nonlcon, options);
                        sas_opt(1,:,dep_idx) = alpha_sas_x_opt;
                        sas_alpha_crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,dep_idx) = alpha_objective(alpha_sas_x_opt);
                        sas_alpha_crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,dep_idx) = t0_objective(alpha_sas_x_opt);

                        opt_coeffs(2,:,dep_idx) = alpha_sas_x_opt;
    
                        % compute optimal ai and bi for scaling and shifting
                        [t0_sas_x_opt, ~] = fmincon(t0_objective, x0, [], [], [], [], lb, ub, sas_nonlcon, options);
                        sas_opt(2,:,dep_idx) = t0_sas_x_opt;
                        sas_t0_crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,dep_idx) = alpha_objective(t0_sas_x_opt);
                        sas_t0_crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,dep_idx) = t0_objective(t0_sas_x_opt);

                        opt_coeffs(3,:,dep_idx) = t0_sas_x_opt;
                            
                        %%% weighted sum crlb
                        % compute optimal ai and bi for sas using weighted sum objective function
                        for jj = 1:length(C_weighted_values)
                            weighted_sum_objective = @(x) alpha_objective(x) + C_weighted_values(jj).*t0_objective(x);
                            [weighted_sum_sas_x_opt, ~] = fmincon(weighted_sum_objective, x0, [], [], [], [], lb, ub, sas_nonlcon, options);

                            sas_weighted_sum_crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,dep_idx,jj) = alpha_objective(weighted_sum_sas_x_opt);
                            sas_weighted_sum_crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,dep_idx,jj) = t0_objective(weighted_sum_sas_x_opt);
                        end
                    end
                    optRuntime = toc(optTic);
                    disp(floor(optRuntime/60) + " minutes " + mod(optRuntime,60) + " seconds")

                    %% estimate alpha and t0
                    disp("Running estimation for SNR =" + num2str(channel_db_values(channel_db_idx)))
                    for strat_idx = 1:num_strats
                        % ["normal", "sas alpha", "sas t0", "sas weighted sum"];
                        strat_name = all_strats(strat_idx);
                        % find peak index from noisy channel-weighted sum (cws)
                        ai = opt_coeffs(strat_idx,1:S,dep_idx);
                        bi = opt_coeffs(strat_idx,S+1:end,dep_idx);

                        % cws = reshape(compute_cws(reshape(ai .* x_i,K,S,1,nTrials,nDeployments), mi .* sensor_signal(t-ti-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);

                        % Find peak index from noisy channel-weighted sum (cws).
                        cws = reshape(matched_filter_integral(reshape(ai .* xi,K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-reshape(t,1,1,length(t))), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                        % Generate y(t).
                        y = cws - sum(channel_gains .* bi, 2) + pagetranspose(scaled_n);

                        % Search for t0 from index 1 to index of t0_max/dt + 1.
                        % Since we know a priori that the estimate of t0 cannot
                        % exceed t0_max. If we don't impose this restriction then we
                        % could select a t0 value such that the joint estimate for
                        % alpha = 0/0, which will result in NaN.
                        if exp_split(1) == "naive" || scheme == "MPC" || scheme == "EPC"
                            [~,I] = max(real(y(:,1:(t0_max/dt + 1),:,:)));
                        else
                            [~,I] = max(abs(y(:,1:(t0_max/dt + 1),:,:)));
                        end
                        t0_estimates_for_plot = (I-1)*dt;

                        % Set t0 estimates to true or estimated values depending on
                        % experiment.
                        if experiment == "disjoint"
                            t0_estimates = t0_true*ones(1,1,nTrials,nDeployments);
                        else
                            t0_estimates = t0_estimates_for_plot;
                        end

                        % Compute alpha estimates for AC-NAE
                        if exp_split(1) == "naive"
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
                                % Compute y(t0).
                                t0_cws = reshape(matched_filter_integral(reshape(xi,K,S,1,nTrials,nDeployments), reshape(mi .* sensor_signal(t-ti-repmat(t0_true,1,1,K)), K,S,K,1,nDeployments), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                                % All column idx in t0_cws are the same currently,
                                % hence 2nd idx = 1.
                                peak_y_vals =  t0_cws(:,1,:,:) + pagetranspose(scaled_n(1,1,:,:));
                            end
                            % compute alpha estimates
                            alpha_estimates =  real(peak_y_vals) ./ sum(real(channel_gains) .* Ei,2);
                           %%%%% STOPPED HERE on 07-13-2025

                        if scheme ~= "MLE" && exp_split(1) ~= "naive"
                            fs = 1/dt;
                            L = K-1;
                            f = 0:fs/(2*L):fs/2;
                            indet_idx = find(abs(round(f,4)) == 0.5);
                            omega = 2*pi*f;

                            if (scheme == "NPC" || scheme == "PPC")
                                real_channel = real(channel_gains);
                                imag_channel = imag(channel_gains);

                                b = sum(real_channel.*imag_channel.*mi.^2,2) ./ (sum(mi.^2 .* (real_channel.^2 - imag_channel.^2),2));

                                disc = 1 + 4.*b.^2;
                                a1 = (-1 + sqrt(disc)) ./ (2*b);
                                a2 = (-1 - sqrt(disc)) ./ (2*b);
                                a = a1;

                                SEs = ones(1,K,1,nDeployments) .* (2 * pi^2 * (1+cos(omega)) ./ (omega.^2 - pi^2).^2);
                                gamma = 1;

                                mag_sqr_SHdI = 1 ./ (gamma + SEs);

                                for idx=indet_idx
                                    disp('filling indeterminate values')
                                    % fill by L'Hopital's
                                    mag_sqr_SHdI(1,idx,:) = 8*w0^2 / (gamma*8*w0^2 - 2*pi^2*cos(w0));
                                end

                                mag_SHdI = sqrt(mag_sqr_SHdI);

                                [hdI, hdI_matrix] = get_time_domain(mag_SHdI,dt,nDeployments);

                                hdR_matrix = reshape(a,1,1,nDeployments) .* hdI_matrix;

                                hd_matrix = hdR_matrix + 1i*hdI_matrix;

                                pR = scaled_w_psd_constant * sum(mi.^2 .* (a.*real_channel - imag_channel).^2,2);
                                qR = (scaled_n_psd_constant/2 * (a.^2 + 1)) - gamma*pR;

                                pI = scaled_w_psd_constant * sum(mi.^2 .* (real_channel + a.*imag_channel).^2,2);
                                qI = (scaled_n_psd_constant/2 * (a.^2 + 1)) - gamma*pI;

                                SKndR = pR + mag_sqr_SHdI .* qR;
                                SKndI = pI + mag_sqr_SHdI .* qI;

                                if sum(SKndR < 0, "all")
                                    error("SKndR less than 0!")
                                elseif sum(SKndI < 0, "all")
                                    error("SKndI less than 0!")
                                end

                                SQnR = 1 ./ SKndR;
                                SQnI = 1 ./ SKndI;

                                [~,QnR_matrix] = get_time_domain(SQnR,dt,nDeployments);
                                [~,QnI_matrix] = get_time_domain(SQnI,dt,nDeployments);

                                %%% for alpha
                                % aaa represents sum gi int si(tau - t0_estimates) si(tau-t) dtau
                                % each trial will have a different t0 estimate
                                aaa = reshape(compute_cws(mi .* reshape(sensor_signal(t-ti-t0_estimates),K,S,1,nTrials,nDeployments), mi .* sensor_signal(t-ti-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);

                                aaaR = real(aaa);
                                aaaI = imag(aaa);

                                % the d is decorrelation, not derivative
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
                                %%%
                            else
                                % recompute y if alpha is using different ai and bi
                                % cws = reshape(compute_cws(reshape(ai .* x_i,K,S,1,nTrials,nDeployments), mi .* sensor_signal(t-ti-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt),1,K,nTrials,nDeployments);
                                % y =  cws - sum(channel_gains .* bi, 2) + pagetranspose(scaled_n);

                                Beta = 2 * pi^2 * sum(channel_gains.^2 .* ai.^2 .* mi.^2,2);
                                % constants for solving H
                                C1 = Beta .* scaled_w_psd_constant;
                                C2 = scaled_n_psd_constant/2;

                                V = (Beta .* scaled_w_psd_constant .*(1+cos(omega)) ./ (omega.^2 - pi^2).^2) + scaled_n_psd_constant/2;

                                mag_sqr_H = 1./V;

                                for idx=indet_idx
                                    % fill by L'Hopital's
                                    disp('filling indeterminate values')
                                    mag_sqr_H(1,idx,:) = 8*pi^2 / (C2*8*pi^2 + C1);
                                end

                                [~,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments);
                                Omega = compute_cws(ai .* mi .* reshape(sensor_signal(t-ti-t0_estimates),K,S,1,nTrials,nDeployments), mi .* sensor_signal(t-ti-reshape(t,1,1,length(t))), channel_gains, K, S, nTrials, nDeployments, dt);

                                num = dt*dt*pagemtimes(pagemtimes(real(y) + sum(channel_gains .* bi,2),reshape(Qn_matrix,length(t),length(t),1,nDeployments)),pagetranspose(reshape(Omega,1,length(t),nTrials,nDeployments)));
                                % denom is constant for all deployments
                                denom = dt*dt*pagemtimes(pagemtimes(reshape(Omega,1,length(t),nTrials,nDeployments),reshape(Qn_matrix,length(t),length(t),1,nDeployments)),pagetranspose(reshape(Omega,1,length(t),nTrials,nDeployments)));
                            end
                            alpha_estimates = num ./ denom;
                        end

                        empirical_var(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:,strat_idx) = var(alpha_estimates,0,3);
                        empirical_var(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:,strat_idx) = var(t0_estimates_for_plot,0,3);

                        empirical_mse(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:,strat_idx) = mean((alpha_estimates - alpha_true).^2,3);
                        empirical_mse(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:,strat_idx) = mean((t0_estimates_for_plot - t0_true).^2,3);

                        bias(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:,strat_idx) = mean(alpha_estimates,3);
                        bias(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:,strat_idx) = mean(t0_estimates_for_plot,3);
                    end

                    %% Calculate the CRLBs
                    % dsalpha = reshape(squeeze(sum(channel_gains .* sum(mi .* sensor_signal(t-ti-t0_true) .* mi .* sensor_signal(t-ti-reshape(t,1,1,length(t))) * dt,1),2)),K,1,nDeployments);
                    % temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-T_s));
                    % dst0 = reshape(squeeze(sum(channel_gains .* sum(alpha_true .* mi.^2 .* temp .* sensor_signal(t-ti-reshape(t,1,1,length(t))) * dt,1),2)),K,1,nDeployments);
                    % 
                    % if scheme == "MLE" || exp_split(1) == "naive"
                    %     crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = scaled_w_psd_constant / sum(E_i,2);
                    %     temp = -w0*cos(w0*(t-ti-t0_true)).*(heaviside(t-ti-t0_true) - heaviside(t-ti-t0_true-T_s)) + ...
                    %     sin(w0*(t-ti-t0_true)).*-1.*(dirac(t-ti-t0_true) - dirac(t-ti-t0_true-T_s));
                    %     crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = scaled_w_psd_constant / sum(sum((alpha_true .* mi .* temp).^2 * dt,1),2);
                    % elseif (scheme == "NPC" || scheme == "PPC")
                    % 
                    %     %%% for alpha
                    %     dsdR_alpha = dt*(pagemtimes(pagetranspose(real(dsalpha)),hdR_matrix) - pagemtimes(pagetranspose(imag(dsalpha)),hdI_matrix));
                    %     dsdI_alpha = dt*(pagemtimes(pagetranspose(imag(dsalpha)),hdR_matrix) + pagemtimes(pagetranspose(real(dsalpha)),hdI_matrix));
                    % 
                    %     %%% for t0
                    %     dsdR_t0 = dt*(pagemtimes(pagetranspose(real(dst0)),hdR_matrix) - pagemtimes(pagetranspose(imag(dst0)),hdI_matrix));
                    %     dsdI_t0 = dt*(pagemtimes(pagetranspose(imag(dst0)),hdR_matrix) + pagemtimes(pagetranspose(real(dst0)),hdI_matrix));
                    % 
                    %     fisher_matrix = dt*dt*[(pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_alpha)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_alpha)))...
                    %                 (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_t0)));...
                    %                 (pagemtimes(pagemtimes(dsdR_alpha,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_alpha,QnI_matrix),pagetranspose(dsdI_t0)))...
                    %                 (pagemtimes(pagemtimes(dsdR_t0,QnR_matrix),pagetranspose(dsdR_t0)) + pagemtimes(pagemtimes(dsdI_t0,QnI_matrix),pagetranspose(dsdI_t0)))];
                    % 
                    %     inv_zz = pageinv(fisher_matrix);
                    %     crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz(1,1,:);
                    %     crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz(2,2,:);
                    % else
                    %     fisher_matrix = dt*dt*[pagemtimes(pagemtimes(pagetranspose(dsalpha),Qn_matrix),dsalpha), pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dsalpha);...
                    %         pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dsalpha), pagemtimes(pagemtimes(pagetranspose(dst0),Qn_matrix),dst0)];
                    %     inv_zz = pageinv(fisher_matrix);
                    %     crlb(1,agent_db_idx,channel_db_idx,1,scheme_idx,save_dim,:) = inv_zz(1,1,:);
                    %     crlb(1,agent_db_idx,channel_db_idx,2,scheme_idx,save_dim,:) = inv_zz(2,2,:);
                    % end
                end
                disp('')
            end
        end
    end
end

runtime = toc(loopTic);
disp(floor(runtime/60) + " minutes " + mod(runtime,60) + " seconds")

%% plot optimal scale ai (bi not used)
% param_text = ["\alpha", "t_0"];
% for param_idx = 1:2
%     fig1 = figure;
%     set(fig1,'Position',[100,100,450,450])
% 
%     f = tiledlayout(min(nDeployments,3),1);
%     f.Padding = "compact";
%     for jj = 1:min(nDeployments,3)
%         x_vals = mi.^2 .* ci(:,:,jj);
%         nexttile
%         % Stem plot for optimal scale ai
%         stem(x_vals, scale_opt(param_idx,1:S,jj), 'o', 'Color', '#77AC30', 'LineWidth', 2, 'MarkerSize', 8);
%         xlabel('m_i^2 \cdot c_i');
%         ylabel('a_i');
%         grid on;
%     end
%     sgtitle("Scale Coefficients for " + param_text(param_idx) + ", CSNR = " + channel_db_values(channel_db_idx))
% end

%% plot optimal sas ai and bi
% for param_idx = 1:2
%     fig1 = figure;
%     set(fig1,'Position',[100,100,450,450])
% 
%     f = tiledlayout(min(nDeployments,3),1);
%     f.Padding = "compact";
%     tiledlayout(min(nDeployments,3),2);
%     % Stem plot for optimal sas ai
%     for jj = 1:min(nDeployments,3)
%         x_vals = mi.^2 .* ci(:,:,jj);
%         nexttile
%         stem(x_vals, sas_opt(param_idx,1:S,jj), 'bo', 'LineWidth', 2, 'MarkerSize', 8);
%         xlabel('m_i^2 \cdot c_i');
%         xlim([0,max(x_vals*1.1)])
%         ylabel('Optimal a_i');
%         grid on;
% 
%         % Stem plot for optimal sas bi
%         nexttile
%         stem(x_vals, sas_opt(param_idx,S+1:end,jj), 'ro', 'LineWidth', 2, 'MarkerSize', 8);
%         xlabel('m_i^2 \cdot c_i');
%         xlim([0,max(x_vals*1.1)])
%         ylabel('Optimal b_i');
%         grid on;
%     end
%     sgtitle("SaS Coefficients for " + param_text(param_idx) + ", CSNR = " + channel_db_values(channel_db_idx))
% end

%% avg linear plots
avg_dep_var = mean(empirical_var, 5);
avg_dep_mse = mean(empirical_mse, 5);

avg_anon_crlb = mean(anon_crlb, 5);

avg_sas_alpha_crlb = mean(sas_alpha_crlb, 5);
avg_sas_t0_crlb = mean(sas_t0_crlb, 5);
avg_sas_weighted_sum_crlb = mean(sas_weighted_sum_crlb, 5);

params_latex = ["\alpha","t_0"];
params_text = ["alpha","t0"];
color_vec = ["#4F4F4F", "#A52A2A", "#2E7D32", "#1565C0", "#D95319",...
          "#D4AF37", "#8E24AA", "#4DBEEE"];

sensor_vals_for_plot = sensor_vals;

for agent_db_idx = 1:length(agent_db_values)
    for param_idx = 1:2
        all_vals = [];
        for strat_idx = 1:num_strats
            all_vals = [all_vals;...
                        avg_dep_var(:,agent_db_idx,:,param_idx,:,:,:,strat_idx);...
                        avg_dep_mse(:,agent_db_idx,:,param_idx,:,:,:,strat_idx)];
        end
        all_vals = [flat(avg_sas_alpha_crlb(:,agent_db_idx,:,param_idx,:,:))...
                    flat(avg_sas_t0_crlb(:,agent_db_idx,:,param_idx,:,:))...
                    flat(avg_sas_weighted_sum_crlb(:,agent_db_idx,:,param_idx,:))];
        
        for scheme_idx = find(~strcmp(selected_schemes,"MLE")) 
            fig1 = figure;
            set(fig1,'Position',[100,100,750,750])

            nexttile
            hold on

            if ismatrix(channel_db_values)
                channel_db_for_plot = channel_db_values;
            else
                channel_db_for_plot = channel_db_values(scheme_idx,:,agent_db_idx);
            end

            symbol_vec = [repmat("o",1,3),repmat("square",1,length(C_weighted_values))];

            plot(nan,nan,'-^','color','black');
            plot(nan,nan,'-x','color','black');
            for jj = 1:num_strats
                plot(nan,nan,["--"+symbol_vec(jj)],'color',color_vec(jj),'LineWidth',2);
            end

            for sensor_idx = 1:length(sensor_vals_for_plot)
                for strat_idx = 1:num_strats
                    plot(channel_db_for_plot,(squeeze(avg_dep_var(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,:,strat_idx))),'-^','color',color_vec(strat_idx),'LineWidth', 1)
                    plot(channel_db_for_plot,(squeeze(avg_dep_mse(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,:,strat_idx))),'-x','color',color_vec(strat_idx),'LineWidth', 1)
                end
                plot(channel_db_for_plot,(squeeze(avg_anon_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'--o','color',color_vec(1),'LineWidth', 1)
                plot(channel_db_for_plot,(squeeze(avg_sas_alpha_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'--o','color',color_vec(2),'LineWidth', 1)
                plot(channel_db_for_plot,(squeeze(avg_sas_t0_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1))),'--o','color',color_vec(3),'LineWidth', 1)
                
                for ii = 1:length(C_weighted_values)
                    plot(channel_db_for_plot,(squeeze(avg_sas_weighted_sum_crlb(:,agent_db_idx,:,param_idx,scheme_idx,sensor_idx,1,ii))),'--square','color',color_vec(ii+3),'LineWidth', 1)
                end
                
            end

            if experiment == "agent_server_noise"
                xlabel('$$\Gamma$$','Interpreter','latex')
            else
                xlabel('Channel SNR (dB)')
            end
            ylabel("CRLB")

            % legend(all_strats, 'Location', 'best');
            for jj = 1:length(C_weighted_values)
                temp_legend(jj) = "CW" + floatToFraction(C_weighted_values(jj));
            end

            legend(["VAR","MSE",all_strats], 'Location', 'best');

            % title('$$\hat{'+params(jj)+'}$$, ' + selected_schemes(scheme_idx),'Interpreter','latex')
            title('SaS $$\hat{'+params_latex(param_idx)+'}$$','Interpreter','latex')
            set(gca, 'FontSize', 15);
            set(gca, 'YScale', 'log')

            % ylim([10^(-0.25)*min(all_vals,[],"all"),1.05*max(all_vals,[],"all")]);
            % 
            % Save the figure
            % exportgraphics(fig1, fullfile(save_folder, sprintf(experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
        end
    end
end

%% Functions
function fractionStr = floatToFraction(x)
    % Convert a floating-point number to its fraction representation as a string
    
    % Use the rat function to approximate the fraction
    [num, den] = rat(x);
    
    % Convert to string format
    if x >= 1
        if den == 1
            fractionStr = string(num);
        else
            fractionStr = num2str(num) + "/" + num2str(den);
        end
    else
        fractionStr = num2str(x);
    end
end

function y = zeroIfnan(x)
    y = x;
    y(isnan(x)) = 0;
end

% define constraints for optimization problem
function [c, ceq] = sas_constraints(x, C1i, C2i, T0, P_max)
    num_vars = length(C1i);
    ai = x(1:num_vars);
    bi = x(num_vars+1:2*num_vars);

    ind_power_const = C1i/T0;
    ind_power_vals = ai.^2 .* C1i ./ T0 - 2 .* ai .* bi .* C2i ./ T0 + bi.^2;

    % First constraint: sum(ai.^2 .* C1i ./ T0 - 2 .* ai .* bi .* C2i ./ T0 + bi.^2) <= P_max
    c1 = sum(ind_power_vals) - 0.8*P_max;
    c_ind = (ind_power_vals-ind_power_const)';

    c = [c1;c_ind];
    ceq = [];
end

% function [out] = compute_cws(tempA, tempB, channel_gains, K, S, nTrials, nDeployments, dt)
%     tempC = pagemtimes(pagetranspose(tempA),tempB);
% 
%     diag_idx = sub2ind(size(tempC),1:S,1:S);
%     KTD = K*nTrials*nDeployments;
%     all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);
% 
%     tempD = tempC(all_diag_idx)*dt;
%     out = pagemtimes(tempD,reshape(channel_gains,S,1,1,1,nDeployments));
%     % disp()
% end

function [out] = matched_filter_integral(tempA, tempB, channel_gains, K, S, nTrials, nDeployments, dt)
% MATCHED_FILTER_INTEGRAL Returns the matched filter output of tempA and tempB,
% which is computed in a vectorized manner.
    tempC = pagemtimes(pagetranspose(tempA),tempB);

    diag_idx = sub2ind(size(tempC),1:S,1:S);
    KTD = K*nTrials*nDeployments;
    all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);

    tempD = tempC(all_diag_idx)*dt;
    out = pagemtimes(tempD,reshape(channel_gains,S,1,1,1,nDeployments));
end

% function [out] = calculate_m_t(tempA, tempB, m_sensors, K, S, nTrials, nDeployments, dt)
%     tempC = pagemtimes(pagetranspose(tempA),tempB);
% 
%     diag_idx = sub2ind(size(tempC),1:S,1:S);
%     KTD = K*nTrials*nDeployments;
%     all_diag_idx = diag_idx + S^2 *reshape(0:(KTD-1),1,1,K,nTrials,nDeployments);
% 
%     tempD = tempC(all_diag_idx)*dt;
%     out = pagemtimes(tempD,reshape(m_sensors,S,1,1,1,nDeployments));
%     % disp()
% end

% function [Qn,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments)
%     mag_sqr_H_2 = [mag_sqr_H(1,1,:) mag_sqr_H(1,2:end,:) fliplr(conj(mag_sqr_H(1,2:end,:)))];
%     Qn = ifft(mag_sqr_H_2/dt,[],2);
%     L = floor(length(Qn)/2);
%     mirrored_Qn = [Qn(1,L+2:end,:), Qn(1,1:L+1,:)];
%     Qn_matrix = zeros(L+1,L+1,nDeployments);
%     for i = 1:(L+1)
%         Qn_matrix(i,:,:) = mirrored_Qn(1,L+1-(i-1):end-(i-1),:);
%     end
% end

function [Qn,Qn_matrix] = get_time_domain(mag_sqr_H,dt,nDeployments)
% GET_TIME_DOMAIN Returns the time-domain matrix of a given frequency domain
% vector.
    mag_sqr_H_2 = [mag_sqr_H(1,1,:) mag_sqr_H(1,2:end,:) fliplr(conj(mag_sqr_H(1,2:end,:)))];
    Qn = ifft(mag_sqr_H_2/dt,[],2);
    L = floor(size(Qn,2)/2);
    mirrored_Qn = [Qn(1,L+2:end,:), Qn(1,1:L+1,:)];
    Qn_matrix = zeros(L+1,L+1,nDeployments);
    for i = 1:(L+1)
        Qn_matrix(i,:,:) = mirrored_Qn(1,L+1-(i-1):end-(i-1),:);
    end
end

function [qg,cq] = quantize_comp(g,M)
    phi = angle(g);
    phi = phi + (2*pi)*(phi<0);
    
    all_cq = 0;

    for l = 1:M
        if l == 1
            cq_temp = exp(-(l-1)*1i*2*pi/M) * ((phi > (2*M-1)*pi/M) | (phi <= pi/M));
        else
            cq_temp = exp(-(l-1)*1i*2*pi/M) * ((phi > (2*(l-1)-1)*pi/M) & (phi <= (2*l-1)*pi/M));
        end
        all_cq = all_cq + cq_temp;
    end
    qg = g .* all_cq;
    cq = all_cq;
end