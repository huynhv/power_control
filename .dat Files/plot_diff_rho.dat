close all
clear all

load("../cwe_disjoint_peak_power_opt_diff_rho_results.mat")
save_images = true;

params_latex = ["\hat{\alpha}","\hat{t}_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr(unique(iter_arr)), colors(1:numel(iter_arr)));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Peak Power AirComp Results\";

exclude = all_strats(contains(all_strats, "hetero.") & (contains(all_strats, "asymp. approx.") | contains(all_strats, "prop. scaling"))); % ["None"];

for agent_db_idx = 1:length(agent_db_values)
    selected_strat_idxs = find((1:numel(all_strats)) .* ~ismember(all_strats,exclude).');
    selected_strats = all_strats(selected_strat_idxs);
    channel_db_idx = 1; % only considering one channel SNR value here

    max_rho = 1.5;
    accepted_rho_idxs = rho_vals <= max_rho;

    for param_idx = 1:2
        % normalize performance metrics
        if param_idx == 1
            norm_coeff = alpha_true^2;
        else
            norm_coeff = t0_true^2;
        end

        avg_dep_crlb(selected_strat_idxs,agent_db_idx,:,param_idx,:,:) = avg_dep_crlb(selected_strat_idxs,agent_db_idx,:,param_idx,:,:) ./ norm_coeff;
        avg_dep_var(selected_strat_idxs,agent_db_idx,:,param_idx,:,:) = avg_dep_var(selected_strat_idxs,agent_db_idx,:,param_idx,:,:) ./ norm_coeff;
        avg_dep_mse(selected_strat_idxs,agent_db_idx,:,param_idx,:,:) = avg_dep_mse(selected_strat_idxs,agent_db_idx,:,param_idx,:,:) ./ norm_coeff;
        
        % Collect all plot values for y-axis scaling.
        all_vals = [avg_dep_mse(selected_strat_idxs,agent_db_idx,channel_db_idx,param_idx,:,accepted_rho_idxs)];

        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);

            count_vals = rho_vals;

            for constraint_idx = 1:length(constraints)
                constraint = constraints(constraint_idx);
                data_idx_seq = selected_strat_idxs(find(contains(selected_strats, constraint)).');

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

                plot(ax, nan,nan,'x','color','black');

                x_axis_series = rho_vals(accepted_rho_idxs);
                plot_line_width = 1.5;

                for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                    strat = all_strats(strat_idx);
                    split_str = split(strat);
                    iter_key = join(split_str(2:end), " ");

                    plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,accepted_rho_idxs,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
                end

                xlabel('$$\rho$$','Interpreter','latex')
                ylabel(" ")

                title('$$'+params_latex(param_idx)+'$$','Interpreter','latex')

                ax.FontSize = 15;          % font size
                ax.YScale = 'log';          % log scale if needed
                grid(ax, 'on');             % turn on grid

                legend(ax, [lgd, "MSE"], 'Location', 'best')

                grid on

                y_lower = 10^(-0.1)*min(all_vals,[],"all");
                y_upper = (1.05)*max(all_vals,[],"all");
                ylim([y_lower,y_upper]);

                % Save the figure.
                if save_images
                    save_folder = folder_path + "diff_rho";
                    if ~exist(save_folder, 'dir')
                        mkdir(save_folder);
                    end
                    exportgraphics(fig1, fullfile(save_folder, sprintf(constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                end
            end
        end
    end
end