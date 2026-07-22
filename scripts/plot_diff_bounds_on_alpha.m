close all
clear all

identifiers = ["^", ">", "v", "<", "*"];

load("../cwe_disjoint_peak_power_opt_diff_bounds_on_alpha_results.mat")
save_images = true;

params_latex = ["\hat{\alpha}","\hat{t}_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr(unique(iter_arr)), colors(1:numel(iter_arr)));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Peak Power AirComp Results\";

% exclude = all_strats(contains(all_strats, "hetero.") & contains(all_strats, "asymp. approx.")); % ["None"];

for agent_db_idx = 1:length(agent_db_values)
    selected_strat_idxs = find(contains(all_strats, "hetero. affine opt.")); % find((1:numel(all_strats)) .* ~ismember(all_strats,exclude).');
    selected_strats = all_strats(selected_strat_idxs);
    channel_db_idx = 1:5;
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
        all_vals = [avg_dep_mse(selected_strat_idxs,agent_db_idx,channel_db_idx,param_idx,:,:)
                    % avg_dep_crlb(selected_strat_idxs,agent_db_idx,channel_db_idx,param_idx,:,1:length(rho_vals));
                    % avg_dep_var(selected_strat_idxs,agent_db_idx,channel_db_idx,param_idx,:,1:length(rho_vals));
                    ];

        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);
            for constraint_idx = 1:length(constraints)
                constraint = constraints(constraint_idx);
                data_idx_seq = selected_strat_idxs(find(contains(selected_strats, constraint)).');
                % Plot pseudoplots for legend symbols.

                fig1 = figure;
                set(fig1,'Position',[100,100,450,450])
                ax = gca;
                hold(ax, 'on');

                lgd = [];

                for pivot_idx = 1:numel(pivot_vals)
                    lgd = [lgd "$$\beta = " + pivot_vals(pivot_idx) + "$$"];
                    plot(ax, nan,nan,'color',RGB(pivot_idx,:),'LineWidth',2);
                end

                plot(ax, nan,nan,'x','color','black');

                x_axis_series = channel_db_values;
                plot_line_width = 1.5;

                for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                    strat = all_strats(strat_idx);
                    split_str = split(strat);
                    iter_key = join(split_str(2:end), " ");

                    for ii = 1:numel(pivot_vals)
                        % plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,:,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,ii))),'-x','color',RGB(ii,:),'LineWidth', plot_line_width)
                        % plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,:,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
                    end
                end

                xlabel('Channel SNR','Interpreter','latex')
                ylabel(" ")

                title('$$'+params_latex(param_idx)+'$$','Interpreter','latex')

                ax.FontSize = 15;          % font size
                ax.YScale = 'log';          % log scale if needed
                grid(ax, 'on');             % turn on grid

                legend(ax, [lgd, "MSE"], 'Location', 'best','Interpreter','latex')

                grid on

                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 1.05*max(all_vals,[],"all");
                ylim([y_lower,y_upper]);

                % Save the figure.
                if save_images
                    save_folder = folder_path + "diff_bounds_on_alpha";
                    if ~exist(save_folder, 'dir')
                        mkdir(save_folder);
                    end
                    exportgraphics(fig1, fullfile(save_folder, sprintf(constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                end
            end
        end
    end
end