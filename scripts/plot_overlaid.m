close all
clear all

load('../cwe_disjoint_peak_power_opt_results.mat')
save_images = true;

params_latex = ["\hat{\alpha}","\hat{t}_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr([unique(iter_arr)]), colors(1:numel([unique(iter_arr)])));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Peak Power AirComp Results\";

exclude = all_strats(contains(all_strats, "hetero.") & contains(all_strats, "prop. scaling")); % ["None"];

for agent_db_idx = 1:length(agent_db_values)
    selected_strat_idxs = find((1:numel(all_strats)) .* ~ismember(all_strats,exclude).');
    selected_strats = all_strats(selected_strat_idxs);
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
        all_vals = [avg_dep_crlb(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                    avg_dep_var(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension);
                    avg_dep_mse(selected_strat_idxs,agent_db_idx,:,param_idx,:,1:sensor_dimension)
                    ];

        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);

            count_vals = sensor_vals;

            for constraint_idx = 1:length(constraints)
                constraint = constraints(constraint_idx);
                data_idx_seq = selected_strat_idxs(find(contains(selected_strats, constraint)).');
                % Plot pseudoplots for legend symbols.

                if constraint == "indiv."
                    figLgd = figure;
                    axLgd = gca;
                    plot_w = 450;
                    set(figLgd,'Position',[100,100,plot_w,plot_w])
                    hold(axLgd, 'on');

                    lgd = [];

                    % Plot pseudoplots for legend symbols.
                    for count_idx = data_idx_seq
                        strat = all_strats(count_idx);
                        split_str = split(strat);
                        iter_key = join(split_str(2:end), " ");

                        lgd = [lgd iter_key];

                        plot(axLgd, nan,nan,'color',colorMap(iter_key),'LineWidth',2);
                    end

                    plot(axLgd, nan,nan,'x','color','black');
                    plot(axLgd, nan,nan,'^','color','black');
                    plot(axLgd, nan,nan,'o','color','black');
                    
                    lgdObj = legend(axLgd, [lgd, "MSE", "VAR", "CRLB"]);
                    lgdObj.Location = 'none';
                    lgdObj.Units = 'normalized';
                    lgdObj.Position(1) = (1 - lgdObj.Position(3)) / 2;  % center horizontally
                    lgdObj.Position(2) = (1 - lgdObj.Position(4)) / 2;  % center vertically
                    
                    lgdObj.FontSize = 15;        % increase text size
                    lgdObj.ItemTokenSize = [30 18];  % increase marker/line size [width height]
                    axLgd.Color = 'none';
                    axLgd.XColor = 'none';
                    axLgd.YColor = 'none';
                    figLgd.Color = 'white';
                    drawnow;
                end

                fig1 = figure;
                plot_w = 450;
                set(fig1,'Position',[100,100,plot_w,plot_w])
                ax = gca;
                hold(ax, 'on');

                x_axis_series = channel_db_values(scheme_idx,:,agent_db_idx);
                plot_line_width = 1.5;

                for count_idx = 1:length(count_vals)
                    for strat_idx = data_idx_seq % (constraint_idx-1)*iter_length + (1:iter_length) %%% num_strats
                        strat = all_strats(strat_idx);
                        split_str = split(strat);
                        iter_key = join(split_str(2:end), " ");

                        plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-x','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_var(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'-^','color',colorMap(iter_key),'LineWidth', plot_line_width)
                        plot(ax, x_axis_series,(squeeze(avg_dep_crlb(strat_idx,agent_db_idx,:,param_idx,scheme_idx,count_idx,1))),'--o','color',colorMap(iter_key),'LineWidth', plot_line_width)
                    end
                end

                xlabel('Channel SNR (dB)')
                ylabel(" ")

                title('$$'+params_latex(param_idx)+'$$','Interpreter','latex')

                ax.FontSize = 15;          % font size
                ax.YScale = 'log';          % log scale if needed
                grid(ax, 'on');             % turn on grid

                y_lower = 10^(-0.25)*min(all_vals,[],"all");
                y_upper = 1.05*max(all_vals,[],"all");
                ylim([y_lower,y_upper]);

                ax.XTick = channel_snr;

                % Save the figure.
                if save_images
                    save_folder = folder_path + "overlaid";
                    if ~exist(save_folder, 'dir')
                        mkdir(save_folder);
                    end
                    exportgraphics(fig1, fullfile(save_folder, sprintf("overlaid" + "_" + constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                    if constraint == "indiv."
                        exportgraphics(figLgd, fullfile(save_folder, sprintf("legend_" + "overlaid" + "_" + constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                    end
                end
            end
        end
    end
end