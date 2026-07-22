close all
clear all

load("../cwe_disjoint_peak_power_opt_diff_weights_results.mat")
save_images = true;

params_latex = ["\hat{\alpha}","\hat{t}_0"];
params_text = ["alpha","t0"];

RGB = orderedcolors("gem12");
H = compose("#%02X%02X%02X",round(RGB*255));
colors = cellstr(H);
colorMap = containers.Map(cellstr(unique(iter_arr)), colors(1:numel(iter_arr)));

home_dir = char(java.lang.System.getProperty('user.home'));
folder_path = home_dir + "\Desktop\Peak Power AirComp Results\";

exclude = all_strats(contains(all_strats, "prop. scaling") | contains(all_strats, "hetero. linear opt.") | contains(all_strats, "total") | contains(all_strats, "mixed")); % ["None"];

for agent_db_idx = 1:length(agent_db_values)
    selected_strat_idxs = find((1:numel(all_strats)) .* ~ismember(all_strats,exclude).');
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
        all_vals = [avg_dep_mse(selected_strat_idxs,agent_db_idx,channel_db_idx,param_idx,:,find(contains(constraints, "indiv.")))];
        avg_vals = mean(squeeze(all_vals),2);

        for scheme_idx = 1:length(selected_schemes)
            scheme = selected_schemes(scheme_idx);
            for constraint_idx = find(contains(constraints, "indiv.")) % 1:length(constraints)
                constraint = constraints(constraint_idx);
                data_idx_seq = selected_strat_idxs(find(contains(selected_strats, constraint)).');

                % Plot pseudoplots for legend symbols.
                fig1 = figure;
                fig_x = 100;
                fig_y = 100;
                fig_w = 450;
                fig_h = 450;

                set(fig1,'Position',[fig_x, fig_y, fig_w, fig_h],'Name',constraint)
                t = tiledlayout(fig1, 1, 1);  % 1x1 layout just to get access to Layout-aware legend placement
                ax = nexttile(t);
                hold(ax, 'on');

                inset_x = 0.50;
                inset_y = 0.45;
                inset_w = 0.35;
                inset_h = 0.28;

                inset_bl = [inset_x*fig_w, inset_y*fig_h];
                inset_br = [(inset_x+inset_w)*fig_w, inset_y*fig_h];
                inset_tl = [inset_x*fig_w, (inset_y+inset_h)*fig_h];
                inset_tr = [(inset_x+inset_w)*fig_w, (inset_y+inset_h)*fig_h];

                axInset = axes('Position', [inset_x, inset_y, inset_w, inset_h]);
                hold(axInset, 'on')

                lgd = [];

                for pivot_idx = 1:numel(pivot_vals)
                    % lgd = [lgd pivot_vals(pivot_idx)];
                    lgd = [lgd "$$w_{t_0}^{(" + pivot_idx + ")}$$"];
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
                        plot(ax, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,ii))),'-x','color',RGB(ii,:),'LineWidth', plot_line_width)
                        plot(axInset, x_axis_series,(squeeze(avg_dep_mse(strat_idx,agent_db_idx,channel_db_idx,param_idx,scheme_idx,ii))),'-x','color',RGB(ii,:),'LineWidth', plot_line_width)
                    end
                end

                xlabel(ax, 'Channel SNR')
                ylabel(" ")

                title('$$'+params_latex(param_idx)+'$$','Interpreter','latex')

                ax.FontSize = 15;          % font size
                ax.YScale = 'log';          % log scale if needed
                grid(ax, 'on');             % turn on grid

                lgd_handle = legend(ax, [lgd, "MSE"], 'Interpreter', 'latex', 'Orientation', 'horizontal');
                lgd_handle.Layout.Tile = 'north';
                lgd_handle.ItemTokenSize = [15, 18];

                x0 = 12;
                y0 = avg_vals(end)*0.985;
                w0 = 7;
                h0 = avg_vals(3)-avg_vals(end);

                axPos = ax.Position; % [left, bottom, width, height] of axes in figure
                xLimits = ax.XLim;
                yLimits = ax.YLim;

                dataX = x0;
                dataY = y0;
                dataW = w0;
                dataH = h0;

                % 3. Convert position (X, Y) - Includes the axes offset
                normX = axPos(1) + ((dataX - xLimits(1)) / (xLimits(2) - xLimits(1))) * axPos(3);
                normY = 0.025 + axPos(2) + ((dataY - yLimits(1)) / (yLimits(2) - yLimits(1))) * axPos(4);

                % 4. Convert size (Width, Height) - Scales without the offset
                normW = (dataW / (xLimits(2) - xLimits(1))) * axPos(3);
                normH = (dataH / (yLimits(2) - yLimits(1))) * axPos(4);

                normROI = [normX, normY, normW, normH];

                rect_bl = [x0, y0];
                rect_br = [x0+w0, y0];
                rect_tl = [x0, y0+h0];
                rect_tr = [x0+w0, y0+h0];

                normROI_bl = [normX, normY];
                normROI_br = [normX+normW, normY];
                normROI_tl = [normX, normY+normH];
                normROI_tr = [normX+normW, normY+normH];

                % Account for the actual visual bounds of the plot including labels
                ax_outer = ax.OuterPosition;   % includes labels/ticks, not just the axes box
                ax_center_x = ax_outer(1) + ax_outer(3)/2;
                lgd_width = lgd_handle.Position(3);
                lgd_handle.Position(1) = ax_center_x - lgd_width/2;

                y_lower = 10^(-0.01)*min(all_vals,[],"all");
                y_upper = 1.03*max(all_vals,[],"all");
                ylim(ax, [y_lower,y_upper]);

                xlim(axInset, [x0, x0+w0])
                box(axInset, 'on')
                grid(axInset, 'on')

                % 5. Draw the annotation using the normalized coordinates
                annotation('rectangle', normROI, 'Color', 'k', 'LineWidth', 1);
                annotation(fig1, 'line', [normROI_tl(1), inset_bl(1)/fig_w], [normROI_tl(2), inset_bl(2)/fig_h], ...
                    'Color', 'k', 'LineStyle', ':', 'LineWidth', 1);
                annotation(fig1, 'line', [normROI_tr(1), inset_br(1)/fig_w], [normROI_tr(2), inset_br(2)/fig_h], ...
                    'Color', 'k', 'LineStyle', ':', 'LineWidth', 1);

                % Save the figure.
                if save_images
                    save_folder = folder_path + "diff_weights";
                    if ~exist(save_folder, 'dir')
                        mkdir(save_folder);
                    end
                    exportgraphics(fig1, fullfile(save_folder, sprintf(constraint + "_" + experiment + "_" + string(agent_db_values(agent_db_idx)) + "_db_Tp=" + string(Tp) + "_" + params_text(param_idx) + "_" + selected_schemes(scheme_idx) + ".png")), 'Resolution', 300);
                end
            end
        end
    end
end