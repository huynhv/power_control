function s = sensor_signal(t, Tp, norm_fact)
    % Ts = 1;
    B = pi/Tp;
    s = sin(B*t);
    % s = ones(size(t));
    % s(or(t<0,t>T_s)) = 0;
    s(t<0 | t>=Tp) = 0;
    % s(t==0) = s(t==0)*0.5;
    % s(t == T_s) = s(t == T_s)*0.5;
    s = s*norm_fact;
end
