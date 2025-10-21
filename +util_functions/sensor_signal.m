function s = sensor_signal(t, Ts)
    % Ts = 1;
    B = 1/(2*Ts);
    s = sin(B*2*pi*t);
    % s = ones(size(t));
    % s(or(t<0,t>T_s)) = 0;
    s(t<0 | t>Ts) = 0;
    % s(t==0) = s(t==0)*0.5;
    % s(t == T_s) = s(t == T_s)*0.5;
end
